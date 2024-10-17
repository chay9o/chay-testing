import time
import gc
from celery import shared_task
import json
import tempfile
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from django.core.files.storage import default_storage
from django.conf import settings
import os
import re
import psutil
import base64
from io import BytesIO
from jinja2 import Template
from django.http import JsonResponse
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from together import Together
import logging

chat_template = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['text'] + ' [/INST]' }}"
    "{% elif message['role'] == 'bot' %}{{ message['text'] + eos_token }}"
    "{% endif %}{% endfor %}"
)


# Function to generate the chat history string
def render_chat_history(messages):
    template = Template(chat_template)
    data = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "messages": messages
    }
    return template.render(data)

@shared_task
def process_prompts1A(final_text, language):
    # Maintain the chat history
    chat_history = []

    for i in range(1):
        text_template = get_text_template(i)  # A function to return the corresponding template
        user_input = text_template.format(final_text=final_text, language=language)

        # Append user input to chat history
        chat_history.append({"role": "user", "text": user_input})

        # Render the chat history string using the template
        chat_history_str = render_chat_history(chat_history)

        # Call the process_user_input function to process the user input and chat history
        answer = process_user_input(user_input, chat_history_str)

        # Append the bot's answer to the chat history
        chat_history.append({"role": "bot", "text": answer})

        # Print or return the answer (to make sure it's passed back correctly)
        print(f"Iteration {i+1}: {user_input}, Answer: {answer}")

    return "All iterations completed"

def get_text_template(iteration):
    if iteration == 0:
        return f"""
        Based on the following text, create a new version of this text that gives an improved narrative with better flow between ideas. 
        You are a very strategic person. If needed, also reorder ideas. Make it extensive. This is just the introduction of a report (we call it Strategic Insights document) on the situation. 
        The situation below is happening these days. The situation is happening in our company.

        Follow these rules:

        1. Sentence Structure: Use a mix of sentence lengths.
           Short sentences: To emphasize points.
           Longer sentences: To explain or elaborate.

        2. Vocabulary: Use clear and straightforward language.
           Avoid: Technical jargon or complex vocabulary unless necessary.
           Use: Everyday language that is easy to understand.

        3. All the following text is happening in our company.

        4. Provide just the text, no what it was improved.

        Remember this will be part of a report written by John.

        Text to rewrite (keep a similar writing style but improved). Add a title for this Strategic Insight: 
        {{final_text}}

        Format:
        Title
        Description
        """
    elif iteration == 1:
        return f"""
        Template for iteration 2: Add your own text here.
        """
    # Add more templates for other iterations as needed
    else:
        return f"""
        Default template for iteration {iteration}: Add your own text here.
        """


def process_user_input(combined_input, chat_history):
    TOGETHER_API_KEY = settings.TOGETHER_API_KEY
    client = Together(api_key=TOGETHER_API_KEY)
    
    #MODEL_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    max_tokens = 8192

    # This is the OpenAI chat completion client call
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": combined_input},
            {"role": "system", "content": chat_history}
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True
    )

    # Process the streamed response
    generated_text = ""
    
    for chunk in response:
        if len(chunk.choices) > 0:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                if chunk.choices[0].message.content:
                    generated_text += chunk.choices[0].message.content
        else:
            logger.info(f"CHUNK HAS NO CHOICES: {chunk.choices}")

    # Return the generated text back to process_prompts1
    return generated_text


logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.GPT_MODEL_3,
    openai_api_base=settings.BASE_URL,
    max_tokens=4096,
    streaming=True,
    top_p=0.9
)
llm1 = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.GPT_MODEL_2,
    openai_api_base=settings.BASE_URL,
    max_tokens=4096,
    streaming=True,
    top_p=0.9
)
llm2 = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.GPT_MODEL_3,
    openai_api_base=settings.BASE_URL,
    max_tokens=2048,
    streaming=True,
    top_p=0.9
)

def clean_chat_history(chat_history_raw):
    """Function to clean and convert chat history."""
    return [{"role": msg["role"], "text": msg["text"]} for msg in chat_history_raw]

def set_font(para, font_name='Arial', font_size=Pt(18), bold=False, italic=False, color=RGBColor(255, 255, 255)):
    """Function to set text formatting."""
    for run in para.runs:
        font = run.font
        font.name = font_name
        font.size = font_size
        font.bold = bold
        font.italic = italic
        font.color.rgb = color

def to_sentence_case(text):
    """Function to convert text to sentence case, preserving leading punctuation and quotes."""
    if not text:
        return ""
    # Match leading punctuation and the first alphabetical character
    match = re.match(r'^([^\w"]*["]*)(\w)(.*)', text)
    if match:
        leading_punct, first_char, rest = match.groups()
        return f"{leading_punct}{first_char.upper()}{rest.lower()}"
    else:
        return text


def replace_placeholders(slide, replacements):
    """Function to replace placeholders in a slide."""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for key, value in replacements.items():
                if key in paragraph.text:
                    if key == "{Quote}":
                        value = to_sentence_case(value)  # Convert quote to sentence case
                    paragraph.text = paragraph.text.replace(key, value)
                    if key == "{Title}":
                        set_font(paragraph, font_size=Pt(36), bold=True)
                    elif key == "{Description}":
                        set_font(paragraph, font_size=Pt(22))
                    elif key.startswith("{Bullet Point"):
                        set_font(paragraph, font_size=Pt(20))
                    elif key == "{Name}":
                        set_font(paragraph, font_size=Pt(24))
                    elif key == "{Quote}":
                        set_font(paragraph, font_size=Pt(26), italic=True)


def parse_ai_generated_content(content):
    """Function to parse AI-generated content into a structured format."""
    if not content.strip():
        raise ValueError("AI-generated content is empty")
    slides_content = []
    current_slide = None
    smartnote_title = ""
    smartnote_description = ""

    capturing_smartnote_title = False
    capturing_smartnote_description = False

    for line in content.split("\n"):
        line = line.strip()
        #print(f"Processing line: {line}")  

        if line.startswith("[Slide"):
            if current_slide:
                slides_content.append(current_slide)
            current_slide = {"title": "", "description": "", "bullets": [], "quote": ""}
        elif line.startswith("[Title]"):
            current_slide["title"] = line.replace("[Title]", "").replace("[/Title]", "").strip()
        elif line.startswith("[Description]"):
            current_slide["description"] = line.replace("[Description]", "").replace("[/Description]", "").strip()
        elif line.startswith("[bullet point]"):
            current_slide["bullets"].append(line.replace("[bullet point]", "").replace("[/bullet point]", "").strip())
        elif line.startswith("[Creative Phrase]"):
            current_slide["quote"] = line.replace("[Creative Phrase]", "").replace("[/Creative Phrase]", "").strip()
        elif line.startswith("[Smartnote Title]"):
            capturing_smartnote_title = True
        elif line.startswith("[/Smartnote Title]"):
            capturing_smartnote_title = False
        elif capturing_smartnote_title:
            smartnote_title += line.strip()
            #print(f"Found Smartnote Title: {smartnote_title}")  # Debug print
        elif line.startswith("[Smartnote Description]"):
            capturing_smartnote_description = True
        elif line.startswith("[/Smartnote Description]"):
            capturing_smartnote_description = False
        elif capturing_smartnote_description:
            smartnote_description += line.strip()
            #print(f"Found Smartnote Description: {smartnote_description}")  # Debug print

    if current_slide:
        slides_content.append(current_slide)
    if not slides_content or all(not slide["title"] and not slide["description"] and not slide["bullets"] and not slide["quote"] for slide in slides_content):
        raise ValueError("Parsed content is empty")
    #print(f"Final Smartnote Title: {smartnote_title}")  # Debug print
    #print(f"Final Smartnote Description: {smartnote_description}")  # Debug print
    return slides_content, smartnote_title, smartnote_description


def create_presentation(slides_content, name):
    """Function to create a presentation with the given slides content and user name."""
    template_path = 'follow_template.pptx'  # Corrected path to the uploaded template
    prs = Presentation(template_path)

    for slide_number, slide_content in enumerate(slides_content):
        if slide_number >= len(prs.slides):
            break

        replacements = {
            "{Title}": slide_content["title"],
            "{Description}": slide_content["description"],
            "{Bullet Point 1}": slide_content["bullets"][0] if len(slide_content["bullets"]) > 0 else "",
            "{Bullet Point 2}": slide_content["bullets"][1] if len(slide_content["bullets"]) > 1 else "",
            "{Bullet Point 3}": slide_content["bullets"][2] if len(slide_content["bullets"]) > 2 else "",
            "{Quote}": to_sentence_case(slide_content["quote"]),
            "{Name}": name,
            "{TITLE}": slide_content["title"],  
                        
        }
        replace_placeholders(prs.slides[slide_number], replacements)

    return prs


def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


@shared_task
def generate_ppt_task(chat_history, tab_name, username, language):
    try:
        def log_memory_usage(stage):
            process = psutil.Process(os.getpid())
            print(f"[{stage}] Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        log_memory_usage("Start")

        OPENAI_API_KEY = settings.OPENAI_API_KEY
        GPT_MODEL = settings.GPT_MODEL_2
        BASE_URL = settings.BASE_URL
        TOGETHER_API_KEY = settings.TOGETHER_API_KEY

        with open("PPT_Prompt.txt", "r") as file:
            prompt_ = file.read()
        log_memory_usage("Read Prompt")

        SYSPROMPT = str(prompt_)
        prompt = PromptTemplate.from_template(SYSPROMPT)

        chat_history_cleaned = clean_chat_history(chat_history)
        chat_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history_cleaned])
        log_memory_usage("Clean Chat History")

        client = Together(api_key=TOGETHER_API_KEY)
        messages = [
            {"role": "system", "content": SYSPROMPT},
            {"role": "user", "content": f"{chat_text}\n\nLanguage: {language}"}
        ]


        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=GPT_MODEL,
            max_tokens=2048,
            temperature=0.7,
            openai_api_base=BASE_URL,
        )

        #chain = prompt | llm
        #response = chain.stream({'chat_text': chat_text, 'language': language})
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=True
        )

        #analysis = ""
        #for chunk in response:
        #    analysis += chunk.content
        print(response)
        analysis = ""
        for chunk in response:
            # Print the chunk to debug the response structure
            print(f"Chunk: {chunk}")
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                analysis += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                analysis += chunk.choices[0].message.content

 
        log_memory_usage("Generate Analysis")
        print(analysis)
        if not analysis.strip():
            raise ValueError("AI-generated content is empty")
        slides_content, smartnote_title, smartnote_description = parse_ai_generated_content(analysis)
        log_memory_usage("Parse AI Content")
        # Check if parsed content is empty
        if not slides_content or all(not slide["title"] and not slide["description"] and not slide["bullets"] and not slide["quote"] for slide in slides_content):
            raise ValueError("Parsed content is empty")
        print(slides_content)
        print("Smartnote Title:", smartnote_title)
        print("Smartnote Description:", smartnote_description)
        
        presentation = create_presentation(slides_content, username)
        log_memory_usage("Create Presentation")

        pptx_stream = BytesIO()
        presentation.save(pptx_stream)
        pptx_stream.seek(0)
        pptx_base64 = base64.b64encode(pptx_stream.read()).decode('utf-8')
        log_memory_usage("Save Presentation")

        return {
            'pptx_base64': pptx_base64,
            'smartnote_title': smartnote_title,
            'smartnote_description': smartnote_description
        }

    except Exception as e:
        raise ValueError(f"Task failed: {str(e)}")


MODEL_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

#MODEL_405B = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

@shared_task
def process_single_prompt1(prompt, chat_history, language, model_name):
    TOGETHER_API_KEY = settings.TOGETHER_API_KEY
    client = Together(api_key=TOGETHER_API_KEY)
    max_tokens = 8192  # Adjust based on requirement

    # Add the current prompt to the chat history
    if chat_history.strip() != "":
        chat_history += f"[INST]\n{prompt.strip()}\n[/INST]\n"

    # Combine the chat history with the user's language and instruction
    current_input = f"[INST]\nLanguage: {language}\n[/INST]\n"
    combined_input = chat_history + current_input

    # Call Together API to get response
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": combined_input}],
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=True
    )



    # Process the streamed response
    generated_text = ""

    # print("TEST GEN: ", generated_text)
    # print("TEST RES: ", response)

    # response = response.choices[0].message.content
    # generated_text = response

    for chunk in response:
        if len(chunk.choices) > 0:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                if chunk.choices[0].message.content:
                    generated_text += chunk.choices[0].message.content
        else:
            logger.info(f"CHUNK HAS NO CHOICES: {chunk.choices}", )

    # Update the chat history with the generated response
    chat_history += f"[INST]\n{generated_text.strip()}\n[/INST]\n"

    return chat_history, generated_text

##option1
@shared_task
def process_prompts1(final_content, language):
    try:
        # Load system prompts from a file
        with open("SA_option1.txt", "r") as file:
            prompt_file_content = file.read()

        # Split prompts by custom delimiter
        prompts = prompt_file_content.split("######")

        # Initialize chat history with the user's first input
        chat_history = f"[INST]\n{final_content.strip()}\n[/INST]\n"
        all_responses = []
        final_response = ""

        # Iterate through prompts
        for i, prompt in enumerate(prompts):
            model_name = MODEL_70B

            # Process each prompt with chat history
            chat_history, generated_text = process_single_prompt1(
                prompt, chat_history, language, model_name=model_name)

            # Collect the response for each prompt
            all_responses.append(generated_text)

            # For the last prompt, keep the final response
            if i == len(prompts) - 1:
                final_response = generated_text

            # Log the response for debugging
            print(f"Response for Prompt {i + 1}: {generated_text}")

        # Return final response and all intermediate responses
        return {
            "final_text": final_response,
            "all_responses": all_responses
        }

    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise ValueError(f"Task failed: {str(e)}")

######################################################################################################
@shared_task
def process_single_prompt2(prompt, chat_history, language, model_name=None, is_deepinfra=False):
    if is_deepinfra:
        # Use DeepInfra model for the 3rd prompt
        messages = f"Language: {language}\n{chat_history}<INST>\n{prompt.strip()}\n</INST>\n"
        response = llm.stream(messages)  # Assuming llm is the DeepInfra model object
        generated_text = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                generated_text += chunk.content
            else:
                generated_text += str(chunk)
        
        # After generating the response, update the chat history with the short prompt and the generated response
        short_prompt = "Identify the most suitable model from the options provided or create a new one if necessary, explaining its relevance and usage."
        chat_history += f"<INST>\n{short_prompt}\n</INST>\n"
        chat_history += f"<INST>\n{generated_text.strip()}\n</INST>\n"
    else:
        # Use Together API for other prompts
        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)
        max_tokens = 8192

        messages = [
            {"role": "system", "content": prompt.strip()},
            {"role": "user", "content": f"{chat_history}\n\nLanguage: {language}"}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
            stream=True
        )

        generated_text = ""
        for chunk in response:
            if len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        generated_text += chunk.choices[0].delta.content
                elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                    if chunk.choices[0].message.content:
                        generated_text += chunk.choices[0].message.content
            else:
                logger.info(f"CHUNK HAS NO CHOICES: {chunk.choices}", )
        # Update chat history with the full prompt and generated response
        chat_history += f"<INST>\n{prompt.strip()}\n</INST>\n"
        chat_history += f"<INST>\n{generated_text.strip()}\n</INST>\n"
    
    return chat_history, generated_text


##option2
@shared_task
def process_prompts2(final_content, language):
    try:
        # Load prompts from the text file
        with open("SA_option2.txt", "r") as file:
            prompt_file_content = file.read()

        # Split the content by a custom delimiter like "######"
        PROMPTS = prompt_file_content.split("######")

        chat_history = f"<INST>\n{final_content.strip()}\n</INST>\n"
        all_responses = []
        final_response = ""

        for i, prompt in enumerate(PROMPTS):
            if i == 2:
                # Third prompt with DeepInfra model
                chat_history, generated_text = process_single_prompt2(prompt, chat_history, language, is_deepinfra=True)
            else:
                # Other prompts with Together API
                model_name = MODEL_70B
                chat_history, generated_text = process_single_prompt2(prompt, chat_history, language, model_name=model_name)
                #chat_history, generated_text = process_single_prompt2(prompt, chat_history, language, is_deepinfra=True)
            
            # Append the response to the list for final return
            all_responses.append(generated_text)

            # Check if this is the final prompt (prompt 10)
            if i == len(PROMPTS) - 1:
                final_response = generated_text  # Store the final response

            # Log the response for each prompt
            print(f"Response for Prompt {i + 1}: {generated_text}")

        # Print the final chat history after all prompts have been processed
        #print("Final Chat History:")
        #print(chat_history)
        #print("finalone--------")
        #print(final_response)

        # Return the final response only for prompt 10
        return {
            "final_text": final_response,
            "all_responses": all_responses
        }

    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise ValueError(f"Task failed: {str(e)}")

####################################################################################################

##option3
@shared_task
def process_prompts3(final_content, language):
    try:
        # Load prompts from the text file
        with open("SA_option3.txt", "r") as file:
            prompt_file_content = file.read()

        # Split the content by a custom delimiter like "######"
        PROMPTS = prompt_file_content.split("######")

        # Initialize chat history
        chat_history = f"<INST>\n{final_content.strip()}\n</INST>\n"
        all_responses = []
        final_response = ""

        # Process Prompt 1
        prompt_1 = PROMPTS[0]  # First prompt
        chat_history, generated_text = process_single_prompt3(prompt_1, chat_history, language, model_name=MODEL_70B)
        print(f"Response for Prompt 1: {generated_text}")

        # Overwrite Prompt 1 response with Prompt 2
        prompt_2 = PROMPTS[1]  # Second prompt
        chat_history, generated_text = process_single_prompt3(prompt_2, chat_history, language, model_name=MODEL_70B)
        checkpoint_response = generated_text  # Store checkpoint after Prompt 2
        print(f"Response for Prompt 2 (checkpoint): {generated_text}")

        # Initialize a container for Prompt 3 iterations
        prompt_3_responses = ""

        # Iterations of Prompt 3 (from 1 to 6)
        prompt_3_template = PROMPTS[2]  # Template for prompt with {number}
        for number in range(1, 7):  # Replace {number} with values 1 to 6
            prompt_with_number = prompt_3_template.replace("{number}", str(number))
            _, generated_text = process_single_prompt3(prompt_with_number, chat_history, language, model_name=MODEL_70B)
            
            # Append each iteration's response independently to the final combined result
            prompt_3_responses += generated_text
            print(f"Response for Prompt 3 (iteration {number}): {generated_text}")

        # Now append the final combined responses from Prompt 3
        all_responses.append(prompt_3_responses)

        # Process Prompt 4 (reassessing the situation)
        prompt_4 = PROMPTS[3]  # Final reassessment prompt
        chat_history, reassessment_response = process_single_prompt3(prompt_4, prompt_3_responses, language, model_name=MODEL_70B)
        all_responses.append(reassessment_response)
        final_response += reassessment_response  # Append to final response
        print(f"Response for Prompt 4: {reassessment_response}")

        # Process Prompt 5 (final formatting and HTML)
        final_prompt = PROMPTS[4]  # Final prompt with HTML instructions
        chat_history, final_formatted_response = process_single_prompt3(final_prompt, chat_history, language, model_name=MODEL_70B)
        final_response = final_formatted_response  # Overwrite with final formatted response
        print("Final formatted response:")
        print(final_formatted_response)

        # Return the final response including all stages of processing
        return {
            "final_text": final_response,
            "all_responses": all_responses
        }

    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise ValueError(f"Task failed: {str(e)}")



##option3
@shared_task
def process_single_prompt3(prompt, chat_history, language, model_name=None):
    TOGETHER_API_KEY = settings.TOGETHER_API_KEY
    client = Together(api_key=TOGETHER_API_KEY)
    max_tokens = 8192

    messages = [
        {"role": "system", "content": prompt.strip()},
        {"role": "user", "content": f"{chat_history}\n\nLanguage: {language}"}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=0.9,
        stream=True
    )

    generated_text = ""
    for chunk in response:
        if len(chunk.choices) > 0:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                if chunk.choices[0].message.content:
                    generated_text += chunk.choices[0].message.content
        else:
            logger.info(f"CHUNK HAS NO CHOICES: {chunk.choices}", )
    # Update chat history with the full prompt and generated response
    chat_history += f"<INST>\n{prompt.strip()}\n</INST>\n"
    chat_history += f"<INST>\n{generated_text.strip()}\n</INST>\n"

    return chat_history, generated_text






@shared_task
def process_prompts4(final_content, language):
    try:
        # Load system prompt from the text file
        with open("cpromptcheck.txt", "r") as file:
            prompt_file_content = file.read()

        # Replace {final_content} in the system prompt with the actual final_content input
        system_prompt = prompt_file_content.replace("{final_content}", final_content.strip())

        # Send the system prompt to the DeepInfra LLM
        response = llm.stream(system_prompt)
        print(f"Raw response from DeepInfra: {response}")

        # Collect response content
        generated_text = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                generated_text += chunk.content
            else:
                generated_text += str(chunk)

        final_response = generated_text.strip()

        # Log the final response
        print(f"Final LLM Response:\n{final_response}")

        # Search for the JSON content in the response, even if the exact pattern is not found
        json_start = final_response.find("{")
        json_end = final_response.rfind("}") + 1  # Locate the last closing brace for the JSON

        if json_start != -1 and json_end != -1:
            # Extract the JSON string from the response
            json_string = final_response[json_start:json_end]

            # Parse the extracted JSON
            try:
                canvas_data = json.loads(json_string)
                print(f"Extracted JSON: {canvas_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {str(e)}")
                raise ValueError(f"Failed to decode JSON: {str(e)}")

            # Check for the template type
            template_type = canvas_data.get("canvas", {}).get("template_type")
            print(f"Template Type: {template_type}")
            response_data = {
                "final_text": final_response,
                "template_type": template_type
            }

            # Based on the template type, forward to the appropriate function
            if template_type == 1:
                handle_template_type_1(canvas_data)
            elif template_type == 2:
                handle_template_type_2(canvas_data)
            elif template_type == 3:
                handle_template_type_3(canvas_data)
            elif template_type == 4:
                pptx_data = handle_template_type_4(canvas_data)
                response_data.update(pptx_data) 
            else:
                logger.error(f"Unknown template type: {template_type}")
                raise ValueError(f"Unknown template type: {template_type}")
        
        else:
            logger.error("No valid JSON output found in the LLM response")
            raise ValueError("No valid JSON output found in the LLM response")

        return response_data

    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise ValueError(f"Task failed: {str(e)}")

# Example functions to handle each template type
def handle_template_type_1(canvas_data):
    print(f"Handling template type 1 with data: {canvas_data}")

def handle_template_type_2(canvas_data):
    print(f"Handling template type 2 with data: {canvas_data}")

def handle_template_type_3(canvas_data):
    print(f"Handling template type 3 with data: {canvas_data}")

def handle_template_type_4(canvas_data):
    presentation = Presentation("Hex Canvas Design (1).pptx")
    print(f"Handling template type 4 with data: {canvas_data}")
    # Adjust the replacement dictionary, including 'cut1' and 'cut2' with different font sizes and title color
    replacement_dict = {
        "box1": f"        {canvas_data['canvas']['top_hexagons'][0]['title']}\n\n        {canvas_data['canvas']['top_hexagons'][0]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['top_hexagons'][0]['key_elements']),
        "top_hex2": f"        {canvas_data['canvas']['top_hexagons'][1]['title']}\n\n        {canvas_data['canvas']['top_hexagons'][1]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['top_hexagons'][1]['key_elements']),
        "top_hex3": f"        {canvas_data['canvas']['top_hexagons'][2]['title']}\n\n        {canvas_data['canvas']['top_hexagons'][2]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['top_hexagons'][2]['key_elements']),
        "top_hex4": f"        {canvas_data['canvas']['top_hexagons'][3]['title']}\n\n        {canvas_data['canvas']['top_hexagons'][3]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['top_hexagons'][3]['key_elements']),
        "box2": f"        {canvas_data['canvas']['bottom_hexagons'][0]['title']}\n\n        {canvas_data['canvas']['bottom_hexagons'][0]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['bottom_hexagons'][0]['key_elements']),
        "bottom_hex2": f"        {canvas_data['canvas']['bottom_hexagons'][1]['title']}\n\n        {canvas_data['canvas']['bottom_hexagons'][1]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['bottom_hexagons'][1]['key_elements']),
        "bottom_hex3": f"        {canvas_data['canvas']['bottom_hexagons'][2]['title']}\n\n        {canvas_data['canvas']['bottom_hexagons'][2]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['bottom_hexagons'][2]['key_elements']),
        "bottom_hex4": f"        {canvas_data['canvas']['bottom_hexagons'][3]['title']}\n\n        {canvas_data['canvas']['bottom_hexagons'][3]['description']}\n        - " + "\n        - ".join(canvas_data['canvas']['bottom_hexagons'][3]['key_elements']),
        "cut1": canvas_data["canvas"]["canvas_name"],
        "cut2": canvas_data["canvas"]["canvas_description"],
    }
    # Iterate through slides and apply formatting for 'cut1' and 'cut2' (different sizes and black titles)
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                for placeholder, replacement in replacement_dict.items():
                    if placeholder in shape.text:
                        shape.text = shape.text.replace(placeholder, replacement)
    
                        # Adjust text fitting, size, alignment, and color
                        if hasattr(shape, "text_frame") and shape.text_frame is not None:
                            shape.text_frame.word_wrap = True
                            shape.text_frame.auto_size = True  # Ensure text boxes resize to fit content
                            for paragraph in shape.text_frame.paragraphs:
                                for run in paragraph.runs:
                                    if placeholder == "cut1":
                                        run.font.size = Pt(20)  # Larger font for cut1
                                        run.font.name = "Arial"
                                        run.font.color.rgb = RGBColor(0, 0, 0)  # Black color for title
                                    elif placeholder == "cut2":
                                        run.font.size = Pt(14)  # Smaller font for cut2
                                        run.font.name = "Arial"
                                        run.font.color.rgb = RGBColor(0, 0, 0)  # Black color for title
                                    else:
                                        run.font.size = Pt(9)  # Standard font size for other shapes
                                        run.font.name = "Arial"
                                        run.font.color.rgb = RGBColor(255, 255, 255)  # White color for other texts
    
    pptx_stream = BytesIO()
    presentation.save(pptx_stream)
    pptx_stream.seek(0)  # Move the stream position to the start
    pptx_base64 = base64.b64encode(pptx_stream.read()).decode('utf-8')
    
    log_memory_usage("Save Presentation")
    print("done")

    return {
        'pptx_base64': pptx_base64,
    }
    
    



    





