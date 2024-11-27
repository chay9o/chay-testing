import os
import uuid
from django.http import StreamingHttpResponse
from langchain_core.callbacks import BaseCallbackHandler
from django.conf import settings
from django.core.cache import cache
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from .retriever import LLMHybridRetriever
from .master_vectors.MV import MasterVectors
from .chunker import ChunkText
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
import json
from .credentials import ClientCredentials
from .backup import AWSBackup
from jinja2 import Template
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .tasks import generate_ppt_task, process_prompts_1, process_prompts_2, process_prompts_3, process_prompts4, evaluate_text_task
from celery.result import AsyncResult
import re
from together import Together
import fasttext
import psycopg2
import logging
import asyncio
from datetime import datetime, timedelta

load_dotenv()


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

job_done = "STOP"

def log_info_async(message):
    logger.info(message)


if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'main.settings')
load_dotenv()

# Load prompt templates
with open("system-prompt.txt", "r") as file:
    SYSPROMPT = file.read()

with open("chatnote-prompt.txt", "r") as file:
    CHATNOTE_PROMPT = file.read()

with open("analytics-prompt.txt", "r") as file:
    ANYPROMPT = file.read()

prompt = PromptTemplate.from_template(SYSPROMPT)
chat_note_prompt = PromptTemplate.from_template(CHATNOTE_PROMPT)

llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.GPT_MODEL_2,
    openai_api_base=settings.BASE_URL,
    max_tokens=1000
)

mv = MasterVectors()
slice_document = ChunkText()
chat_template = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '[INST] ' + message['text'] + ' [/INST]' }}{% elif message['role'] == 'bot' %}{{ message['text'] + eos_token}}{% endif %}{% endfor %}"
)

@csrf_exempt
def log_query_counters(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            #company_id = data.get("company_id")
            #initiative_id = data.get("initiative_id")
            total_queries = data.get("total_queries")
            #trained_data_queries = data.get("trained_data_queries")

            # Log and print the query counters
            #logger.info(f"Company ID: {company_id}, Initiative ID: {initiative_id}")
            #logger.info(f"Total Queries: {total_queries}, Trained Data Queries: {trained_data_queries}")
            logger.info(f"Total Queries: {total_queries}")
            #print(f"Company ID: {company_id}, Initiative ID: {initiative_id}")
            #print(f"Total Queries: {total_queries}, Trained Data Queries: {trained_data_queries}")
            print(f"Total Queries: {total_queries}")

            return JsonResponse({"status": "success", "message": "Query counters logged successfully."})
        except Exception as e:
            logger.error(f"Error logging query counters: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "Invalid request method."}, status=405)
    
@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        collection = "C" + str(data['collection'])
        query = str(data['query']).lower()
        user_id = str(data['user_id'])
        chat_history = data.get('chatHistory', [])
        language = data.get('language', 'en')
        current_date = data.get('current_date', '')

        log_info_async(f"data: {data}")

        # Check if collection exists
        if not llm_hybrid.collection_exists(collection):
            return JsonResponse({'error': 'This collection does not exist!'}, status=400)

        master_vector = mv.search_master_vectors(query=query, class_="MV001")

        key = llm_hybrid.important_words(query=query)

        keyword_pattern = r'KEYWORD:\s*\["(.*?)"\]'
        keyword_match = re.search(keyword_pattern, key)

        if keyword_match:
            keywords_str = keyword_match.group(1)
            keywords_list = [keyword.strip().strip("'") for keyword in keywords_str.split(",")]
        else:
            keywords_list = []

        if 2 <= len(keywords_list) <= 3:
            for keyword in keywords_list:
                r1 = mv.search_master_vectors(query=keyword, class_="MV001")
                master_vector.extend(r1)

        final_query = f"{query}, {' '.join(keywords_list)}"
        top_master_vec = mv.reranker(query=final_query, batch=master_vector, return_type=str)

        retriever = f"{top_master_vec}"

        # If there's no chat history, initialize it
        if not chat_history:
            chat_history_str = ""
        else:
            template = Template(chat_template)
            data = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "messages": [
                    {"role": msg['role'], "text": msg['text']} for msg in chat_history
                ]
            }
            chat_history_str = template.render(data)

        # LLMChain for generating a response
        chain = prompt | llm

        # Prepare input for LLM
        response = chain({
            'matching_model': retriever,
            'question': query,
            'username': data.get('user', 'Unknown User'),
            'chat_history': chat_history_str,
            'language_to_use': language,
            'current_date': current_date,
        })

        return JsonResponse({"message": response}, status=200)

    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
    
DEFAULT_AREAS = {
    "Strategy": 0,
    "Teams": 0,
    "Customer": 0,
    "Company": 0,
    "Workforce_Wellbeing": 0,
    "Change_Management": 0,
    "Risk_Management": 0,
    "Operations": 0,
    "Innovation": 0,
    "Finance": 0,
    "Compliance_Governance": 0,
    "Sustainability": 0,
    "Technology": 0
}

# A global dictionary to simulate an in-memory table (not persistent)
classification_store = {}

# New helper function that accepts the parsed payload directly
def insert_classification_data(data):
    company_id = data.get("Company_ID")
    initiative_id = data.get("Initiative_ID")
    date = data.get("Date")
    
    # Create a unique key for each record
    record_key = f"{company_id}-{initiative_id}-{date}"

    # Ensure all categories are present with a value of 0 or 1
    validated_areas = {category: min(max(data.get("areas", {}).get(category, 0), 0), 1) for category in DEFAULT_AREAS}

    # If the record already exists, update the values
    if record_key in classification_store:
        for key, value in validated_areas.items():
            classification_store[record_key]["areas"][key] = min(classification_store[record_key]["areas"][key] + value, 1)
    else:
        # Store new data
        classification_store[record_key] = {
            "Month": date[:7] + "-01",  # 'YYYY-MM-01' format
            "areas": validated_areas
        }
    print(f"Stored data for {record_key}: {classification_store[record_key]}")
    return {"status": "success"}




def classify_text_with_llm_together(query_text):
    TOGETHER_API_KEY = settings.TOGETHER_API_KEY
    client = Together(api_key=TOGETHER_API_KEY)
    
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    max_tokens = 2048

    # Insert the user query into the system prompt
    prompt = f"{ANYPROMPT}\nClassify the following text:\n\n{query_text}"

    # LLM response
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
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
                    
    print("Generated text:", generated_text)

    return generated_text
    
@csrf_exempt
def webhook_handler(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query")
            source = data.get("source")
            company_id = data.get("collection")  
            initiative_id = data.get("entity")  
            date = data.get("date")
            
            # Perform classification or analytics logging
            print(f"Received query from {source}: {query}")
            classification_result = classify_text_with_llm_together(query)
            print(f"classification: {classification_result}")
            
            try:
                classification_data = json.loads(classification_result)
                areas = classification_data.get("areas", {})
            except json.JSONDecodeError as e:
                return JsonResponse({"error": f"Failed to parse classification result: {str(e)}"}, status=500)

            # Return classification data to the user
            return JsonResponse({
                "result": classification_data
            }, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def view_classifications(request):
    try:
        # Get filter parameters from the request
        filter_type = request.GET.get("filter_type")  # 'company_initiative' or 'company'
        company_id = request.GET.get("company_id")
        initiative_id = request.GET.get("initiative_id")
        area_condition = request.GET.get("area", "Strategy")

        # Prepare filtered results
        filtered_results = {}

        if filter_type == "company_initiative" and company_id and initiative_id:
            # Filter by Company ID, Initiative ID, and a condition (e.g., Strategy > 0)
            for key, value in classification_store.items():
                if (key[0] == company_id and key[1] == initiative_id and
                        value["areas"].get(area_condition, 0) > 0):
                    filtered_results[key] = value

        elif filter_type == "company" and company_id:
            # Filter by Company ID and a condition (e.g., Strategy = 1)
            for key, value in classification_store.items():
                if (key[0] == company_id and value["areas"].get(area_condition, 0) == 1):
                    filtered_results[key] = value

        # Return the filtered results
        return JsonResponse(filtered_results, safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
        
credentials = ClientCredentials()

awsb = AWSBackup(bucket_name=settings.FILE_BUCKET_NAME)
weaviate_backup = AWSBackup(bucket_name=settings.BUCKET_NAME)

file = open("system-prompt.txt", "r")
prompt_ = file.read()
file.close()

read_prompt = open("chatnote-prompt.txt", "r")
note_prompt = read_prompt.read()
read_prompt.close()

live_data_prompt_file = open("live-data-prompt.txt", "r")
LIVE_DATA_PROMPT = live_data_prompt_file.read()
live_data_prompt_file.close()

SYSPROMPT = str(prompt_)
CHATNOTE_PROMPT = str(note_prompt)

prompt = PromptTemplate.from_template(SYSPROMPT)
chat_note_prompt = PromptTemplate.from_template(CHATNOTE_PROMPT)
live_data_prompt = PromptTemplate.from_template(LIVE_DATA_PROMPT)

llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY,
                 model_name=settings.GPT_MODEL_2,
                 openai_api_base=settings.BASE_URL,
                 # temperature=0.3,
                 max_tokens=1000,
                 streaming=True).configurable_fields(
                                    temperature=ConfigurableField(
                                        id="llm_temperature",
                                        name="LLM Temperature",
                                        description="The temperature of the LLM",
                                    )
                                )


llm_hybrid = LLMHybridRetriever(verbose=True)

mv = MasterVectors()

slice_document = ChunkText()


@api_view(http_method_names=['GET'])
def home(request) -> Response:

    return Response({'msg': 'this is hanna enterprise suite'}, status=status.HTTP_200_OK)


# ------------ TASK MANAGEMENT ------------
class SimpleCallback(BaseCallbackHandler):

    async def on_llm_start(self, serialized, prompts, **kwargs):
        if settings.DEBUG is True:
            print(f"The LLM has Started")

    async def on_llm_end(self, *args, **kwargs):

        if settings.DEBUG is True:
            print("The LLM has ended!")


# Load the template string into a Jinja object.
chat_template = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['text'] + ' [/INST]' }}"
    "{% elif message['role'] == 'bot' %}{{ message['text'] + eos_token}}"
    "{% endif %}{% endfor %}"
)

template = Template(chat_template)


@api_view(['POST'])
def generate_ppt(request):
    try:
        data = json.loads(request.body)
        chat_history = data['chatHistory']
        tab_name = data['tabName']
        username = data['username']
        language = data['language']
        
        result = generate_ppt_task.delay(chat_history, tab_name, username, language)

        return Response({'task_id': result.id}, status=status.HTTP_202_ACCEPTED)
    
    except Exception as e:
        print("VIEW GENERATE PPT:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_task_status(request, task_id):
    try:
        result = AsyncResult(task_id)
        if result.state == 'FAILURE':
            return Response({
                'task_id': task_id,
                'status': result.status,
                'result': str(result.result) 
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({'task_id': task_id, 'status': result.status, 'result': result.result})
    except Exception as e:
        print("GET TASK STATUS:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def download_ppt(request, task_id):
    try:
        result = AsyncResult(task_id)
        if result.status == 'SUCCESS':
            result_data = result.result
            pptx_base64 = result_data.get('pptx_base64')
            smartnote_title = result_data.get('smartnote_title', 'presentation')
            smartnote_description = result_data.get('smartnote_description', '')
            print("Smartnote Title:", smartnote_title)
            print("Smartnote Description:", smartnote_description)

            if pptx_base64:
                return JsonResponse({
                    'pptx_base64': pptx_base64,
                    'smartnote_title': smartnote_title,
                    'smartnote_description': smartnote_description
                })
            else:
                return JsonResponse({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            return JsonResponse({'error': 'Task not completed'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def download_ppt1(request, task_id):
    try:
        result = AsyncResult(task_id)
        print(f"Task ID: {task_id}, Task status: {result.status}")
        if result.status == 'SUCCESS':
            result_data = result.result
            print(result_data)
            if result_data is None:
                return JsonResponse({'error': 'Task result is None'}, status=status.HTTP_400_BAD_REQUEST)

            pptx_base64 = result_data.get('pptx_base64')

            if pptx_base64:
                return JsonResponse({'pptx_base64': pptx_base64})
            else:
                return JsonResponse({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)
        elif result.status == 'FAILURE':
            return JsonResponse({
                'error': 'Task failed', 
                'traceback': result.traceback
            }, status=status.HTTP_400_BAD_REQUEST)

        else:
            return JsonResponse({
                'error': 'Task not completed', 
                'status': result.status
            }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print(e)
        return JsonResponse({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
@api_view(['POST'])
def generate_questions(request):
    try:
        # Parse request body
        data = json.loads(request.body)
        response_input = data['response_input']
        language = data['language']
        
        # Load system prompt template from suggestions.txt
        with open("suggestions.txt", "r") as file:
            prompt_ = file.read()

        # Replace placeholders with actual values
        SYSPROMPT = prompt_.replace("{response_input}", response_input).replace("{language}", language)

        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)

        # Pass system prompt with both response_input and language to LLM
        messages = [
            {"role": "system", "content": SYSPROMPT},
            {"role": "user", "content": f"Analyze the following text in {language}:\n\n{response_input}"}
        ]

        # Generate LLM response
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=1024,
            temperature=0.4,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stream=True
        )

        generated_response = ""
        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                generated_response += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                generated_response += chunk.choices[0].message.content

        # Now the generated_response is a complete string, parse it as JSON
        print(f"Raw AI Response: {generated_response}")
        json_response = json.loads(generated_response)
        

        return JsonResponse(json_response, safe=False)
      
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def create_collection(request) -> Response:
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])

        llm_hybrid.add_collection(collection)

        return Response({'msg': f'Collection created!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW CREATE COLLECTION:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@csrf_exempt
@api_view(['POST'])
def evaluate_text(request):
    try:
        data = json.loads(request.body)
        note_text = data['noteText']
        guidelines = data['guidelines']

        # Trigger the Celery task
        task = evaluate_text_task.delay(note_text, guidelines)
        return JsonResponse({'task_id': task.id}, status=202)
    except Exception as e:
        print(e)
        return JsonResponse({'error': 'Something went wrong!'}, status=500)

@csrf_exempt
@api_view(['GET'])
def check_evaluation_status(request, task_id):
    task = AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'status': 'Pending'}
    elif task.state == 'SUCCESS':
        response = {'status': 'Completed', 'result': task.result}
    elif task.state == 'FAILURE':
        response = {'status': 'Failed', 'error': str(task.result)}
    else:
        response = {'status': 'Unknown'}

    return JsonResponse(response)


# Define global dictionaries to store user-specific inputs and generated questions
user_inputs_global = {}
generated_questions_global = {}
problem_description_global = {}  # Dictionary to store the problem description for each user

def store_problem_description(user_id, invocation_id, step, description, timeout):
    """
    Store the problem description for a specific user, invocation, and step.
    """
    key = f"problem_description_{user_id}_{invocation_id}_step_{step}"
    cache.set(key, description, timeout=timeout)
    print(f"[Redis] Stored problem description for user {user_id}, invocation {invocation_id}, step {step}: {description}")


def get_problem_description(user_id, invocation_id, step):
    """
    Retrieve the problem description for a specific user, invocation, and step.
    """
    key = f"problem_description_{user_id}_{invocation_id}_step_{step}"
    description = cache.get(key, "")
    print(f"[Redis] Retrieved problem description for user {user_id}, invocation {invocation_id}, step {step}: {description}")
    return description


def store_user_input(user_id, invocation_id, step, user_input, timeout):
    """
    Store user input for a specific user, invocation, and step.
    """
    key = f"user_inputs_{user_id}_{invocation_id}_step_{step}"
    existing_inputs = cache.get(key, [])
    existing_inputs.append(user_input)
    cache.set(key, existing_inputs, timeout=timeout)
    print(f"[Redis] Updated user inputs for user {user_id}, invocation {invocation_id}, step {step}: {existing_inputs}")

def get_user_inputs(user_id, invocation_id, step):
    """
    Retrieve user inputs for a specific user, invocation, and step.
    """
    key = f"user_inputs_{user_id}_{invocation_id}_step_{step}"
    inputs = cache.get(key, [])
    print(f"[Redis] Retrieved user inputs for user {user_id}, invocation {invocation_id}, step {step}: {inputs}")
    return inputs

def store_generated_question(user_id, invocation_id, step, question, timeout):
    """
    Store a generated question for a specific user, invocation, and step.
    """
    key = f"generated_questions_{user_id}_{invocation_id}_step_{step}"
    existing_questions = cache.get(key, [])
    existing_questions.append(question)
    cache.set(key, existing_questions, timeout=timeout)
    print(f"[Redis] Updated generated questions for user {user_id}, invocation {invocation_id}, step {step}: {existing_questions}")


def get_generated_questions(user_id, invocation_id, step):
    """
    Retrieve generated questions for a specific user, invocation, and step.
    """
    key = f"generated_questions_{user_id}_{invocation_id}_step_{step}"
    questions = cache.get(key, [])
    print(f"[Redis] Retrieved generated questions for user {user_id}, invocation {invocation_id}, step {step}: {questions}")
    return questions


# Helper function to build the system prompt by appending previous steps
def build_system_prompt(base_prompt, user_id, invocation_id, current_step):
    """
    Construct a system prompt using only the current step and the immediately previous steps.
    """
    steps_content = ""
    
    # Append only the relevant previous step and the current step
    for step in range(1, current_step + 1):
        generated_questions = get_generated_questions(user_id, invocation_id, step)
        user_inputs = get_user_inputs(user_id, invocation_id, step)
        
        # Append the data for the current step
        for generated_question, user_input in zip(generated_questions, user_inputs):
            if generated_question and user_input:  # Only append if both are non-empty
                steps_content += f"{generated_question}\n{user_input}\n"
    print(f"[Redis] Built system prompt for user {user_id}, invocation {invocation_id}, step {current_step}: {steps_content}")
    return base_prompt.replace("{previous_steps}", steps_content.strip())


@csrf_exempt
@api_view(['POST'])
def stinsight_step1(request):
    try:
        data = json.loads(request.body)
        print("Received data:", data)
        user_id = data.get('user_id')
        invocation_id = data.get('invocation_id', str(uuid.uuid4()))
        timeout = data.get('timeout', 1200)  # Default to 1 hour if timeout is not provided
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        
        # Capture the problem description from step1
        problem_description = data['user_input']
        language = data['language']
        store_problem_description(user_id, invocation_id, 1, problem_description, timeout)

        with open("strategic-insight-prompt.txt", "r") as file:
            prompt_template = file.read()

        # Replace the placeholder in the system prompt with user input and initialize previous steps as empty
        prompt_with_values = prompt_template.replace("{user_input}", problem_description).replace("{previous_steps}", "")

        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)

        # Step 1 only includes the problem description
        messages = [
            {"role": "system", "content": prompt_with_values},
            {"role": "user", "content": f"{problem_description}\n\nLanguage: {language}"}
        ]

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=4096,
            temperature=0,
            top_p=0.4,
            top_k=50,
            repetition_penalty=1,
            stream=True
        )

        generated_question = ""
        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                generated_question += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                generated_question += chunk.choices[0].message.content

        if not generated_question:
            return Response({'error': 'Empty response from API'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Parse the generated question JSON to extract the actual question text
        try:
            generated_question_json = json.loads(generated_question)
            generated_question_value = generated_question_json['Questions'][0]['Question']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            return Response({'error': 'Failed to parse API response'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Store only the extracted question in the global list
        store_generated_question(user_id, invocation_id, 1, generated_question_value, timeout)

        # Print the system prompt with actual values
        print("System prompt:", prompt_with_values)
        
        step1_data = {
            'invocation_id': invocation_id,
            'step1': {
                'user_input': problem_description,
                'generated_question': generated_question_value
            }
        }

        print(f"[Redis] Step 1 data for user {user_id}, invocation {invocation_id}: {step1_data}")
        return JsonResponse(step1_data, safe=False)
    except Exception as e:
        print(f"Error in step1: {e}")
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@csrf_exempt
@api_view(['POST'])
def stinsight_step2(request):
    try:
        # Extract user_id from request (assuming it's provided in the request body)
        data = json.loads(request.body)
        user_id = data.get('user_id')
        invocation_id = data.get('invocation_id')
        timeout = data.get('timeout', 1200) 
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not invocation_id:
            return Response({'error': 'invocation_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        user_input = data['user_input']
        language = data['language']

        # Store the user input for the current step
        store_user_input(user_id, invocation_id, 2, user_input, timeout)

        with open("strategic-insight-step2-3-prompt.txt", "r") as file:
            base_prompt = file.read()

        # Build the system prompt by appending previous steps
        problem_description = get_problem_description(user_id, invocation_id, 1)
        system_prompt = base_prompt.replace("{user_input}", problem_description)
        system_prompt = build_system_prompt(base_prompt, user_id, invocation_id, 2)
        
        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)

        # The user input is already included in the previous steps, so no need to repeat it
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{problem_description}\n\nLanguage: {language}"}
        ]

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=4096,
            temperature=0,
            top_p=0.4,
            top_k=50,
            repetition_penalty=1,
            stream=True
        )

        generated_question = ""
        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                generated_question += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                generated_question += chunk.choices[0].message.content

        if not generated_question:
            return Response({'error': 'Empty response from API'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Parse the generated question JSON to extract the actual question text
        try:
            generated_question_json = json.loads(generated_question)
            generated_question_value = generated_question_json['Questions'][0]['Question']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            return Response({'error': 'Failed to parse API response'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Store only the extracted question value in the global list
        store_generated_question(user_id, invocation_id, 2, generated_question_value, timeout)
        # Rebuild the system prompt again, now with the extracted question


        # Print the rebuilt system prompt
        print(f"[Redis] Rebuilt system prompt for user {user_id}, invocation {invocation_id}: {system_prompt}")

        # Return the updated data
        previous_steps_data = {
            'invocation_id': invocation_id,
            'step2': {
                'user_input': user_input,
                'generated_question': generated_question_value
            }
        }

        print(f"[Redis] Step 2 data for user {user_id}, invocation {invocation_id}: {previous_steps_data}")

        return JsonResponse(previous_steps_data, safe=False)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
@api_view(['POST'])
def stinsight_step3(request):
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        invocation_id = data.get('invocation_id')
        timeout = data.get('timeout', 1200) 

        if not user_id or not invocation_id:
            return Response({'error': 'user_id and invocation_id are required'}, status=status.HTTP_400_BAD_REQUEST)

        user_input = data['user_input']
        language = data['language']

        # Store the user input for the current step
        store_user_input(user_id, invocation_id, 3, user_input, timeout)
        
        with open("strategic-insight-step2-3-prompt.txt", "r") as file:
            base_prompt = file.read()

        # Build the system prompt by appending previous steps
        problem_description = get_problem_description(user_id, invocation_id, 2)
        system_prompt = base_prompt.replace("{user_input}", problem_description)
        system_prompt = build_system_prompt(base_prompt, user_id, invocation_id, 3)
        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)

        # The user input is already included in the previous steps, so no need to repeat it
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{problem_description}\n\nLanguage: {language}"}
        ]
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=4096,
            temperature=0,
            top_p=0.4,
            top_k=50,
            repetition_penalty=1,
            stream=True
        )

        generated_question = ""
        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                generated_question += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                generated_question += chunk.choices[0].message.content

        if not generated_question:
            return Response({'error': 'Empty response from API'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Parse the generated question JSON to extract the actual question text
        try:
            generated_question_json = json.loads(generated_question)
            generated_question_value = generated_question_json['Questions'][0]['Question']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            return Response({'error': 'Failed to parse API response'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Store only the value of the question in the global list
        store_generated_question(user_id, invocation_id, 3, generated_question_value, timeout)
        
        # Print the system prompt with actual values
        print(f"[Redis] Rebuilt system prompt for user {user_id}, invocation {invocation_id}: {system_prompt}")

        # Return the updated data
        previous_steps_data = {
            'invocation_id': invocation_id,
            'step3': {
                'user_input': user_input,
                'generated_question': generated_question_value
            }
        }
        print(f"[Redis] Step 3 data for user {user_id}, invocation {invocation_id}: {previous_steps_data}")
        return JsonResponse(previous_steps_data, safe=False)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@csrf_exempt
@api_view(['POST'])
def stinsight_step4(request):
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        invocation_id = data.get('invocation_id')
        timeout = data.get('timeout', 1200) 

        if not user_id or not invocation_id:
            return Response({'error': 'user_id and invocation_id are required'}, status=status.HTTP_400_BAD_REQUEST)

        user_input = data['user_input']
        language = data['language']

        store_user_input(user_id, invocation_id, 4, user_input, timeout)

        with open("strategic-insight-step2-3-prompt.txt", "r") as file:
            base_prompt = file.read()

        # Build the system prompt by appending previous steps
        problem_description = get_problem_description(user_id, invocation_id, 3)
        system_prompt = base_prompt.replace("{user_input}", problem_description)
        system_prompt = build_system_prompt(base_prompt, user_id, invocation_id, 4)


        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)

        # The user input is already included in the previous steps, so no need to repeat it
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{problem_description}\n\nLanguage: {language}"}
        ]
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=4096,
            temperature=0,
            top_p=0.4,
            top_k=50,
            repetition_penalty=1,
            stream=True
        )

        generated_question = ""
        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                generated_question += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                generated_question += chunk.choices[0].message.content

        if not generated_question:
            return Response({'error': 'Empty response from API'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Parse the generated question JSON to extract the actual question text
        try:
            generated_question_json = json.loads(generated_question)
            generated_question_value = generated_question_json['Questions'][0]['Question']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            return Response({'error': 'Failed to parse API response'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Store only the value of the question in the global list
        store_generated_question(user_id, invocation_id, 4, generated_question_value, timeout)

    

        # Print the system prompt with actual values
        print(f"[Redis] Rebuilt system prompt for user {user_id}, invocation {invocation_id}: {system_prompt}")

        # Return the updated data
        previous_steps_data = {
            'invocation_id': invocation_id,
            'step4': {
                'user_input': user_input,
                'generated_question': generated_question_value
            }
        }
        print(f"[Redis] Step 4 data for user {user_id}, invocation {invocation_id}: {previous_steps_data}")
        return JsonResponse(previous_steps_data, safe=False)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@csrf_exempt
@api_view(['POST'])
def stinsight_step5(request):
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        invocation_id = data.get('invocation_id')
        timeout = data.get('timeout', 1200) 

        if not user_id or not invocation_id:
            return Response({'error': 'user_id and invocation_id are required'}, status=status.HTTP_400_BAD_REQUEST)
            
        user_input = data['user_input']
        language = data['language']

        store_user_input(user_id, invocation_id, 5, user_input, timeout)

        with open("strategic-insight-step2-3-prompt.txt", "r") as file:
            base_prompt = file.read()

        # Build the system prompt by appending previous steps
        problem_description = get_problem_description(user_id, invocation_id, 4)
        system_prompt = base_prompt.replace("{user_input}", problem_description)
        system_prompt = build_system_prompt(base_prompt, user_id, invocation_id, 5)


        TOGETHER_API_KEY = settings.TOGETHER_API_KEY
        client = Together(api_key=TOGETHER_API_KEY)

        # The user input is already included in the previous steps, so no need to repeat it
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{problem_description}\n\nLanguage: {language}"}
        ]

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=4096,
            temperature=0,
            top_p=0.4,
            top_k=50,
            repetition_penalty=1,
            stream=True
        )

        generated_question = ""
        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                generated_question += chunk.choices[0].delta.content
            elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                generated_question += chunk.choices[0].message.content

        if not generated_question:
            return Response({'error': 'Empty response from API'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Parse the generated question JSON to extract the actual question text
        try:
            generated_question_json = json.loads(generated_question)
            generated_question_value = generated_question_json['Questions'][0]['Question']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            return Response({'error': 'Failed to parse API response'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Store only the value of the question in the global list
        store_generated_question(user_id, invocation_id, 5, generated_question_value, timeout)
        
        # Print the system prompt with actual values
        print(f"[Redis] Rebuilt system prompt for user {user_id}, invocation {invocation_id}: {system_prompt}")


        # Return the updated data
        previous_steps_data = {
            'invocation_id': invocation_id,
            'step5': {
                'user_input': user_input,
                'generated_question': generated_question_value
            }
        }
        print(f"[Redis] Step 5 data for user {user_id}, invocation {invocation_id}: {previous_steps_data}")
        return JsonResponse(previous_steps_data, safe=False)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
@api_view(['POST'])
def stinsight_step6(request):
    try:
        data = json.loads(request.body)
        user_id = data.get('user_id')
        invocation_id = data.get('invocation_id')
        timeout = data.get('timeout', 1200) 
        
        if not user_id or not invocation_id:
            return Response({'error': 'user_id and invocation_id are required'}, status=status.HTTP_400_BAD_REQUEST)

        user_input = data.get('user_input', '')
        language = data.get('language', 'en')
        selected_option = data.get('selected_option', 'option1')

        if user_input:
            store_user_input(user_id, invocation_id, 6, user_input, timeout)

        # Collect the final content that includes the problem description and all user inputs
        final_content = ""
        for step in range(1, 7):  # Assuming a maximum of 6 steps
            problem_description = get_problem_description(user_id, invocation_id, step)
            user_inputs = get_user_inputs(user_id, invocation_id, step)
            if problem_description:
                final_content += f"{problem_description}\n\n"
            if user_inputs:
                final_content += "\n".join(user_inputs) + "\n\n"

        final_content = final_content.strip()  # Clean up trailing whitespace

        print(f"[Redis] Final content for user {user_id}, invocation {invocation_id}: {final_content}")
        print(f"[Redis] Selected option for user {user_id}, invocation {invocation_id}: {selected_option}")

        # Trigger the appropriate Celery task based on the dropdown value
        if selected_option == 'option1':
            # Trigger process_prompts for option 1 (Strategic Analysis for Decision Makers)
            task = process_prompts_1.apply_async(args=[final_content, language])
            print(f"[Celery] Task initiated for Option 1: {task.id}")
        elif selected_option == 'option2':
            # Trigger process_prompts2 for option 2 (Strategic Analysis for Organizational Architects)
            task = process_prompts_2.apply_async(args=[final_content, language])
            print(f"[Celery] Task initiated for Option 2: {task.id}")
        elif selected_option == 'option3':
            # Trigger process_prompts3 for option 3 (Strategic Analysis for Cognitive Dynamics)
            task = process_prompts_3.apply_async(args=[final_content, language])
            print(f"[Celery] Task initiated for Option 3: {task.id}")
        elif selected_option == 'option4':
            # Trigger process_prompts3 for option 3 (Strategic Analysis for Cognitive Dynamics)
            response_data = trigger_ppt_generation(final_content, language, user_id, invocation_id)
            if 'error' in response_data:
                return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return JsonResponse(response_data, status=status.HTTP_202_ACCEPTED)
        else:
            return JsonResponse({'error': 'Invalid option selected'}, status=status.HTTP_400_BAD_REQUEST)

        return JsonResponse({"task_id": task.id, "status": "Processing initiated"}, status=status.HTTP_202_ACCEPTED)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def trigger_ppt_generation(final_content, language, user_id, invocation_id):
    try:
        # Trigger the asynchronous task for PPT generation
        task = process_prompts4.apply_async(args=[final_content, language, user_id, invocation_id])
        print(f"[Celery] Task initiated for PPT generation: {task.id}")
        return {'task_id': task.id, 'status': 'PPT generation initiated'}
    except Exception as e:
        logger.error(f"Error in trigger_ppt_generation: {e}")
        return {'error': 'Failed to initiate PPT generation'}

@api_view(['POST'])
def generate_ppt_for_option4(request):
    """
    Endpoint to trigger PPT generation specifically for 'option4'.
    It initiates the task and returns the task ID for status tracking.
    """
    try:
        data = json.loads(request.body)
        final_content = data.get('final_content')
        language = data.get('language')
        user_id = data.get('user_id')
        

        response_data = trigger_ppt_generation(final_content, language, user_id)
        if 'error' in response_data:
            return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(response_data, status=status.HTTP_202_ACCEPTED)
    except Exception as e:
        logger.error(f"Error in generate_ppt_for_option4: {e}")
        return Response({'error': 'Failed to initiate PPT generation'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_ppt_for_option4(request, task_id):
    """
    Endpoint to retrieve the generated PPT file for 'option4' after the task is complete.
    """
    try:
        result = AsyncResult(task_id)
        
        if result.status == 'SUCCESS':
            result_data = result.result
            pptx_base64 = result_data.get('pptx_base64', '')
            #print(f"Retrieved pptx_base64 in result data: {pptx_base64}")
            smartnote_title = result_data.get('smartnote_title', 'Default Title')
            smartnote_description = result_data.get('smartnote_description', 'Default Description')
            #title = "title"
            #description = "description"

           
            return JsonResponse({
                'status': 'SUCCESS',
                'pptx_base64': pptx_base64,
                'smartnote_title': smartnote_title,
                'smartnote_description': smartnote_description
            })
                
        elif result.status == 'FAILURE':
            return JsonResponse({'status': 'FAILURE', 'error': 'Task failed'}, status=status.HTTP_400_BAD_REQUEST)
        
        else:  # For PENDING or any other status
            return JsonResponse({'status': result.status})
            
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@csrf_exempt
@api_view(['POST'])
def DA_tester(request):
    try:
        # Parse the request data
        data = json.loads(request.body)
        user_input = data.get('user_input', '')
        language = data.get('language', 'en')
        selected_option = data.get('selected_option', 'option1')  # Get the dropdown choice

        # Check if user_input or language are missing
        if not user_input or not language:
            return JsonResponse({'error': 'user_input and language are required fields'}, status=status.HTTP_400_BAD_REQUEST)

        # Trigger the appropriate Celery task based on the dropdown value
        if selected_option == 'option1':
            # Trigger process_prompts1 for option 1 (e.g., Data Analysis Option 1)
            task = process_prompts_1.apply_async(args=[user_input, language])
        elif selected_option == 'option2':
            # Trigger process_prompts2 for option 2 (e.g., Data Analysis Option 2)
            task = process_prompts_2.apply_async(args=[user_input, language])
        elif selected_option == 'option3':
            # Trigger process_prompts3 for option 3 (e.g., Data Analysis Option 3)
            task = process_prompts_3.apply_async(args=[user_input, language])
        elif selected_option == 'option4':
            # Trigger process_prompts3 for option 3 (Strategic Analysis for Cognitive Dynamics)
            response_data = trigger_ppt_generation(final_content, language)
            if 'error' in response_data:
                return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return JsonResponse(response_data, status=status.HTTP_202_ACCEPTED)
        else:
            return JsonResponse({'error': 'Invalid option selected'}, status=status.HTTP_400_BAD_REQUEST)

        # Return task id and status
        return JsonResponse({"task_id": task.id, "status": "Processing initiated"}, status=status.HTTP_202_ACCEPTED)
    except Exception as e:
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_task_status1(request, task_id):
    try:
        result = AsyncResult(task_id)

        if result.state == 'FAILURE':
            return Response({
                'task_id': task_id,
                'status': result.status,
                'result': str(result.result)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif result.state == 'SUCCESS':
            # Concatenate all accumulated answers into a single text without iteration numbers or subheadings
            accumulated_answers = result.result.get('accumulated_answers', [])
            concatenated_answers = " \n\n\n\n ".join([item['answer'] for item in accumulated_answers])

            return JsonResponse({
                'task_id': task_id,
                'status': result.status,
                'final_text': result.result['final_text'],
                'accumulated_answers': concatenated_answers,  # Concatenated answers without iteration numbers
                'chat_history': result.result.get('chat_history')  # Include chat history in success response
            })

        elif result.state == 'PROGRESS':
            return JsonResponse({
                'task_id': task_id,
                'status': result.status,
                'iteration': result.info.get('iteration'),
                'user_input': result.info.get('user_input'),
                'answer': result.info.get('answer'),
                'chat_history': result.info.get('chat_history')  # Include chat history in progress response
            })

        else:
            return JsonResponse({
                'task_id': task_id,
                'status': result.status,
                'result': str(result.result)
            })
    except Exception as e:
        logger.error(f"Error fetching task status: {e}")
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





@api_view(http_method_names=['POST'])
def chat_summary(request):
    try:
        jdata = json.loads(request.body)
        ps = llm_hybrid.summarize_chat(jdata['chat_history'])
        return Response({'msg': str(ps)}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error("CHAT SUMMARY VEC:")
        logger.error(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def create_solution(request) -> None:

    try:
        
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        collection_name = str(company['collection_name']).lower()
        type_ = str(company['type'])
        title = company.get('Title') 
        username = company.get('Username')  
        author = company.get('Author') 
        created_initiative = company.get('Created_initiative')  
        situation = company.get('situation')
        solution = company.get('solution')
        ranking = company.get('ranking')
        shared_initiatives = company.get('shared_initiatives', [])  

        
        if not situation or not solution or not ranking:
            return Response({"error": "Missing required fields: situation, solution, ranking."}, status=400)

        solution_id = str(uuid.uuid4())
        initiatives_str = ', '.join(shared_initiatives)
        solution_obj = {
            "uuid": solution_id,
            "entity": initiatives_str,  
            "situation": situation,
            "solution": solution,
            "ranking": ranking,
            "note_type": "solution",  
            "collection": collection,
            "collection_name": collection_name,
            "type": type_,
            "title": title,
            "username": username,
            "author": author,
            "created_initiative": created_initiative
        }

        llm_hybrid.weaviate_client.data_object.create(
            data_object=solution_obj,
            class_name=collection  
        )

        response_data = {
            "solution_id": solution_id,
            "situation": situation,
            "solution": solution,
            "ranking": ranking,
            "shared_initiatives": shared_initiatives,  
            "collection": collection,
            "collection_name": collection_name,
            "type": type_,
            "title": title,
            "username": username,
            "author": author,
            "created_initiative": created_initiative
        }

        return Response(response_data, status=201)

    except Exception as e:
        return Response({"error": str(e)}, status=400)

@api_view(http_method_names=['POST'])
def delete_solutions(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
             return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)


        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["id"],
                "operator": "Like",
                "valueText": obj_id  
            }
        )

        return Response({'msg': 'Solution deleted successfully!'}, status=status.HTTP_200_OK)

    except Exception as e:
        print("Error in delete_solutions:", e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





@api_view(http_method_names=['POST'])
def add_vectors_text(request) -> Response:
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        collection_name = str(company['collection_name']).lower()
        entity = str(company['entity'])
        user_id = str(company['user_id'])
        type_ = str(company['type'])
        note_type = str(company['note_type'])
        meeting_date = str(company['meeting_date'])
        time_zone = str(company['time_zone'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        pre_filter = str(company['text']).strip().replace("\n", "").lower()

        documents = slice_document.chunk_corpus(pre_filter)

        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search

        if type_ == "PV":
            # uid = "task started"
            combine_ids = "INP" + entity  # -> Initiative ID
            uid = llm_hybrid.add_batch(documents, user_id, combine_ids, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "CMV":
            # C72 -> entity
            uid = llm_hybrid.add_batch(documents, user_id, collection, collection, collection_name, note_type, meeting_date, time_zone)
        elif type_ == "INV":
            uid = llm_hybrid.add_batch(documents, user_id, entity, collection, collection_name, note_type, meeting_date, time_zone)
        else:
            logger.error("No such type! only [PV, CMV, INV] can be used!!!")
            return Response({'error': 'No such type for key -> type !'}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error("VIEW ADD TEXT VEC:")
        logger.error(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def add_vectors_all(request) -> Response:
    try:
        collection = "C" + str(request.POST.get('collection'))
        collection_name = str(request.POST.get('collection_name')).lower()
        entity = str(request.POST.get('entity'))
        user_id = str(request.POST.get('user_id'))
        type_ = str(request.POST.get('type'))
        note_type = str(request.POST.get('note_type'))
        meeting_date = str(request.POST.get('meeting_date'))
        time_zone = str(request.POST.get('time_zone'))
        text = str(request.POST.get('text'))
        hash_id = uuid.uuid4()

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        logger.info(request.POST)

        # print("PRE-FILTERING TEXT!")
        pre_filter = text.strip().replace("\n", "").lower()

        # print("SLICING DOCUMENTS!!!")
        document = slice_document.chunk_corpus(pre_filter)
        # print("DOCUMENTS SLICED!!!")

        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search

        combine_ids = ""
        if type_ == "PV":
            # uid = "task started"
            combine_ids = "INP" + entity  # -> Initiative ID
            uid = llm_hybrid.add_batch(document, user_id, combine_ids, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "CMV":
            # C72 -> entity
            combine_ids = collection
            uid = llm_hybrid.add_batch(document, user_id, collection, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "INV":
            combine_ids = entity
            uid = llm_hybrid.add_batch(document, user_id, entity, collection, collection_name, note_type, meeting_date, time_zone)

        else:
            return Response({'error': 'No such type!'}, status=status.HTTP_400_BAD_REQUEST)

        if 'file_upload' in request.POST:

            if request.POST.get('file_upload') != '':

                file_name = request.POST.get('file_upload')

                if awsb.verify_file_ext(file_name) is False:
                    logger.error("VIEW ADD VEC TEXT FILE: FILE EXTENSION NOT SUPPORTED!!!")
                    llm_hybrid.remove_uuid(uid, collection)
                    return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                if awsb.download_object_file(file_name) is False:
                    logger.error("VIEW ADD VEC TEXT FILE: CANNOT DOWNLOAD FILE!!!")
                    llm_hybrid.remove_uuid(uid, collection)
                    return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                file_ext = slice_document.return_ext(file_name)

                if file_ext == "pptx":
                    if slice_document.check_ppt(file_name) is False:
                        logger.error("VIEW ADD VEC TEXT FILE: PPTX LIMIT!!!")
                        llm_hybrid.remove_uuid(uid, collection)
                        return Response({'error': 'Max 40 slides allowed!'}, status=status.HTTP_400_BAD_REQUEST)

                if file_ext == "csv":
                    if slice_document.check_csv(file_name) is False:
                        logger.error("VIEW ADD VEC TEXT FILE: CSV LIMIT!!!")
                        llm_hybrid.remove_uuid(uid, collection)
                        return Response({'error': 'Max 500 rows allowed for csv!'}, status=status.HTTP_400_BAD_REQUEST)

                if file_ext == "xlsx":
                    if slice_document.check_excel(file_name) is False:
                        logger.error("VIEW ADD VEC TEXT FILE: XLSX LIMIT!!!")
                        llm_hybrid.remove_uuid(uid, collection)
                        return Response({'error': 'Max 500 rows and 3 sheets allowed for xlsx!'}, status=status.HTTP_400_BAD_REQUEST)

                documents = slice_document.chunk_document(file_name)

                # print(uid.split('-'))
                uuid_file = "file-" + uid.split('-')[4] + uid.split('-')[4] + uid.split('-')[3] + uid.split('-')[2] + uid.split('-')[1] + uid.split('-')[0]
                # print("FILE UUID: ", uuid_file)
                llm_hybrid.add_batch_uuid(documents, user_id, combine_ids, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

                del uuid_file
                del documents
                del collection_name
                del meeting_date
                del time_zone
                os.remove("./_tmp/" + file_name)
            else:
                logger.info("VIEW ADD VEC TEXT FILE: NO FILES!")
        else:
            logger.info("VIEW ADD VEC TEXT FILE: NO FILES!")

        del collection
        del entity
        del user_id
        del type_
        del text
        del document

        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:

        logger.error("VIEW ADD VEC TEXT FILE:")
        logger.error(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def add_vectors_file(request):
    try:
        data_file = request.FILES['file_upload']
        print(f'File Upload! {data_file}')
        print(request.POST)

        collection = "C" + str(request.POST.get('collection'))
        collection_name = str(request.POST.get('collection_name')).lower()
        entity = str(request.POST.get('entity'))
        user_id = str(request.POST.get('user_id'))
        type_ = str(request.POST.get('type'))
        note_type = str(request.POST.get('note_type'))
        meeting_date =  str(request.POST.get('meeting_date'))
        time_zone = str(request.POST.get('time_zone'))
        uid = str(request.POST.get('uuid'))

        uuid_file = "file-" + uid.split('-')[4] + uid.split('-')[4] + uid.split('-')[3] + uid.split('-')[2] + \
                    uid.split('-')[1] + uid.split('-')[0]

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        file_name = data_file.name

        destination = open("./_tmp/" + file_name, 'wb+')
        for chunk in data_file.chunks():
            destination.write(chunk)
        destination.close()

        documents = slice_document.chunk_document(file_name)

        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search
        # combine_ids = ""
        if type_ == "PV":
            # uid = "task started"
            combine_ids = "INP" + entity  # -> Initiative ID
            llm_hybrid.add_batch_uuid(documents, user_id, combine_ids, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "CMV":
            # C72 -> entity
            uid = llm_hybrid.add_batch_uuid(documents, user_id, collection, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "INV":

            uid = llm_hybrid.add_batch_uuid(documents, user_id, entity, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

        else:
            return Response({'error': 'No such type!'}, status=status.HTTP_400_BAD_REQUEST)

        os.remove("./_tmp/" + file_name)
        return Response({'msg': "Success"}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW ADD VEC FILE:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def update_chunk_text(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        collection_name = str(company['collection_name'])
        uuid_ = str(company['uuid'])
        entity = str(company['entity'])
        user_id = str(company['user_id'])
        chunk = str(company['chunk'])
        type_ = str(company['type'])
        note_type = str(company['note_type'])
        meeting_date = str(company['meeting_date'])
        time_zone = str(company['time_zone'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Equal",
                "valueText": uuid_
            },
        )

        pre_filter = str(chunk).strip().replace("\n", "").lower()

        documents = slice_document.chunk_corpus(pre_filter)

        if type_ == "PV":
            combine_ids = "INP" + entity  # -> Initiative ID
            uid = llm_hybrid.add_batch_uuid(documents, user_id, combine_ids, uuid_, collection, collection_name, note_type, meeting_date, time_zone)
        elif type_ == "CMV":
            # C72 -> entity
            uid = llm_hybrid.add_batch_uuid(documents, user_id, collection, uuid_, collection, collection_name, note_type, meeting_date, time_zone)
        elif type_ == "INV":
            uid = llm_hybrid.add_batch_uuid(documents, user_id, entity, uuid_, collection, collection_name, note_type, meeting_date, time_zone)

        return Response({'msg': 'success'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW UPDATE CHUNK:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def update_chunk_file(request):
    try:
        collection = "C" + str(request.POST.get('collection'))
        collection_name = str(request.POST.get('collection_name'))
        entity = str(request.POST.get('entity'))
        user_id = str(request.POST.get('user_id'))
        type_ = str(request.POST.get('type'))
        # text = str(request.POST.get('text'))
        uid = str(request.POST.get('uuid'))
        note_type = str(request.POST.get('note_type'))
        meeting_date = str(request.POST.get('meeting_date'))
        time_zone = str(request.POST.get('time_zone'))


        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        uuid_file = "file-" + uid.split('-')[4] + uid.split('-')[4] + uid.split('-')[3] + uid.split('-')[2] + \
                    uid.split('-')[1] + uid.split('-')[0]

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Equal",
                "valueText": uuid_file
            },
        )

        data_file = request.FILES['file_upload']
        print(f'File Upload! {data_file}')
        print(request.POST)

        file_name = data_file.name

        destination = open("./_tmp/" + file_name, 'wb+')
        for chunk in data_file.chunks():
            destination.write(chunk)
        destination.close()

        documents = slice_document.chunk_document(file_name)
        print("FILE UUID: ", uuid_file)

        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search

        # combine_ids = ""
        if type_ == "PV":
            # uid = "task started"
            combine_ids = "INP" + entity  # -> Initiative ID
            llm_hybrid.add_batch_uuid(documents, user_id, combine_ids, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "CMV":
            # C72 -> entity
            llm_hybrid.add_batch_uuid(documents, user_id, collection, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "INV":
            llm_hybrid.add_batch_uuid(documents, user_id, entity, uuid_file, collection, collection_name, note_type, meeting_date, time_zone)

        else:
            return Response({'error': 'No such type!'}, status=status.HTTP_400_BAD_REQUEST)

        os.remove("./_tmp/" + file_name)

        return Response({'msg': "Success"}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW UPDATE VEC FILE:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def update_chunk_all(request):
    try:
        collection = "C" + str(request.POST.get('collection'))
        collection_name = str(request.POST.get('collection_name'))
        entity = str(request.POST.get('entity'))
        user_id = str(request.POST.get('user_id'))
        type_ = str(request.POST.get('type'))
        note_type = str(request.POST.get('note_type'))
        meeting_date = str(request.POST.get('meeting_date'))
        time_zone = str(request.POST.get('time_zone'))
        text = str(request.POST.get('text'))
        uuid = str(request.POST.get('uuid'))

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        uuid_file = "file-" + uuid.split('-')[4] + uuid.split('-')[4] + uuid.split('-')[3] + uuid.split('-')[2] + \
                    uuid.split('-')[1] + uuid.split('-')[0]

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "ContainsAny",
                "valueTextArray": [uuid, uuid_file]
            },
        )

        pre_filter = text.strip().replace("\n", "").lower()

        document = slice_document.chunk_corpus(pre_filter)

        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search

        combine_ids = ""
        if type_ == "PV":
            # uid = "task started"
            combine_ids = "INP" + entity  # -> Initiative ID

            uid = llm_hybrid.add_batch_uuid(document, user_id, combine_ids, uuid, collection, collection_name, note_type, meeting_date, time_zone)

        elif type_ == "CMV":
            # C72 -> entity
            combine_ids = collection
            uid = llm_hybrid.add_batch_uuid(document, user_id, collection, uuid, collection, collection_name, note_type, meeting_date, time_zone)
        elif type_ == "INV":
            combine_ids = entity
            uid = llm_hybrid.add_batch_uuid(document, user_id, entity, uuid, collection, collection_name, note_type, meeting_date, time_zone)
        else:
            return Response({'error': 'No such type!'}, status=status.HTTP_400_BAD_REQUEST)

        if 'file_upload' in request.POST:

            if request.POST.get('file_upload') != '':

                file_name = request.POST.get('file_upload')

                if awsb.verify_file_ext(file_name) is False:
                    logger.error("VIEW ADD VEC TEXT FILE: FILE EXTENSION NOT SUPPORTED!!!")
                    return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                if awsb.download_object_file(file_name) is False:
                    logger.error("VIEW ADD VEC TEXT FILE: CANNOT DOWNLOAD FILE!!!")
                    return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                file_ext = slice_document.return_ext(file_name)

                if file_ext == "pptx":
                    if slice_document.check_ppt(file_name) is False:
                        logger.error("VIEW ADD VEC TEXT FILE: PPTX LIMIT!!!")
                        return Response({'error': 'Max 40 slides allowed!'}, status=status.HTTP_400_BAD_REQUEST)

                if file_ext == "csv":
                    if slice_document.check_csv(file_name) is False:
                        logger.error("VIEW ADD VEC TEXT FILE: CSV LIMIT!!!")
                        return Response({'error': 'Max 500 rows allowed for csv!'}, status=status.HTTP_400_BAD_REQUEST)

                if file_ext == "xlsx":
                    if slice_document.check_excel(file_name) is False:
                        logger.error("VIEW ADD VEC TEXT FILE: XLSX LIMIT!!!")
                        return Response({'error': 'Max 500 rows and 3 sheets allowed for xlsx!'},
                                        status=status.HTTP_400_BAD_REQUEST)

                documents = slice_document.chunk_document(file_name)
                # print(uid.split('-'))
                uuid_file = "file-" + uid.split('-')[4] + uid.split('-')[4] + uid.split('-')[3] + uid.split('-')[2] + \
                            uid.split('-')[1] + uid.split('-')[0]
                # print("FILE UUID: ", uuid_file)
                llm_hybrid.add_batch_uuid(documents, user_id, combine_ids, uuid_file, collection, collection_name,
                                          note_type, meeting_date, time_zone)

                del uuid_file
                del documents
                del collection_name
                del meeting_date
                del time_zone
                os.remove("./_tmp/" + file_name)
            else:
                logger.info("VIEW ADD VEC TEXT FILE: NO FILES!")
        else:
            logger.info("VIEW ADD VEC TEXT FILE: NO FILES!")

        del collection
        del entity
        del user_id
        del type_
        del text
        del uuid
        del document
        del pre_filter

        return Response({'msg': "Success"}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error("VIEW UPDATE ALL VEC FILE:")
        logger.error(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_collection(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = (llm_hybrid.weaviate_client.query.get(collection, ['entity', 'uuid', 'content', 'user_id', 'collection_name', 'note_type', 'meeting_date', 'time_zone'])
                       .with_additional(["id"])
                       .do())

        if 'data' in data_object:
            res = data_object['data']['Get'][collection]
        else:
            res = []

        # print(res)

        del data_object
        del collection
        del company

        return Response({'msg': res}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET COLLECTION:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_objects_entity(request):
    try:
        company = json.loads(request.body)
        entity = str(company['entity'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        # data_object = llm_hybrid.get_by_uuid('entity', entity, collection)
        return Response({'msg': 'In progress!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET OBJECTS ENTITY:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = str(company['uuid'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = llm_hybrid.filter_by_uuid(uid, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET OBJECTS UUID:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = llm_hybrid.get_by_id(collection, obj_id)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET OBJECT:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_user_objects(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        entity = str(company['entity'])
        user_id = str(company['user_id'])
        combine_id = "INP" + entity

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        res = llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "operator": "And",
                "operands": [
                        {
                            "path": ["entity"],
                            "operator": "Equal",
                            "valueText": combine_id,
                        },
                        {
                            "path": ["user_id"],
                            "operator": "Equal",
                            "valueText": user_id,
                        }
                    ]
        },
        )

        # print(res)

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE USER OBJECT:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["id"],
                "operator": "Like",
                "valueText": obj_id
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECT:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_entity(request):
    try:
        company = json.loads(request.body)
        entity = str(company['entity'])
        collection = "C" + str(company['collection'])
        combine_id = "INP" + entity

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["entity"],
                "operator": "ContainsAny",
                "valueTextArray": [entity, combine_id]
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECT ENTITY:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = str(company['uuid'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Equal",
                "valueText": uid
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECTS UUID:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_uuid_file(request):
    try:
        company = json.loads(request.body)
        uid = str(company['uuid'])
        collection = "C" + str(company['collection'])

        uuid_file = "file-" + uid.split('-')[4] + uid.split('-')[4] + uid.split('-')[3] + uid.split('-')[2] + uid.split('-')[1] + uid.split('-')[0]

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Equal",
                "valueText": uuid_file
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECTS UUID:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_all(request):
    try:
        company = json.loads(request.body)
        uid = str(company['uuid'])
        collection = "C" + str(company['collection'])

        uuid_file = "file-" + uid.split('-')[4] + uid.split('-')[4] + uid.split('-')[3] + uid.split('-')[2] + uid.split('-')[1] + uid.split('-')[0]

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "ContainsAny",
                "valueTextArray": [uid, uuid_file]
            },
        )

        return Response({'msg': 'Vector Deletion Successful!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECTS UUID:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_uuids(request):
    try:
        company = json.loads(request.body)
        uids = list(company['uuids'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        for uuid in uids:
            llm_hybrid.weaviate_client.batch.delete_objects(
                class_name=collection,
                where={
                    "path": ["uuid"],
                    "operator": "Equal", # ContainsAll
                    "valueText": uuid # ['', '', '']
                },
            )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECTS UUIDS:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_collection(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.schema.delete_class(collection)
        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE COLLECTION:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def search_hybrid(request):
    try:
        master_vector = []
        company_vector = []
        initiative_vector = []
        member_vector = []
        msv = []
        cv = []
        miv = []
        retriever = ""

        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        query = str(company['query'])
        entity = str(company['entity'])
        user_id = str(company['user_id'])
        combine_ids = "INP" + entity

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        cat = llm_hybrid.trigger_vectors(query=query)
        key = llm_hybrid.important_words(query)
        keyword_pattern = r'KEYWORD:\s*\["(.*?)"\]'
        keyword_match = re.search(keyword_pattern, key)

        if keyword_match:
            keywords_str = keyword_match.group(1)  # Get the string inside brackets
            keywords_list = [keyword.strip().strip("'") for keyword in keywords_str.split(",")]
        else:
            keywords_list = []

        print(key)
        print("KEY MATCH: ", keyword_match)
        print("KEY LIST: ", keywords_list)

        combine_ids = "INP" + entity

        if "Meeting" not in cat:

            if "Specific Domain Knowledge" in cat or \
                    "Organizational Change or Organizational Management" in cat or \
                    "Definitional Questions" in cat or \
                    "Context Required" in cat:

                if "Context Required" not in cat:
                    mode = 0

                master_vector = mv.search_master_vectors(query=query, class_="MV001")
                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids,
                                                               user_id=user_id)

            elif "Individuals" in cat or "Personal Information" in cat:
                if "Individuals" in cat:
                    # query = query.title()
                    mode = 0
                if "Personal Information" in cat:
                    mode = 0.2

                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids,
                                                               user_id=user_id)

                if 2 <= len(keywords_list) <= 3:

                    for keyword in keywords_list:
                        r1 = mv.search_master_vectors(query=keyword, class_="MV001")
                        r2 = llm_hybrid.search_vectors_company(query=keyword, entity=collection,
                                                                           class_=collection)
                        r3 = llm_hybrid.search_vectors_initiative(query=keyword, entity=entity,
                                                                                 class_=collection)
                        r4 = llm_hybrid.search_vectors_user(query=keyword, class_=collection,
                                                                       entity=combine_ids,
                                                                       user_id=user_id)

                        r3.extend(r4)

                        msv.extend(r1)
                        cv.extend(r2)
                        miv.extend(r3)

            if "Greeting" in cat:
                mode = 0.4

            initiative_vector.extend(member_vector)

            master_vector.extend(msv)
            company_vector.extend(cv)
            initiative_vector.extend(miv)

            print("MASTER VEC 1: ", msv)
            print("MASTER VEC 2: ", master_vector)

            top_master_vec = mv.reranker(query=query, batch=master_vector, return_type=list)
            top_company_vec = llm_hybrid.reranker(query=query, batch=company_vector, class_=collection, return_type=list)
            top_member_initiative_vec = llm_hybrid.reranker(query=query, batch=initiative_vector, top_k=10,
                                                            class_=collection, return_type=list)

            retriever = f"MASTER VEC: {top_master_vec}, COMPANY VEC: {top_company_vec}, MEMBER VEC: {top_member_initiative_vec}"
        else:
            print("Searching Meeting Vectors!")
            mode = 0

            user_meeting_vec = llm_hybrid.search_vectors_user_type(query, collection, combine_ids, user_id, "Meeting")
            initiative_meeting_vec = llm_hybrid.search_vectors_company_type(query, collection, entity, "Meeting")
            company_meeting_vec = llm_hybrid.search_vectors_company_type(query, collection, collection, "Meeting")

            initiative_meeting_vec.extend(user_meeting_vec)
            company_meeting_vec.extend(initiative_meeting_vec)

            matched_meetings = llm_hybrid.reranker(query, collection, company_meeting_vec)

            retriever = f"{matched_meetings}"

        return Response({'msg': retriever}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW SEARCH HYBRID:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ------------------------- BACKUP -------------------------
@api_view(http_method_names=['POST'])
def add_bucket_object(request):
    try:
        bucket = json.loads(request.body)
        collection = "C" + bucket['collection']

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = (llm_hybrid.weaviate_client.query.get(collection, ['entity', 'uuid', 'content', 'user_id', 'collection_name', 'note_type', 'meeting_date', 'time_zone'])
                       .with_additional(["id"])
                       .do())

        tmp = data_object['data']['Get']
        check = awsb.upload_json(f"{settings.BUKET_NAME}-{collection}", tmp)

        if check is True:
            return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        print("ADD BUCKET OBJECT: ")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ------------------------- Master Vectors -------------------------
@api_view(http_method_names=['POST'])
def create_master_collection(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is True:
            return Response({'error': 'This collection already exists!'}, status=status.HTTP_400_BAD_REQUEST)

        class_obj = {
            "class": f"{company['collection']}",
            "description": f"collection for {company['collection']}",
            "vectorizer": "text2vec-cohere",
            "properties": [
                {
                    "name": "uuid",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True,
                        }
                    }
                },
                {
                    "name": "filename",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True,
                        }
                    }
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "vectorizePropertyName": True,
                            "model": "embed-multilingual-v3.0",
                        }
                    }
                },
                {
                    "name": "type",  # EA, Finance data, block chains, .....
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True
                        }
                    }
                }
            ],
        }

        mv.weaviate_client.schema.create_class(class_obj)
        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_collection(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = (mv.weaviate_client.query.get(company['collection'], ['uuid', 'filename', 'content', 'type'])
                       .with_additional(["id"])
                       .do())

        res = data_object['data']['Get']
        # print(res)
        return Response({'msg': res}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_objects_filename(request):
    try:
        company = json.loads(request.body)
        filename = company['filename']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.filter_by('filename', filename, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_objects_type(request):
    try:
        company = json.loads(request.body)
        type_ = company['type']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.filter_by('type', type_, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = company['uuid']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.filter_by('uuid', uid, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.get_by_id(collection, obj_id)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["id"],
                "operator": "Like",
                "valueText": obj_id
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_collection(request):
    try:
        company = json.loads(request.body)
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.schema.delete_class(collection)
        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = company['uuid']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Like",
                "valueText": uid
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_objects_file(request):
    try:
        company = json.loads(request.body)
        filename = company['filename']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["filename"],
                "operator": "Like",
                "valueText": filename
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def add_master_vectors(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        documents = slice_document.chunk_corpus(company['text'])

        uid = mv.add_batch(documents, company['filename'], company['type'], company['collection'])

        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def upload_master_file(request):
    try:
        data = request.data
        documents = data.get('chunks')
        file_name = data.get('filename')
        doc_type = data.get('type')
        collection = data.get('collection')

        if not documents or not file_name or not doc_type or not collection:
            return Response({'msg': 'Invalid data'}, status=status.HTTP_400_BAD_REQUEST)

        # Add batch processing logic
        uid = mv.add_batch(documents, file_name, doc_type, collection)

        return Response({'msg': uid}, status=status.HTTP_201_CREATED)
    except Exception as e:
        print("VIEW MASTER UPLOAD FILE:")
        print(e)
        return Response({'msg': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ----------- WARNING -----------
@api_view(http_method_names=['POST'])
def destroy_all(request):
    llm_hybrid.weaviate_client.schema.delete_all()
    return Response({'msg': 'Destroyed!!!'}, status=status.HTTP_200_OK)


# ----------- BACKUP -----------
@api_view(http_method_names=['POST'])
def backup_master_vectors(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = (mv.weaviate_client.query.get(company['collection'], ['uuid', 'filename', 'content', 'type'])
                       .with_additional(["id"])
                       .do())

        res = data_object['data']['Get']

        with open(f"./_tmp/{company['collection']}.json", "w") as jfile:
            jfile.write(json.dumps(res))

        awsb.upload_file(file_name=f"./_tmp/{company['collection']}.json", bucket=settings.BUCKET_NAME, object_name=f"{company['collection']}.json")
        os.remove(f"./_tmp/{company['collection']}.json")

        return Response({'msg': 'Successfully Created Backup!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW BACKUP MASTER VEC:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def restore_master_vectors(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        jdata = weaviate_backup.read_object(f"{company['collection']}.json")

        mv.add_data(jdata, company['collection'])

        return Response({'msg': 'Successfully Restored!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def async_backup(collection: str) -> str:
    data_object = (llm_hybrid.weaviate_client.query.get(collection, ['entity', 'uuid', 'content', 'user_id', 'collection_name', 'note_type', 'meeting_date', 'time_zone'])
                   .with_additional(["id"])
                   .do())

    if 'data' in data_object:
        res = data_object['data']['Get']

        with open(f"./_tmp/{collection}.json", "w") as jfile:
            jfile.write(json.dumps(res))

        awsb.upload_file(file_name=f"./_tmp/{collection}.json", bucket=settings.BUCKET_NAME,
                         object_name=f"{collection}.json")
        os.remove(f"./_tmp/{collection}.json")

        return f"Backup for {collection} created successfully!"

    else:
        return f"Cannot create backup. There was no data in {collection}!"


async def async_collection(data: list) -> str:
    tasks = []
    for collection in data:
        task = asyncio.create_task(async_backup(collection))  # Create async task
        tasks.append(task)

    # Run all tasks simultaneously and gather results
    results = await asyncio.gather(*tasks)
    logger.info("All tasks completed for backups!")
    return "\n".join(results)


async def start_backup(data):
    results = await async_collection(data)
    logger.info(results)


@api_view(http_method_names=['POST'])
def backup_company_vectors(request):
    try:
        response = llm_hybrid.weaviate_client.schema.get()
        data = [class_['class'] for class_ in response['classes'] if class_['class'][0] != "M"]
        logger.info(data)
        asyncio.run(start_backup(data))

        return Response({'msg': 'Successfully Created Backup!'}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"VIEW BACKUP COMPANY VEC: {e}")
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def async_restore(collection: str) -> str:
    name = collection.split('.')[0]
    print(name, collection)
    jdata = weaviate_backup.read_object(collection)

    if llm_hybrid.collection_exists(name) is False:
        logger.info(f"There was no collection for {name}, so i created it!")
        llm_hybrid.add_collection(name)

    await asyncio.sleep(0.05)
    llm_hybrid.add_data(jdata, name)

    return f"{collection} restored successfully!"


async def async_objects(data: list) -> str:
    tasks = []
    for collection in data:
        task = asyncio.create_task(async_restore(collection))  # Create async task
        tasks.append(task)

    # Run all tasks simultaneously and gather results
    results = await asyncio.gather(*tasks)
    logger.info("All tasks completed for Restore!")
    return "\n".join(results)


async def start_restore(data):
    results = await async_objects(data)
    logger.info(results)


@api_view(http_method_names=['POST'])
def restore_company_vectors(request):
    try:
        objects = weaviate_backup.get_object_list()

        if objects is None:
            return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        obs = [obj for obj in objects if obj[0] != "M"]

        asyncio.run(start_restore(obs))

        return Response({'msg': 'Successfully Restored!'}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"VIEW RESTORE COMPANY BACKUP: {e}")
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
