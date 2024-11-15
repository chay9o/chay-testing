import os
import json
import logging
import requests
import time
import anthropic
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RetryOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from .retriever import LLMHybridRetriever
from .master_vectors.MV import MasterVectors
from .chunker import ChunkText
from .tasks import send_data_to_webhook
from jinja2 import Template
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio
from langchain.callbacks.base import BaseCallbackHandler
import re
import aiohttp
import sys


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.basicConfig(filename="query_counts.log", level=logging.INFO)


# Asynchronous logging
executor = ThreadPoolExecutor(max_workers=2)

job_done = "STOP"

def log_info_async(message):
    executor.submit(logger.info, message)

class SimpleCallback(BaseCallbackHandler):
    def __init__(self, q: asyncio.Queue):
        self.q = q

    async def on_llm_start(self, serialized, prompts, **kwargs):
        pass

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.q.put(token)

    async def on_llm_end(self, *args, **kwargs):
        await self.q.put(job_done)


if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'main.settings')
load_dotenv()

# Load prompt templates
with open("system-prompt.txt", "r") as file:
    SYSPROMPT = file.read()

with open("chatnote-prompt.txt", "r") as file:
    CHATNOTE_PROMPT = file.read()

with open("live-data-prompt.txt", "r") as file:
    LIVE_DATA_PROMPT = file.read()

with open("img-prompt.txt", "r") as file:
    IMGPROMPT = file.read()

with open("summary_prompt.txt", "r") as file:
    SUMMARY_PROMPT = file.read()

prompt = PromptTemplate.from_template(SYSPROMPT)
chat_note_prompt = PromptTemplate.from_template(CHATNOTE_PROMPT)
live_data_prompt = PromptTemplate.from_template(LIVE_DATA_PROMPT)
summary_prompt = PromptTemplate.from_template(SUMMARY_PROMPT)

llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.GPT_MODEL_2,
    openai_api_base=settings.BASE_URL,
    max_tokens=1000,
    streaming=True
).configurable_fields(
            temperature=ConfigurableField(
                id="llm_temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            )
        )


llm_hybrid = LLMHybridRetriever(verbose=True)
mv = MasterVectors()
slice_document = ChunkText()
chat_template = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['text'] + ' [/INST]' }}"
    "{% elif message['role'] == 'bot' %}{{ message['text'] + eos_token}}"
    "{% endif %}{% endfor %}"
)

template = Template(chat_template)


class ChatConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.que = asyncio.Queue()
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.GPT_MODEL_2,
            openai_api_base=settings.BASE_URL,
            streaming=True,
            max_tokens=1000,
            callbacks=[SimpleCallback(self.que)]
        ).configurable_fields(
            temperature=ConfigurableField(
                id="llm_temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            )
        )
    
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def generate_response(self, is_trained_data_used):
        final_response = ""
        partial_response = ""
        while True:
            next_token = await self.que.get()
            if next_token == job_done:
                break
            # Filter out tokens that include metadata or unwanted information
            if next_token.startswith('content=') or next_token.startswith('id=') or next_token.startswith('response_metadata='):
                continue
            partial_response += str(next_token)
            final_response += str(next_token)
            await self.send(text_data=json.dumps({"message": partial_response}))
            partial_response = ""
        await self.send(text_data=json.dumps({
            "message": "job done",
            "query_count": 1,  # Always 1 for a single query
            "is_trained_data_used": 1 if is_trained_data_used else 0
        }))

    async def handle_response(self, response, is_trained_data_used):
        final_response = ""
        async for res in self._convert_to_async_iter(response):
            res_text = str(res)
            if res_text.startswith('content=') or res_text.startswith('id=') or res_text.startswith('response_metadata='):
                continue
            final_response += res_text
        await self.send(text_data=json.dumps({
            "message": final_response,
            "query_count": 1,  # Always 1 for a single query
            "is_trained_data_used": 1 if is_trained_data_used else 0
        }))
        await self.send(text_data=json.dumps({"message": "job done"}))

    async def _convert_to_async_iter(self, generator):
        for item in generator:
            yield item
            await asyncio.sleep(0)


    async def receive(self, text_data=None, bytes_data=None):
        logging.info("Received message in ChatConsumer")
        retriever = ""

        master_vector = []
        company_vector = []
        initiative_vector = []
        member_vector = []

        msv = []
        cv = []
        miv = []

        user_meeting_vec = []
        initiative_meeting_vec = []
        company_meeting_vec = []

        data = json.loads(text_data)
        collection = "C" + str(data['collection'])
        query = str(data['query']).lower()
        entity = str(data['entity'])
        user_id = str(data['user_id'])
        chat_history = data.get('chatHistory', [])
        mode = str(data['mode'])
        user = data.get('user', 'Unknown User')
        language = data.get('language', 'en')
        time_zone = data.get('time_zone', '')
        current_date = data.get('current_date', '')

        company_prompt = data.get('companyPrompt', '')
        initiative_prompt = data.get('initiativePrompt', '')
        command_stop = data.get('command_stop', False)
        
        log_info_async(f"data: {data}")


        # Flag for trained data usage
        is_trained_data_used = False

        if not llm_hybrid.collection_exists(collection):
            await self.send(text_data=json.dumps({'error': 'This collection does not exist!'}))
            return

        if command_stop is True:
            await self.close()
            return

        cat = llm_hybrid.trigger_vectors(query=query)
        key = llm_hybrid.important_words(query=query)

        keyword_pattern = r'KEYWORD:\s*\["(.*?)"\]'
        keyword_match = re.search(keyword_pattern, key)

        if keyword_match:
            keywords_str = keyword_match.group(1)  # Get the string inside brackets
            keywords_list = [keyword.strip().strip("'") for keyword in keywords_str.split(",")]
        else:
            keywords_list = []

        print(key)

        combine_ids = "INP" + entity

        if "Meeting" not in cat:

            if "Specific Domain Knowledge" in cat or \
                    "Organizational Change or Organizational Management" in cat or \
                    "Definitional Questions" in cat or \
                    "Context Required" in cat:

                master_vector = mv.search_master_vectors(query=query, class_="MV001")
                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids, user_id=user_id)
                
                if company_vector or initiative_vector or member_vector:
                    is_trained_data_used = True


            elif "Individuals" in cat or "Personal Information" in cat:

                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids, user_id=user_id)
                
                if company_vector or initiative_vector or member_vector:
                    is_trained_data_used = True


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

            initiative_vector.extend(member_vector)

            master_vector.extend(msv)
            company_vector.extend(cv)
            initiative_vector.extend(miv)

            final_query = f"{query}, {' '.join(keywords_list)}"

            top_master_vec = mv.reranker(query=final_query, batch=master_vector, return_type=str)
            top_company_vec = llm_hybrid.reranker(query=final_query, batch=company_vector, class_=collection, return_type=str)
            top_member_initiative_vec = llm_hybrid.reranker(query=final_query, batch=initiative_vector, top_k=10, class_=collection, return_type=str)

            retriever = f"{top_master_vec} {top_company_vec} {top_member_initiative_vec}"

        else:
            print("Searching Meeting Vectors!")

            tmp = llm_hybrid.date_filter(query, current_date)

            date_pattern = r"FILTER:\s*(\d{4}-\d{1,2}-\d{1,2})"
            query_pattern = r"QUERY:\s*\[(.*?)\]"

            date_match = re.search(date_pattern, tmp)
            query_match = re.search(query_pattern, tmp)

            # Extract values
            if date_match:
                date_value = date_match.group(1)
            else:
                date_value = ""

            if query_match:
                query_value = query_match.group(1)
            else:
                query_value = ""
            print("Date:", date_value)
            print("Query:", query_value)

            if date_value != "" and query_value == "":
                user_meeting_vec = llm_hybrid.search_vectors_user_type(date_value, collection, combine_ids, user_id, "Meeting")
                initiative_meeting_vec = llm_hybrid.search_vectors_company_type(date_value, collection, entity, "Meeting")
                company_meeting_vec = llm_hybrid.search_vectors_company_type(date_value, collection, collection, "Meeting")

                initiative_meeting_vec.extend(user_meeting_vec)
                company_meeting_vec.extend(initiative_meeting_vec)

                retriever = "\n".join(company_meeting_vec) if len(company_meeting_vec) > 0 else ""

            elif query_value != "" and date_value != "":
                user_meeting_vec = llm_hybrid.search_vectors_user_type(date_value, collection, combine_ids, user_id, "Meeting")
                initiative_meeting_vec = llm_hybrid.search_vectors_company_type(date_value, collection, entity, "Meeting")
                company_meeting_vec = llm_hybrid.search_vectors_company_type(date_value, collection, collection, "Meeting")

                initiative_meeting_vec.extend(user_meeting_vec)
                company_meeting_vec.extend(initiative_meeting_vec)

                retriever = "\n".join(company_meeting_vec) if len(company_meeting_vec) > 0 else ""

            elif query_value != "" and date_value == "":
                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids, user_id=user_id)

                initiative_vector.extend(member_vector)
                top_master_vec = mv.reranker(query=query, batch=master_vector, return_type=str)
                top_company_vec = llm_hybrid.reranker(query=query, batch=company_vector, class_=collection, return_type=str)
                top_member_initiative_vec = llm_hybrid.reranker(query=query, batch=initiative_vector, top_k=10, class_=collection, return_type=str)

                retriever = f"{top_master_vec} {top_company_vec} {top_member_initiative_vec}"

        config = {
            'callbacks': [SimpleCallback(self.que)]
        }

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

        chain = prompt | self.llm.with_config(configurable={"llm_temperature": mode})

        response = chain.stream({'matching_model': retriever,
                                 'question': query,
                                 'username': user,
                                 'chat_history': chat_history_str,
                                 'language_to_use': language,
                                 'current_date': current_date,
                                 'time_zone': time_zone,
                                 'company_prompt': company_prompt,
                                 'initiative_prompt': initiative_prompt}, config=config)

        task_1 = asyncio.create_task(self.handle_response(response, is_trained_data_used))
        task_2 = asyncio.create_task(self.generate_response(is_trained_data_used))

        await task_1
        await task_2



class LiveDataChatConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.que = asyncio.Queue()
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.GPT_MODEL_2,
            openai_api_base=settings.BASE_URL,
            streaming=True,
            max_tokens=1000,
            callbacks=[SimpleCallback(self.que)]
        )

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def generate_response(self):
        final_response = ""
        partial_response = ""
        while True:
            next_token = await self.que.get()
            if next_token == job_done:
                break
            # Filter out tokens that include metadata or unwanted information
            if next_token.startswith('content=') or next_token.startswith('id=') or next_token.startswith('response_metadata='):
                continue
            partial_response += str(next_token)
            final_response += str(next_token)
            await self.send(text_data=json.dumps({"message": partial_response}))
            partial_response = ""
        await self.send(text_data=json.dumps({"message": "job done"}))

    async def handle_response(self, response):
        final_response = ""
        async for res in self._convert_to_async_iter(response):
            res_text = str(res)
            # Continue filtering if res_text is a string
            if res_text.startswith('content=') or res_text.startswith('id=') or res_text.startswith('response_metadata='):
                continue
            final_response += res_text
        await self.send(text_data=json.dumps({"message": final_response}))
        await self.send(text_data=json.dumps({"message": "job done"}))

    async def _convert_to_async_iter(self, generator):
        for item in generator:
            yield item
            await asyncio.sleep(0)  # yield control to the event loop

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        query = str(data['query'])
        user_id = str(data['user_id'])
        chat_history = data.get('chatHistory', [])
        mode = str(data['mode'])
        user = data.get('user', 'Unknown User')
        language = data.get('language', '')

        company_prompt = data.get('companyPrompt', '')
        initiative_prompt = data.get('initiativePrompt', '')


        log_info_async(f"Received query: {query}")
        log_info_async(f"User ID: {user_id}")
        
        api_start_time = time.time()
        url = "https://copilot5.p.rapidapi.com/copilot"
        payload = {
            "message": query,
            "conversation_id": None,
            "tone": "PRECISE",
            "markdown": False,
            "photo_url": None
        }
        headers = {
            "x-rapidapi-key": "f1bbf996eemsh3883adcdaffb1c1p1056f6jsnda01f87a24f8",
            "x-rapidapi-host": "copilot5.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        log_info_async(f"API response data: {response_data}")

        if 'data' not in response_data:
            await self.send(text_data=json.dumps({'error': 'Invalid response from API'}))
            return

        vectors_data = response_data['data']['message']
        api_end_time = time.time()

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
        username = data.get('user', 'Unknown User')
        input_data = {
            'question': query,
            'username': username,
            'chat_history': chat_history_str,
            'Vectors_here': vectors_data,
            'language_to_use': language,
        }

        chain = live_data_prompt | self.llm.with_config(configurable={"llm_temperature": mode})
        response = chain.stream({
            **input_data,
            'company_prompt': company_prompt,
            'initiative_prompt': initiative_prompt
        })

        task_1 = asyncio.create_task(self.handle_response(response))
        task_2 = asyncio.create_task(self.generate_response())

        await task_1
        await task_2


class ChatNoteConsumer(AsyncWebsocketConsumer):
    print("hi")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.que = asyncio.Queue()
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.GPT_MODEL_2,
            openai_api_base=settings.BASE_URL,
            streaming=True,
            max_tokens=1000,
            callbacks=[SimpleCallback(self.que)]
        )
        self.prompt = chat_note_prompt

    async def process_question(self, question: str, username: str, language: str, retriever: str):
        chain = LLMChain(llm=self.llm, prompt=self.prompt)

        await chain.arun(
            matching_model=retriever,
            username=username,
            question=question,
            language_to_use=language
        )

    async def generate_response(self):
        txt = ""
        while True:
            next_token = await self.que.get()  # Blocks until an input is available
            if next_token is job_done:
                await self.send(text_data=json.dumps({"msg": "job done"}))
                break
            txt += next_token
            await self.send(text_data=json.dumps({"msg": next_token}))
            await asyncio.sleep(0.01)
            self.que.task_done()

    async def start(self, query: str, name: str, language: str, retriever: str):
        task_1 = asyncio.create_task(
            self.process_question(question=query, username=name, language=language, retriever=retriever))
        task_2 = asyncio.create_task(self.generate_response())

        await task_1
        await task_2

    async def connect(self):
        await self.accept()

    async def disconnect(self, code):
        logger.info("Disconnected from chat note!")

    async def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data)
            collection = "C" + str(data['collection'])
            query = str(data['query'])
            entity = str(data['entity'])
            user_id = str(data['user_id'])
            user = data.get('user', 'Unknown User')
            language = data.get('language', 'en')

            company_prompt = data.get('companyPrompt', '')
            initiative_prompt = data.get('initiativePrompt', '')

            log_info_async(f"Received query: {query}")
            log_info_async(f"User ID: {user_id}")

            if not llm_hybrid.collection_exists(collection):
                await self.send(text_data=json.dumps({'error': 'This collection does not exist!'}))
                return

            cat = llm_hybrid.trigger_vectors(query=query)

            master_vector = []
            company_vector = []
            initiative_vector = []
            member_vector = []

            combine_ids = "INP" + entity

            if "Specific Domain Knowledge" in cat or \
                    "Organizational Change or Organizational Management" in cat or \
                    "Definitional Questions" in cat or \
                    "Context Required" in cat:

                master_vector = mv.search_master_vectors(query=query, class_="MV001")
                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids,
                                                               user_id=user_id)
            elif "Individuals" in cat or "Personal Information" in cat:
                company_vector = llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
                initiative_vector = llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
                member_vector = llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids,
                                                               user_id=user_id)

            initiative_vector.extend(member_vector)
            top_master_vec = mv.reranker(query=query, batch=master_vector)
            top_company_vec = llm_hybrid.reranker(query=query, batch=company_vector, class_=collection)
            top_member_initiative_vec = llm_hybrid.reranker(query=query, batch=initiative_vector, top_k=10,
                                                            class_=collection)

            retriever = f"{top_master_vec}\n{top_company_vec}\n{top_member_initiative_vec}"

            await self.start(query=query, name=user, language=language, retriever=retriever)
        except Exception as e:
            await self.send(text_data=json.dumps({'error': 'Something went wrong!'}))
            logger.info(f"CHATNOTE ERROR: {e}")
            return


class ImageQueryConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.que = asyncio.Queue()
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        self.response_task = None  # Track async tasks for proper cleanup


    async def connect(self):
        log_info_async("WebSocket connection established.")
        await self.accept()

    async def disconnect(self, close_code):
        log_info_async(f"WebSocket connection closed with code: {close_code}")
        await self.que.put('job_done')  # Signal to stop generating the response

        # Cancel async tasks related to the WebSocket if still running
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await self.response_task  # Wait for cancellation
            except asyncio.CancelledError:
                log_info_async("Response task was cancelled.")

    async def generate_response(self):
        final_response = ""
        while True:
            next_token = await self.que.get()
            if next_token == job_done:
                break
            final_response += str(next_token)
            await self.send(text_data=json.dumps({"message": final_response}))
            final_response = ""
        await self.send(text_data=json.dumps({"message": "job done"}))

    async def handle_response(self, response):
        final_response = ""
        async for res in self._convert_to_async_iter(response):
            res_text = str(res)
            if res_text.startswith('content=') or res_text.startswith('id=') or res_text.startswith('response_metadata='):
                continue
            final_response += res_text
        await self.send(text_data=json.dumps({"message": final_response}))
        await self.send(text_data=json.dumps({"message": "job done"}))

    async def _convert_to_async_iter(self, generator):
        for item in generator:
            yield item
            await asyncio.sleep(0)  # yield control to the event loop

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        query = str(data['query'])
        image_data = data['image']  # Assuming image is sent as base64 string
        image_filename = data['image_filename']
        language = data.get('language', 'en') 

        log_info_async(f"Received query: {query}")
        log_info_async(f"Received image data: {image_data[:30]}...")  # Log partial image data for brevity

        if not self.api_key:
            await self.send(text_data=json.dumps({'error': 'API key not found in environment variables.'}))
            return

        # Initialize Claude API client
        client = anthropic.Anthropic(api_key=self.api_key)

        # Define the image media type (adjust if necessary)
        if image_filename.lower().endswith('.png'):
            image_media_type = "image/png"
        elif image_filename.lower().endswith(('.jpg', '.jpeg')):
            image_media_type = "image/jpeg"
        else:
            await self.send(text_data=json.dumps({'error': 'Unsupported image format.'}))
            return
        system_prompt = IMGPROMPT.replace("{language}", language)
        # Create the message with the base64-encoded image
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Please analyze the content of the image file named '{image_filename}' and provide a detailed explanation. Additionally, create a possible title for this image."
                        }
                    ],
                }
            ],
        )

        log_info_async("Message sent to Claude API for processing.")

        
        # Handle the response
        
        response_blocks = []
        for block in message.content:
            if block.type == 'text':
                response_blocks.append(block.text)
                #print(block.text)

        # Run response handling and generation tasks
        task_1 = asyncio.create_task(self.handle_response(response_blocks))
        task_2 = asyncio.create_task(self.generate_response())

        # Track tasks for cleanup
        self.response_task = asyncio.gather(task_1, task_2)

        try:
            # Add a timeout for tasks to avoid hanging during WebSocket closure
            await asyncio.wait_for(self.response_task, timeout=20)
        except asyncio.TimeoutError:
            log_info_async("Timeout while waiting for response generation tasks to complete.")

        log_info_async("Response handling and generation tasks completed.")


class SummaryGenerator(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.que = asyncio.Queue()
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.GPT_MODEL_1,
            openai_api_base=settings.BASE_URL,
            streaming=True,
            max_tokens=4000,
            callbacks=[SimpleCallback(self.que)]
        )
        self.prompt = summary_prompt
        self.summary_types = {
            "Bullet Points Summary": "Create a concise bullet points summary, Do not add any headings. Bullet points summary should be numeric and separated after each point.",

            "Concise Overview": "Provide a very brief, one-sentence summary of the main idea of the text. In 3-5 sentences, provide an overview that captures the article's main topic, central argument, and key supporting points.",

            "Key Insights": "Summarize this text by extracting and emphasizing the most critical insights.",

            "Key Points Summary": "Summarize this text in keypoints summary, focusing on essential points. include the reasons and evidence (key points). Add main idea of this summary. Add headings if necessary. create format in key points separated.",

            "Podcast Summary": "Summarize this text by creating a podcast script that features a discussion between two fictional characters (Hanna Prodigy Interviewing and PETER the interviewee), focusing on storytelling and maintaining a natural, conversational flow",

            "Q & A Format": "Summarize this text by creating a series of questions and answers that highlight the main points and key information.",

            "Risk Assessment Summary": "Summarize this text by identifying key risks or challenges and suggesting strategies to mitigate them.",

            "Simplified Explanation": "Summarize this text by breaking down complex ideas into simple, easy-to-understand language.",

            "Thematic Summary": "Summarize this text by organizing key points into distinct themes or categories.",

            "Value-Based Summary": "Summarize this text by categorizing key points into three areas of value: value for the customer, value for the company, and value for workforce wellbeing.",

            "Using the ATOM model": """Summarize the previous text using the ATOM model by categorizing key information into four quadrants: increase revenue (delighting or disrupting to increase market share and size), protect revenue (improvements to sustain current market share and revenue), reduce costs (efficiency improvements to reduce current costs), and avoid costs (sustaining improvements to avoid future costs). Additionally, provide suggestions on how to maintain or improve organizational health while aligning the company with the new situation.
Add the link to the ATOM model at the end: https://enterpriseagility.community/atom-model-qxkkkws0b6""",

            "Using the EA framework": """Summarize the previous text using the Enterprise Agility Framework (EAF) by categorizing key information into its five types of agility: technical agility, structural agility, outcomes agility, social agility, and mental agility.
Additionally, incorporate the external components of the framework, including:
Strategic Innovation: Analyze both short-term (0 to 12 months) and long-term (12 to 36 months) strategies.
Exponential Markets: Discuss strategies for effectively navigating these markets.
Individuals: Provide insights on how individuals can manage stress in the current environment while staying aligned with the overall strategy and engaging in sustainable practices.
Please highlight how these elements facilitate accelerated change and empower individuals within the organization while keeping stress levels low.
Add the link to the Enterprise Agility Framework at the end: https://enterpriseagility.community/enterprise-agility-framework-eaf-z13hdmz25t""",

            "Using the Change Journey Pyramid": """Summarize the the previous text using the Change Journey Pyramid (CJP), which is a model designed to address organizational change. The pyramid consists of five levels of mindset, from the bottom to the top:

1.	I want things to be as I say. I don't care (bottom mindset)
2.	I don't like the proposed change
3.	I don't understand why the change is happening
4.	I don't understand why we must change
5.	I want to change (top mindset)
In your summary, explain how the Change Journey Pyramid can effectively manage resistance to change and enhance employee mobilization during periods of rapid transformation. Add the link to the Enterprise Agility Framework at the end: https://enterpriseagility.community/change-journey-pyramid-cjp-mv4qp5p7b7""",

            "Using the Fasten Factors": """Summarizes the text by analyzing key information through the dimensions of the FASTEN factors to understand the market dynamics a company faces, particularly in fast-changing environments, while also evaluating technical factors. Summarize this text using the FASTEN factors framework by categorizing key information into the following dimensions:
Future Thinking (F): Evaluate how forward-looking strategies are implemented, including any relevant technical factors. Adaptability (A): Assess the company’s ability to adjust to market changes, focusing on technical adaptability. Sustainability (S): Examine practices that promote long-term environmental and economic viability, including technical aspects of sustainability. Technology (T): Analyze the role of technology in driving innovation and efficiency, emphasizing technical advancements. Experience (E): Consider the experiences that contribute to customer satisfaction and engagement, including technical user experience elements. Networks (N): Discuss the importance of collaboration and partnerships in navigating the market, including any relevant technical collaborations. Highlight how these dimensions, along with technical factors, contribute to a deeper understanding of the market landscape and help the company thrive in a rapidly changing environment.
Add the link to the Fasten Factors at the end https://enterpriseagility.community/fasten-factors-n84fcckl1n
""",
            "Using the Circle Framework for Unlearning": """Summarize this text using the Circle framework for unlearning by categorizing key information into the following dimensions:

Check (C): Evaluate current practices and mindsets that need to be assessed for unlearning.
Identify (I): Identify specific behaviors or beliefs that require change or unlearning.
Replace (R): Suggest new behaviors or beliefs to replace the outdated ones.
Connect (C): Discuss how to connect individuals and teams to foster a supportive environment for unlearning.
Learn (L): Explore the learning opportunities that arise from the unlearning process.
Empower (E): Highlight how to empower individuals within the organization to embrace unlearning and adopt new practices.
Emphasize how these dimensions contribute to a culture of unlearning and adaptability in the organization. Add the link to the Circle framework at the end: https://enterpriseagility.community/circle-framework-2kj467537d
""",

            "Using the BOIS model": """Utilize the BOIS model to summarize this text by aligning organizational behaviors with strategic objectives in a sustainable manner. Follow these steps:
1.	Behaviors: Identify the specific behaviors that need to be encouraged or changed to achieve desired outcomes.
2.	Objectives: Clearly define the objectives these behaviors support, ensuring they align with the organization’s overall mission and goals.
3.	Impact: Assess the impact of these behaviors on both individual and organizational levels, considering how they contribute to achieving objectives and enhancing organizational health.
4.	Sustainability: Develop a strategy for promoting sustainable practices that reinforce these behaviors over time.
Additionally explore Explore Incremental Alignment: Focus on small, manageable changes that can lead to significant improvements in behaviors and outcomes.
Emphasize how this holistic approach can foster a culture of continuous improvement and resilience within the organization, ultimately supporting long-term success and well-being. BOIS model at the end: https://enterpriseagility.community/bois-model-b57z339q7w"""
        }

    async def process_question(self, paragraph: str, type_: str, tone: str):
        chain = LLMChain(llm=self.llm, prompt=self.prompt)

        print(self.prompt.format_prompt(tone=tone,
            summary_type=self.summary_types[type_] if type_ in self.summary_types else "Key Points Summary",
            paragraph=paragraph))

        await chain.arun(
            tone=tone,
            summary_type=self.summary_types[type_] if type_ in self.summary_types else "Key Points Summary",
            paragraph=paragraph
        )

    async def generate_response(self):
        txt = ""
        while True:
            next_token = await self.que.get()  # Blocks until an input is available
            if next_token is job_done:
                await self.send(text_data=json.dumps({"msg": "job done"}))
                break
            txt += next_token
            await self.send(text_data=json.dumps({"msg": next_token}))
            await asyncio.sleep(0.01)
            self.que.task_done()

    async def start(self, paragraph: str, type_: str, tone: str):
        task_1 = asyncio.create_task(self.process_question(paragraph, type_, tone))
        task_2 = asyncio.create_task(self.generate_response())

        await task_1
        await task_2

    async def connect(self):
        await self.accept()

    async def disconnect(self, code):
        logger.info("Disconnected from summary generator!")

    async def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data)
            paragraph = str(data['paragraph'])
            tone = str(data['tone'])
            summary_type = str(data['summary_type'])

            logger.info(f"summary tone: {tone}. summary type: {summary_type}")

            await self.start(paragraph=paragraph, tone=tone, type_=summary_type)
        except Exception as e:
            await self.send(text_data=json.dumps({'error': 'Something went wrong!'}))
            logger.info(f"SUMMARY GENERATOR ERROR: {e}")
            return
