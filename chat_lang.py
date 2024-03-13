# from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
# from fastapi.responses import RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.encoders import jsonable_encoder
from langchain.llms import CTransformers
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.llms import VLLMOpenAI

from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceBgeEmbeddings
#from langchain.document_loaders import PyPDFLoader
# from fastapi import FastAPI, Request, Form, Response
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.encoders import jsonable_encoder
from mangum import Mangum

import os 
import json
import time
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import csv
from mangum import Mangum


# here you want the model path from your container whcih should be your link in HF
model_name = 'cybrtooth/TheBloke-Mistral-7B-Instruct-v0.2-GGUF' # this is actually a AWQ model - misnamed it 
# attach a v1 at the to mimic the completions endpoint from openai 
# you can also do curl <endpoint_url>/v1/models to test the endpoint. See documentation for other endpoints like completions (finishes your sentences)
#vllm_endpoint = "https://c8i1zgqqjmwy42-8000.proxy.runpod.net/v1"
vllm_endpoint ="https://35y80dcf2b4bpx-8000.proxy.runpod.net/v1"

llm = VLLMOpenAI(
    openai_api_key='EMPTY',
    openai_api_base=vllm_endpoint,
    model_name=model_name,
    max_tokens=512, 
    temperature=0.3,
    model_kwargs={'stop':["\n\nHuman"]},
)


email_generation_template = """You are a helpful assistant who generates emails supporting oil and gas, pipeline or coal projects.
A user will pass in information about them, first their name, then their bio information and you should generate a thoughtful, unique and moderate answer email using the 
user's unique information and perspective for this specific project. Start the email with 'To Whom it may concern' or 'Reviewers of the {project} Project' and have 
the user's name: {user_name} signed at the end of the email. Only return a sample email supporting the project on behalf of the user advocating for its support, nothing more. 
USER name: {user_name}
USER Info: {user_info}
ANSWER="""


email_system_prompt = PromptTemplate(template=email_generation_template, input_variables=['project', 'user_name', 'user_info'])
email_generation_chain = LLMChain(llm=llm, prompt=email_system_prompt)
project = 'Energy East'
name = 'Winnie Huang'
info = 'I am a first nations chief that supports the project'

email_generation_output = email_generation_chain.run({'project': project, 'user_name': name, 'user_info': info})
print(f'\n **** llm output: {email_generation_output} \n\n')

#response = llm.
