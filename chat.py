from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceBgeEmbeddings
#from langchain.document_loaders import PyPDFLoader
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from mangum import Mangum
from langchain_community.llms import VLLMOpenAI
import os
import uvicorn
import json


###########################

#this is to test adding chat history 

###########################


# run with uvicorn energy_bot:app
app = FastAPI()
# create a dependencies directory for deployment: pip3 install -t dependencies -r requirements.txt
# remove the dependencies not imported or built in those dependencies imported. Then copy templates dir and paste into created dependencies folder 
# then zip into lambda artifacts files: (cd dependencies; zip ../aws_lambda_artifact.zip -r .) 
# this will create a lambda artifact zip file 
# then lastly, zip the main python script usually named main.py but here this is the energy_bot.py: zip aws_lambda_artifact.zip -u energy_bot.py
# this adds the script into the lambda artifact zip 
handler = Mangum(app)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# here you want the model path from your container whcih should be your link in HF
model_name = 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ'
# attach a v1 at the to mimic the completions endpoint from openai 
# you can also do curl <endpoint_url>/v1/models to test the endpoint. See documentation for other endpoints like completions (finishes your sentences)
vllm_endpoint = "https://c8i1zgqqjmwy42-8000.proxy.runpod.net/v1"

llm = VLLMOpenAI(
    openai_api_key='EMPTY',
    openai_api_base=vllm_endpoint,
    model_name=model_name,
    max_tokens=512, 
    temperature=0.3,
    model_kwargs={'stop':["\n\nHuman"]},
)


# local_llm = "mistral-7b-instruct-v0.1.Q4_K_S.gguf"

# config = {
# 'max_new_tokens': 1024,
# 'repetition_penalty': 1.1,
# 'temperature': 0.1,
# 'top_k': 50,
# 'top_p': 0.9,
# 'stream': True,
# 'threads': int(os.cpu_count() / 2)
# }

# llm = CTransformers(
#     model=local_llm,
#     model_type="mistral",
#     lib="avx2", #for CPU use
#     **config
# )

print(f"LLM Initialized....{llm}\n")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

email_generation_template = """You are a helpful assistant who generates emails supporting oil and gas, pipeline or coal projects.
A user will pass in information about them, first their name, then their bio information and you should generate a thoughtful, unique and moderate answer email using the 
user's unique information and perspective for this specific project. Start the email with 'To Whom it may concern' or 'Reviewers of the {project} Project' and have 
the user's name: {user_name} signed at the end of the email. Only return a sample email supporting the project on behalf of the user advocating for its support, nothing more. 
USER name: {user_name}
USER Info: {user_info}
ANSWER="""

email_system_prompt = PromptTemplate(template=email_generation_template, input_variables=['project', 'user_name', 'user_info'])
email_generation_chain = LLMChain(llm=llm, prompt=email_system_prompt)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/energy_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})

# this is for passing previous questions to the model for conversation history context
chat_log = []

# this is for displaying the conversation history on the frontend 
chat_responses = []


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# so here the js script on the webpage grabs the data.answer and data.source_document and data.doc
@app.post("/get_response")
async def get_response(query: str = Form(...)):
    print(f'\n\n ***** this is what the form looks like: {query} \n\n')
    chat_log.append({'role': 'user', 'content': query})
    chat_responses.append(query)
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    print(response)
    answer = response['result']
    chat_responses.append(answer)
    chat_log.append({'role': 'assistant', 'content': answer})
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
    res = Response(response_data)
    return res


@app.post("/get_email_response")
async def get_email_response(project: str = Form(...), name: str = Form(...), info: str = Form(...)):
    print(f'info: {info} name: {name}, project {project}\n')
    email_generation_output = email_generation_chain.run({'project': project, 'user_name': name, 'user_info': info})
    print(email_generation_output)
    print(f'this email type is: ** {type(email_generation_output)} ** \n\n')
    #if isinstance(email_generation_output, set): email_generation_list = list(email_generation_output)
    
    email_output_dict = {'email_generation_output': email_generation_output}
    email_output_json_string = json.dumps(email_output_dict)
    print(f'\n\n ****  this is the email output dict of the data in json: {email_output_json_string}\n\n ')
    email_output_data = jsonable_encoder(email_output_json_string)
    print(f'\n\n ****  this is the email response of the data in json: {email_output_data}\n\n ')
    email_response = Response(email_output_data)
    return email_response 


# if __name__ == '__main__':
#     uvicorn.run('main:app', host='0.0.0.0', port=8000)