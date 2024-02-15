import streamlit as st 
from streamlit_chat import message
import os
import re
import sys
import torch
import base64
import textwrap
import warnings
from typing import List

from langchain_community.vectorstores import Chroma
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from constants import CHROMA_SETTINGS
from constants import Settings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings 

from accelerate import accelerator
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from langchain import PromptTemplate, LLMChain
#SimpleSequentialChain

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    BitsAndBytesConfig
)
# model classes
from falcon_class import StopGenerationCriteria
warnings.filterwarnings("ignore", category=UserWarning)


# model types and classes. Add to this the precision it should be loaded in but for now default to 8 bit 
CLASSES = {'encoder-decoder': AutoModelForSeq2SeqLM, 'causal-decoder': AutoModelForCausalLM}

# model id and type
MODEL_ARCHITECTURES = {"tiiuae/falcon-7b-instruct": 'causal-decoder', "tiiuae/falcon-7b": 'causal-docoder',
                       "LaMini-T5-738M": 'encoder-decoder'}
CHECKPOINT = 'tiiuae/falcon-7b-instruct'
PERSIST_DIRECTORY = 'db'

persist_directory = 'db'

@st.cache_resource
def load_base_model_and_tokenizer(model_id):
    warnings.filterwarnings("ignore", category=UserWarning)
    print('loading base model')
    #base_model = None
    if model_id in MODEL_ARCHITECTURES:
        print(f'did we get here?')
        print(model_id)
        #model_id_key = next(iter(model_id))
        #print(model_id_key)
        model_id_type = MODEL_ARCHITECTURES[model_id]
        try:
            model_class = CLASSES[model_id_type]
        except:
            print('could not find model class global class dictionary \n')
        if model_id_type == 'encoder-decoder':
            base_model = model_class.from_pretrained(model_id, device_map='auto', trust_remote_code=True, torch_dtype=torch.float32)
            print(f'base model: {base_model}') 
            model_tokenizer = AutoTokenizer.from_pretrained(model_id)

            return base_model, model_tokenizer
        
        elif model_id_type == 'causal-decoder':
            base_model = model_class.from_pretrained(model_id, device_map='auto', load_in_8bit=True, trust_remote_code=True)
            print(f'base model: {base_model}') 
            print(f'model class: {model_class}\n')
            model_tokenizer = AutoTokenizer.from_pretrained(model_id)

            return base_model, model_tokenizer
                
    print(f'Model type not in global dictionry classes. Model type value: {model_id_type} \n')
    return base_model, model_tokenizer
    


# define functions for loading model, creating qa bot, loading embedding model, creating vector store and embeddings 
# cache the llm using streamlit's cache resource, they also have cache data for caching csv, pdfs, etc.  
# hash_funcs={tokenizers.Tokenizer: my_hash_func}
@st.cache_resource
def model_pipeline(CHECKPOINT, _base_model, _tokenizer):
    warnings.filterwarnings("ignore", category=UserWarning)
    #base_model, model_class, model_id_key = None 
    checkpoint = CHECKPOINT
    base_model = _base_model
    model_tokenizer = _tokenizer
    if checkpoint in MODEL_ARCHITECTURES:
        #model_id_value = CLASSES[model_id]
        if MODEL_ARCHITECTURES[checkpoint] == 'encoder-decoder':
            print('Creating tokenier \n\n')

            model_pipeline = pipeline(
                'text2text-generation',
                model= base_model,
                tokenizer=model_tokenizer,
                max_length=1024,
                do_sample= True,
                temperature =0.3,
                top_p=0.95,
            )
            llm_pipeline = HuggingFacePipeline(model_pipeline)
            print(f'Model pipeline ready {llm_pipeline} \n')
            return llm_pipeline

        elif MODEL_ARCHITECTURES[checkpoint] == 'causal-decoder':
            print(f'In causal decoder logic to and creating tokenizer \n\n\n')
            # we will assume this is falcon 
            
            model_pipeline = transformers.pipeline(
            "text-generation",
            model=base_model,
            tokenizer=model_tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            )

            sequences = model_pipeline(
                "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
                max_length=500,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=model_tokenizer.eos_token_id,
            )
            print(f'Check to see if model sequences will print! \n\n\n')
            for seq in sequences:
                print(f"Result: {seq['generated_text']}")
            
            base_model = base_model.eval()
            #adjust the config for falcon 
            generation_config = base_model.generation_config
            generation_config.temperature = 0
            generation_config.num_return_sequences = 1
            generation_config.max_new_tokens = 512
            generation_config.use_cache = False
            generation_config.repetition_penalty = 1.7
            generation_config.pad_token_id = model_tokenizer.eos_token_id
            generation_config.eos_token_id = model_tokenizer.eos_token_id
            generation_config

            # set up the stopping criteria 
            # stopping criteria 
            stop_tokens = [["Human", ":"], ["AI", ":"]]
            stopping_criteria = StoppingCriteriaList(
                [StopGenerationCriteria(stop_tokens, model_tokenizer, base_model.device)]
            )
            print(f'Created stopping tokens \n\n')

            # now reset the pipeline 
            generation_pipeline = transformers.pipeline(
            model=base_model,
            tokenizer=model_tokenizer,
            return_full_text=True,
            task="text-generation",
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            )
            #print(f'New generation pipeline {generation_pipeline} \n')

            # do a print to test
            falcon_instruct_llm = HuggingFacePipeline(pipeline=generation_pipeline)
            print('Created falcon llm HF pipeline \n')
            prompt = """
                The following is a friendly conversation between a human and an energy expert AI. The AI is
                talkative and provides lots of specific details from its context.

                Current conversation:

                Human: What is the oil and gas industry?
                AI:
                """.strip()
            #output = falcon_instruct_llm(prompt)
            #print(output)

    return falcon_instruct_llm



def generate_llm_answer(user_instruction):
    output = ''
    llm_instruction = user_instruction
    checkpoint = CHECKPOINT
    #print(FALCON7BINSTRUCT)
    #llm_pipeline = None 
    model = base_model
    tokenizer = model_tokenizer
    llm_pipeline = model_pipeline(CHECKPOINT, model, tokenizer)
    #llm_pipeline, tokenizer = model_pipeline(checkpoint, base_model)
    generated_output = llm_pipeline(llm_instruction)
    # output should be a string or text 
    return generated_output


def email_button_click(project_input, user_name_input, user_info):
    # instantiate the model llm 
    falcon_instruct_llm = model_pipeline(CHECKPOINT, base_model, model_tokenizer)
    print('created pipeline for email')
    # set up the system prompt 
    email_generation_template = """You are a helpful assistant who generates emails supporting energy projects. You should generate a thoughtful,
        unique and moderate email for the user using their unique information and perspective for this specific project. Start 
        the email with 'To Whom it may concern' or 'Reviewers of the Project {project}' and have the user's name signed at the end 
        of the email. Only return a sample email that the user would send to the reviewers, nothing more. 
        USER name: {user_name}
        USER info: {user_info}
        ANSWER="""

    energy_project = project_input
    user_name = user_name_input
    #user_info = user_info_input
    email_generation_prompt = PromptTemplate(template=email_generation_template, input_variables=["project", "user_name", "user_info"])
    final_email_prompt = email_generation_prompt.format(project=energy_project, user_name=user_name, user_info=user_info)
    print(falcon_instruct_llm(final_email_prompt))
    email_gen_chain = LLMChain(llm=falcon_instruct_llm, prompt=email_generation_prompt)
    email_generation_output = email_gen_chain.run({'project': energy_project, 'user_name': user_name, 'user_info': user_info})
    print(email_generation_output)
    return email_generation_output



def letter_button_click():
    return


def conversation_button_click():
     # start a conversation chain 
    return


def display_conversation(history):
    for i in range(len(history['generated'])):
        message(history['past'][i], is_user=True, key=str(i) + "_user")
        message(history['generated'][i], key=str(i))


# once done looping through all the docs and creating loaders
@st.cache_resource
def ingest_data():
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith('pdf'): 
                print(file)
                loader = PDFMinerLoader(os.path.join(root,file))
    
    # once done looping through all the docs 
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    # so now we have split texts into chunks of 500 characters from the docs, pdfs
    # create embeddings here using the sentence transformers embedding model

    ################# check if there is a mistake here #################### 
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    #chromadb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=)
    # dbclient = chromadb.PersistentClient(Settings(chroma_db_impl="duckdb+parquet",
    #                                 persist_directory="./db", anonymized_telemetry=False
    #                             ))
    
    #energy_collection.add(documents = [])
    chromadb = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    #chromadb = Chroma.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory='./db'))
    # create a collection 
    #energy_collection = dbclient.create_collection(name='energy_docs')
    
    # this is no longer needed 
    chromadb.persist()
    chromadb=None


@st.cache_resource
def qa_llm():
    falcon_llm = model_pipeline(CHECKPOINT, base_model, model_tokenizer)
    embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    print('***************\n\n\n\n\n\n',embedding_model)
    #texts = ingest_data()
    # this should load vectordb from disk
    chromadb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    retriever = chromadb.as_retriever()
    print('\n \n ****creating qa ***** \n\n\n')
    qa = RetrievalQA.from_chain_type(llm=falcon_llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    print('\n \n **** Succesfully created qa! ***** \n\n\n')
    return qa


def process_answer(instruction):
    response = ''
    qa = qa_llm()
    generated_output = qa(instruction)
    result = generated_output['result']
    source = generated_output['source_documents']
    print(f'\n\n *** Generated outpout object {generated_output} ** \n\n\n')
    return result



# load this up once and they should be cached. This is what takes the most time but it will now be stored to hard disk
base_model, model_tokenizer = load_base_model_and_tokenizer(CHECKPOINT)


# once done looping through all the docs and creating loaders
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    checkpoint = CHECKPOINT
    
    st.title('SWIFT EnergyAI ready for your questions and requests ')
    with st.expander('🛢️ and ⛽ model AI Application. Please enter your info in the sidebar and select a task below:'):
                    st.markdown('''You can try asking this model to do tasks like 
                                  write an email or answer questions about energy.''')

    with st.sidebar:
        st.header('Input fields')
        project_input = st.text_input('Enter the project name')
        user_name = st.text_input('Enter your name')
        user_info_input = st.text_input('Enter your info that you want included in the email')
        #conversation_prompt = st.text_area('Enter prompt or query ')


    #content_container = st.empty()

    if st.button('Generate email (re-click to re-generate)'):
        #user_info = st.text_input('In one sentence, what is your interst or relation to the project. Anything important you want to include in the email?')
        # project_input = st.text_input('Enter the project name')
        # user_name = st.text_input('Enter your name')
        if (project_input == '' or project_input is None) and (user_name == '' or user_name is None):
            st.write('Please enter your info on the sidebar to generate an email')
        
        output_email = email_button_click(project_input, user_name, user_info_input)
        st.write(output_email)

        # llm_answer = generate_llm_answer(f'''Generate email supporting a pipeline project called {project_input}. Start the email
        #                                 with Whom it May Concern. Just write the email in a professional manner, nothing else.''')
        #st.write(llm_answer)

    # if st.button('Re-generate'):
    #     llm_answer = generate_llm_answer(f'''Generate email supporting a pipeline project called {project_input}. Start the email
    #                                 with Whom it May Concern. Just write the email in a professional manner, nothing else.''')
    #     st.write(llm_answer)
            
    #st.markdown("<h2 style='text-align:center; color: grey;'> Start Conversation </h2>", unsafe_allow_html=True)
    ingested_data = ingest_data()
    conversation_prompt = st.text_area('Enter your prompt or query')
    
     # initialize session state for generated reponses and past messages 
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['What can I help you with']
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey there'] 

    # search the database for a response based on user input and update session state 
    if conversation_prompt:
        output = process_answer({'query': conversation_prompt})
        st.session_state['past'].append(conversation_prompt)

        #conversation_prompt = st.text_area('Enter prompt or query ')
        
        if conversation_prompt == '' or conversation_prompt is None: st.write('How can I assist you?')
        #generated_conversation = generate_llm_answer(conversation_prompt)
        #st.write(generated_conversation)
        output = process_answer(conversation_prompt)

        response = output
        #print('\n\n ** source', source)
        st.session_state['generated'].append(response)

    # display conversation history using streamlit messages 
    if st.session_state['generated']:
        display_conversation(st.session_state)

        # if st.button('Message AI'):
        #     generated_conversation = generate_llm_answer(conversation_prompt)
            #content_container.write(generated_conversation)
        
        
        # llm_answer_email = None
        # project_input = st.text_input('Enter the project name')
        # query = st.text_area('Tell us about your interest in the project')
        # #user_input = st.sidebar.text_input("Enter your name:", "")
        # if st.button("generate email"):
        #     st.info('your email')
        #     #llm_pipeline, base_model, generation_pipeline, tokenizer = model_pipeline(FALCON7BINSTRUCT)
        #     llm_answer = generate_llm_answer(f'Generate an email supporting a pipeline project called: {project_input}')
        #     st.write(llm_answer)
        
        #submit_button = st.sidebar.button("Generate email")
        # if submit_button:
        #     processed_output = st.write('did this work?')
        #     st.markdown(f'### Output: \n\n {processed_output}')
        #     st.button
        
        # with st.form("my_form"):
        #     st.write("Inside the form")
        # submitted = st.form_submit_button("Submit")
        # if submitted:
        #     st.write("you clicked the submit button")
        
        # create a form for the project 
        # with st.form(key='email_form'):
        #     project_input = st.text_input(label='Enter the name of the project')
        #     info = st.text_area('Enter info about or your relationship to the project')
        #     #user_name_input = st.text_input(label='Enter your name')
        #     #user_info_input = st.text_input(label='Enter info about you or bio and how it relates to the project as one or two sentence')
        #     form_submit_button = st.form_submit_button('Generate_email')
        # if form_submit_button:
        #     #st.text_area('')
        #     st.info('Your email: ')
        #     st.write(f'Generating email: {project_input}')
            #llm_answer_email = email_button_click(project_input, user_name_input, user_info_input)
            #st.write(llm_answer_email)
            #st.button('Help me send a letter to the editor to a journal', on_click=letter_button_click)
            #st.button('Start a conversation with the Energy AI', on_click=conversation_button_click)
        # if llm_answer_email:
        #     st.write('Here is an email: ')
        #     st.write(llm_answer_email)

    # create forms and chains for the button cl

    # query = st.text_area("Enter your query or prompt ")
    # if st.button('Search'):
    #     st.info('Your answer: ')
    #     #llm_pipeline, base_model, generation_pipeline, tokenizer = model_pipeline(FALCON7BINSTRUCT)
    #     llm_answer = generate_llm_answer(query)
    #     st.write(llm_answer)
        

    # now create a conversation chain 
    #conversation_chain = ConversationChain(llm=llm_pipeline)


if __name__ == '__main__':
    main()





