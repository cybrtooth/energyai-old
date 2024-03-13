from langchain import PromptTemplate, LLMChain



prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
rag_prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


email_generation_template = """You are a helpful assistant who generates emails supporting oil and gas, pipeline and coal projects.
A user will pass in information about them, first their name, then their bio information and you should generate a thoughtful, unique and moderate answer email using the 
user's unique information and perspective for this specific project. Start the email with 'To Whom it may concern' or 'Reviewers of the {project} Project' and have 
the user's name: {user_name} signed at the end of the email. Only return a sample email supporting the project on behalf of the user advocating for its support, nothing more. 
USER name: {user_name}
USER Info: {user_info}
ANSWER="""

email_system_prompt = PromptTemplate(template=email_generation_template, input_variables=['project', 'user_name', 'user_info'])

social_media_template = """ You are a helpful assistant that writes social media posts supporting oil and gas, pipeline and coal project companies. Context often
involves educating the public about the benefits of natural resources and their products. The purpose of many post are to also raise awareness of the work and 
contribution oil and gas and coal companies have in investing to protect the environment. Only write a post based on the user query. Nothing else.
User Query: {user_query} 
Post: 
"""
social_media_post_prompt = PromptTemplate(template=social_media_template, input_variables=['user_query'])


