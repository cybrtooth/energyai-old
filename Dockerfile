# # Use an official Python image as the base image
# FROM python:3.10

# WORKDIR /code
# COPY requirements.txt ./
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# COPY . .

# # EXPOSE 3100

# # #CMD ["gunicorn", "main:app"]
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# # #CMD [ "handler.handler" ]
# # RUN echo 'Image Built'

# # # Set the working directory in the container
# #WORKDIR /app
# #COPY ./requirements.txt /code/requirements.txt
# #COPY . /app
# # Install required dependencies
# #RUN pip install --no-cache-dir --updgrade -r /code/requirements.txt
# #RUN echo 'Image Built'

#  # Copy the Python script into the container
# COPY main.py main.py 
# COPY templates templates 
# COPY static static
# COPY stores stores
# COPY mistral-7b-instruct-v0.1.Q4_K_S.gguf mistral-7b-instruct-v0.1.Q4_K_S.gguf

# # Expose the port where FastAPI will run
# EXPOSE 8000

# # Command to run your FastAPI script with uvicorn
# # run this in terminal by doing: docker build -t image_name
# # then run by: docker run -d -p 8000:8000 image_name
# # to bind the terminal to the docker terminal so that you see the inputs and outputs: docker exec -it containerid bash
# # you can find the container id by running the command: docker ps to see running containers 
# # will then see that you are in the docker container and 
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# #CMD ['app.handler']


# Use the official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your FastAPI app code into the container
COPY . /app

# Install dependencies from requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Download the Hugging Face model (replace with your specific model)
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='mistralai/Mistral-7B-Instruct-v0.1')"

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Command to start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
