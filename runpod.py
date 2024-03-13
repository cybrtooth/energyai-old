import os
#set you runpod key as a environment variable
os.environ['RUNPOD_API_KEY'] = "your_runpod_api_key"

import runpod
from IPython.display import display, Markdown

runpod.api_key = os.getenv("RUNPOD_API_KEY", "your_runpod_api_key")

if runpod.api_key == "your_runpod_api_key":
    display(
        Markdown(
            "It appears that you don't have a RunPod API key. You can obtain one at [runpod.io](https://runpod.io?ref=s7508tca)"
        )
    )
    raise AssertionError("Missing RunPod API key")
    
#show all possible available GPUs
runpod.get_gpus()

# Create your pod, you can set the data_center_id (optional)
# Decide which model you want to use, here we use falcon-40b

pod = runpod.create_pod(
    name="Falcon-40B",
    image_name="ghcr.io/huggingface/text-generation-inference:0.8",
    gpu_type_id="NVIDIA A100 80GB PCIe",
    cloud_type="SECURE",
    # data_center_id="US-KS-1",
    docker_args=f"--model-id tiiuae/falcon-40b --num-shard {gpu_count}",
    gpu_count=2,
    volume_in_gb=195,
    container_disk_in_gb=5,
    ports="80/http,29500/http",
    volume_mount_path="/data",
)

#Create inference server using your pod LLM throw langchain
from langchain.llms import HuggingFaceTextGenInference

inference_server_url = f'https://{pod["id"]}-80.proxy.runpod.net'
llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url,
    max_new_tokens=100,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.001,
    repetition_penalty=1.03,
)