import requests
import time
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.float32):
        return float(obj)  # Convert numpy.float32 to native Python float
    return obj

# Define the Hugging Face API Wrapper
class HuggingFaceAPI:
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token

    def query(self, inputs: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_token}"}
         # Convert the inputs to handle numpy types
        inputs = convert_numpy_types(inputs)
        payload = {"inputs": inputs}
        while True:
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()[0]["generated_text"]
            elif response.status_code == 503:
                print(f"Model loading... Estimated time: {response.json().get('estimated_time', 'unknown')} seconds.")
                time.sleep(10)  # Wait and retry
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
        
        
        
# Initialize the Hugging Face API
api_url =  os.getenv("Huggingface_API_URL") 
api_token = os.getenv("API_TOKEN") 
huggingface_api = HuggingFaceAPI(api_url, api_token)





