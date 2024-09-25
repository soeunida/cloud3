import torch
from transformers import AutoTokenizer
from customed_pipeline import CustomedPipeline
from hf_ref import NewPhi3Config
from model2 import CustomedPhi3ForCausalLM
#import requests

def download_model():
    base_path = '/nas/user/hayoung'
    
    for i in range(6):
        file_path = base_path + f'model-0000{i+1}-of-00006.safetensors'
        with open(file_path, 'wb') as device_file:
            path = f'https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/resolve/main/model-0000{i+1}-of-00006.safetensors'
            response = requests.get(path, stream=True)
            print(f'{i+1}번째 파일 status ', response.status_code)
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  
                    if chunk: 
                        device_file.write(chunk)
            

#download_model()

model_id = "microsoft/Phi-3-medium-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = NewPhi3Config(base_path='/nas/user/hayoung')
model = CustomedPhi3ForCausalLM(config)

pipe = CustomedPipeline(model, config)
pipe.load_data(file_path= "/home/hayoung/ss/test.jsonl", batch_size=30)
outputs = pipe.forward(max_new_tokens=15)
result = pipe.postprocess(outputs)
print(result)

 
