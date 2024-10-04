import torch
from transformers import AutoTokenizer
import time
import json
import gc
from flex_gen import FlaxGeneration
torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cuda",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )


class CustomedPipeline():
    def __init__(
            self,
            model,
            config,
            model_id = "microsoft/Phi-3-medium-4k-instruct",
            device = "cuda"
        ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model =  model
        self.input_ids = []
        self.attention_mask = []
        self.device = device
        self.labels = []
        self.generate_cls =  FlaxGeneration(self.model)
        self.o_batch_size = 0
        self.i_batch_size = 0
        
    
    def outer_batchify(self, batches, o_batch_size):
        outer = []
        for i in range(0, len(batches), o_batch_size):
            outer.append(batches[i:i+o_batch_size])
    
        return outer
    
    def batchify(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    
    def load_data(self, file_path, o_batch_size, i_batch_size):
        model_inputs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                model_inputs.append(json_obj)
                
        messages = [inputs['message'] for inputs in model_inputs]
        self.labels = [inputs['answer'] for inputs in model_inputs]
        batches = self.batchify(messages, i_batch_size)
        outer_batches = self.outer_batchify(batches, o_batch_size)
        
        for outer_batch in outer_batches:
            inner_batch = []
            inner_batch_m = []
            
            tokenized = [self.tokenizer.apply_chat_template(inner, 
                                                tokenize=True, 
                                                padding=True, 
                                                truncation=True,
                                                return_tensors="pt", 
                                                return_dict=True) for inner in outer_batch]
                
            for token in tokenized:
                inner_batch.append(token['input_ids'])
                inner_batch_m.append(token['attention_mask'])
            self.input_ids.append(inner_batch)
            self.attention_mask.append(inner_batch_m)
            
        self.o_batch_size = o_batch_size
        self.i_batch_size = i_batch_size
        # self.input_ids = [token['input_ids'] for token in outer_batch for outer_batch in tokenized_batches]
        # self.attention_mask = [[token['attention_mask'] for token in outer_batch] for outer_batch in tokenized_batches]
  
    def forward(self, max_new_tokens):
        times = 0
        result = []
        cnt = 0
        self.model.eval()
        for batch in zip(self.input_ids, self.attention_mask):
            st = time.time()
            inputs = [batch.to(self.device) for batch in batch[0]]
            masks = [batch.to(self.device) for batch in batch[1]]
            
            outs = self.generate_cls.generate(input_ids_list=inputs, attention_mask_list=masks, max_new_tokens=max_new_tokens)

            for out in outs:
                result.append(out)
                tmp = self.tokenizer.batch_decode(out)
            end = time.time()
            
            print('batch load and inference time ', (end - st))
            print('inference time per item ',(end-st)/(self.o_batch_size*self.i_batch_size))
            times += end - st
            cnt += 1
            if cnt % 5 == 0:
                gc.collect()
        print('total inference time ', times)
        
        return {"generated_sequence": result}


    def find_pattern(self, text):
        idx = []
        for i in range(91,len(text)-1):
            if text[i] == 32007 and text[i+1]==32001:
                idx.append(i)
           
        if len(idx) == 0:
            return text[90:]
        elif len(idx) == 1:
            return text[idx[0]+2:]
        else:
            return text[idx[0]+2: idx[1]]

    def postprocess(self,model_outputs,  clean_up_tokenization_spaces=True):

        result = []
        correct = 0
        i = 0
        for outputs in model_outputs['generated_sequence']:   
            for text in outputs:
                answer = self.find_pattern(text)
                decoded_answer = self.tokenizer.decode(answer)
                if self.labels[i] in decoded_answer:
                    correct += 1
                else:
                    decoded_answer = self.tokenizer.decode(text[91:])
                result.append([{'generated':decoded_answer, 'label' : self.labels[i]}])
                i += 1

        total = i
        print('맞은 개수', correct)
        print('총 개수 ',total)

        print('accuracy : ', float(correct/total))
        return result


