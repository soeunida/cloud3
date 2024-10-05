import torch
from transformers import AutoTokenizer
import time
import json
import gc
from flex_gen import FlexGeneration
torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"


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
        self.prompt_lens = [0] * 50
        self.generate_cls = FlexGeneration(self.model)
        self.o_batch_size = 0
        self.i_batch_size = 0
        self.max_message_length = 0
        
    
    def outer_batchify(self, batches, o_batch_size):
        outer = []
        for i in range(0, len(batches), o_batch_size):
            outer.append(batches[i:i+o_batch_size])
    
        return outer
    
    def batchify(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    def apply_chat_template_with_padding(self, inner, tokenizer, max_len):
        chat_message = self.tokenizer.apply_chat_template(
            inner, 
            tokenize=False  
        )
        
        tokenized = tokenizer(
            chat_message,
            padding='max_length',  
            max_length=max_len, 
            truncation=True,      
            return_tensors="pt",    
            return_attention_mask=True  
        )
    
        return tokenized
    
    def load_data(self, file_path, o_batch_size, i_batch_size):
        model_inputs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                model_inputs.append(json_obj)
        
        sorted_model_inputs = sorted(model_inputs, key=lambda inputs : len(inputs['message'][0]['content']))
        messages = [inputs['message'] for inputs in sorted_model_inputs]
        self.labels = [inputs['answer'] for inputs in sorted_model_inputs]
        self.max_message_length = len(messages[-1][0]['content'])

        batches = self.batchify(messages, i_batch_size)
        outer_batches = self.outer_batchify(batches, o_batch_size)
        idx = 0
        for i, outer_batch in enumerate(outer_batches):
            inner_batch = []
            inner_batch_m = []
            idx += o_batch_size* i_batch_size -1
            if idx >= len(messages):
                idx = len(messages) - 1
            max_len = len(messages[idx][0]['content'])
            self.prompt_lens[i] = max_len

            for inner in outer_batch:
                tokenized = self.apply_chat_template_with_padding(inner, self.tokenizer, max_len)
                inner_batch.append(tokenized['input_ids'])
                inner_batch_m.append(tokenized['attention_mask'])

            try:
                inner_batch_tensor = torch.stack(inner_batch)
                inner_batch_tensor_m = torch.stack(inner_batch_m)
                self.input_ids.append(inner_batch_tensor)
                self.attention_mask.append(inner_batch_tensor_m)
            except:
                inner_batch_tensor = torch.stack(inner_batch[:-1])
                inner_batch_tensor_m = torch.stack(inner_batch_m[:-1])
                self.input_ids.append(inner_batch_tensor)
                self.attention_mask.append(inner_batch_tensor_m)
                self.input_ids.append(inner_batch[-1])
                self.attention_mask.append(inner_batch_m[-1])
            
    
        self.o_batch_size = o_batch_size
        self.i_batch_size = i_batch_size
        
        
    def forward(self, max_new_tokens):
        times = 0
        cnt = 0
        self.model.eval()
 
        result = torch.empty(0, self.max_message_length, device=self.config.device)
        for batch in zip(self.input_ids, self.attention_mask):
            st = time.time()
            inputs = batch[0].to(self.device)
            masks = batch[1].to(self.device)

            outs = self.generate_cls.generate(input_ids_list=inputs, attention_mask_list=masks, max_new_tokens=max_new_tokens)

            if cnt == 0:
                result = outs.reshape(-1, outs.shape[-1])
            else:
                result = torch.cat([result, outs.reshape(-1, outs.shape[-1])], dim=0)

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
            answer = self.find_pattern(outputs)
            decoded_answer = self.tokenizer.decode(outputs[self.prompt_lens[i//(self.o_batch_size*self.i_batch_size)]:])
            
            if self.labels[i] in decoded_answer:
                correct += 1
            
            tmp_dict['generated_text'] = [{'content':prefill, 'role' : 'user'},{'role':'assistant', 'content':decoded_answer}]
            result.append([tmp_dict])
            i += 1
            
        total = i
        print('맞은 개수', correct)
        print('총 개수 ',total)

        print('accuracy : ', float(correct/total))
        return result


