import torch

class FlexGeneration():
    def __init__(self, model):
        super().__init__()
        self.model = model
    
            
    @torch.no_grad()
    def generate(
        self,
        input_ids_list,
        attention_mask_list,
        max_new_tokens,
        logits_processor=None,
        stopping_criteria=None,
        synced_gpus=False,
        streamer=None,
        **model_kwargs
        ) :
        
        o_b, i_b, seqlen = input_ids_list.shape

        tmp_ids_list = torch.zeros((o_b, i_b, seqlen+max_new_tokens), device=input_ids_list[0].device, dtype=input_ids_list[0].dtype)
        tmp_am_list = torch.zeros((o_b, i_b, seqlen+max_new_tokens), device=attention_mask_list[0].device, dtype=attention_mask_list[0].dtype)
        
        tmp_ids_list[:, :,  :seqlen] = input_ids_list
        tmp_am_list[:, :, :seqlen] = attention_mask_list

        cnt = 0
        while cnt < max_new_tokens:
            outputs = self.model(input_ids_list, attention_mask_list)
            print(f'{cnt+1} 번째 토큰 생성중')
            for i, output in enumerate(outputs['logit_list']):
                next_token_logits = output.clone()[:, -1, :].float()/0.7
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                tmp_ids_list[i][:, seqlen + cnt] = next_tokens
                tmp_am_list[i][:, seqlen + cnt] = 1

            del outputs
            cnt += 1
  
        return tmp_ids_list


    
        