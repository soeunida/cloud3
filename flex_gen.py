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

        cnt = 0
        while cnt < max_new_tokens:
            outputs = self.model(input_ids_list, attention_mask_list)
            print(f'{cnt+1} 번째 토큰 생성중')
            next_token_logits = outputs['logit_tensor'].clone()[:, :, -1, :].float()
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids_list = torch.cat((input_ids_list, next_tokens), dim=-1)
            new_attention_mask = torch.ones(o_b, i_b, 1).to(next_tokens.device)
            attention_mask_list = torch.cat((attention_mask_list, new_attention_mask), dim=-1)
            
            del outputs
            cnt += 1
  
        return input_ids_list

    
        