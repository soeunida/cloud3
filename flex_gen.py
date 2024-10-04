import torch

class FlaxGeneration():
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
        
        cnt = 0
        while cnt < max_new_tokens:
            outputs = self.model(input_ids_list,attention_mask_list)
            for i, output in enumerate(outputs['logit_list']):
                next_token_logits = output.clone()[:, -1, :].float()/0.7
            
                next_tokens = torch.argmax(next_token_logits, dim=-1)
         
                input_ids = torch.cat([input_ids_list[i], next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask_list[i], attention_mask_list[i].new_ones((attention_mask_list[i].shape[0], 1))], dim=-1
                )
                input_ids_list[i] = input_ids
                attention_mask_list[i] = attention_mask

            del outputs
            cnt += 1
            
        return input_ids_list


    
        