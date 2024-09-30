import torch

class CustomGeneration():
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        logits_processor=None,
        stopping_criteria=None,
        synced_gpus=False,
        streamer=None,
        **model_kwargs
        ) :
        
        cnt = 0
        while cnt < max_length:
            outputs = self.model(input_ids,attention_mask)
            
            next_token_logits = outputs.logits.clone()[:, -1, :].float()/0.7
            
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )


            del outputs
            cnt += 1
            
        return input_ids
    
