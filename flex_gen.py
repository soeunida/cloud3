from transformers import GenerationMixin, GenerationConfig
import torch

class FlaxGeneration(GenerationMixin):
    def __init__(self, model, generation_config):
        super().__init__()
        self.model = model
        self.generation_config = generation_config
    
            
    @torch.no_grad()
    def generate(
        self,
        input_ids_list,
        attention_mask_list,
        generation_config,
        logits_processor=None,
        stopping_criteria=None,
        synced_gpus=False,
        streamer=None,
        **model_kwargs
        ) :
        
        cnt = 0
        while cnt < self.generation_config.max_length:
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


    

class Cutstom_GenerationConfig():
    def __init__(self, max_length, eos_token_id):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        