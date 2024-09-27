from transformers import GenerationMixin
import torch

class CustomGeneration(GenerationMixin):
    def __init__(self, model, generation_config):
        super().__init__()
        self.model = model
        self.generation_config = generation_config
    
    
        
    def _get_initial_cache_position(self, input_ids, model_kwargs):
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def _update_model_kwargs(
         self,
        model_kwargs,
        is_encoder_decoder= False
    ):
        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + 2, dtype=past_positions.dtype
            ).to(past_positions.device)
        model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs
        
    def _sample(
        self,
        input_ids,
        attention_mask,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=False,
        streamer=None,
        **model_kwargs
    ):
        pad_token_id = generation_config.eos_token_id
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        cnt = 0
        while cnt < self.generation_config.max_length:
            outputs = self.model(input_ids,attention_mask)
            
            next_token_logits = outputs.logits.clone()[:, -1, :].float()/0.7
            
            next_tokens = torch.argmax(next_token_logits, dim=-1)
         
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            cnt += 1

            del outputs
        return input_ids
            
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        generation_config= None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = None,
        assistant_model = None,
        streamer = None,
        negative_prompt_ids = None,
        negative_prompt_attention_mask= None,
        **model_kwargs
    ) :
        
        
        batch_size = input_ids.shape[0]

        device = input_ids.device
        
        input_ids_length = input_ids.shape[-1]
        prepared_stopping_criteria = EosTokenCriteria(generation_config.eos_token_id)
        
        result = self._sample(
                input_ids,
                attention_mask,
                logits_processor=None,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=False,
                streamer=None,
                **model_kwargs
        )
        
        return result
    

class Cutstom_GenerationConfig():
    def __init__(self, max_length, eos_token_id):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        