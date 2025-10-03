from pyexpat import model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken

class LlamaModel():
    def __init__(
        self,
        model_name: str,
        temperature: float,
        device: str,
        **kwargs):

        self.model_name = model_name
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        self.temperature = temperature
        self.do_sample = True if temperature != 0 else False

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncate=True, padding=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding_model_name = self.model_name

        self.in_token = 0
        self.out_token = 0
        self.emb_token = 0

        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.chat_completion

    def calculate_token_count(self, tokenizer, prompt): 
        return tokenizer(prompt, padding=True, return_tensors="pt", truncation=True)['input_ids'].size(1)

    def batch_forward_chatcompletion(self, batch_prompts):
        responses = []
        for prompt in batch_prompts:
            response = self.chat_completion(prompt=prompt)
            responses.append(response)
        return responses

    def chat_completion(self, prompt):
        messages = [{"role": "user", "content": prompt},]
        self.in_token += self.calculate_token_count(self.tokenizer, prompt)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            
            
        ).to(self.device)

        response = self.model.generate(**inputs,
                                        do_sample=self.do_sample,
                                        temperature=self.temperature,
                                        max_new_tokens=1024,
                                        repetition_penalty=1.2,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        pad_token_id = self.tokenizer.eos_token_id) 

        self.out_token += response.sequences.size(1)
        generated_sequences = response.sequences[:, inputs['input_ids'].size(1):]
        ret = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        return ret[0].strip()
    
    def get_text_embeddings(self, texts): 
        emb_token = 0
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            emb_token += inputs['input_ids'].size(1)
            with torch.no_grad():
                embedding = self.model(**inputs).last_hidden_state
            embeddings.append(embedding.cpu().tolist()) 
        self.emb_token += emb_token
        
        return embeddings
    
    
