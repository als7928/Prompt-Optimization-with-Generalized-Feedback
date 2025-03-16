from openai import OpenAI, AuthenticationError
import time
import tiktoken

class OpenAIModel():
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        **kwargs):
        
        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        try:
            self.model = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Init openai client error: \n{e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e
        
        self.model_name = model_name 
        self.temperature = temperature
        
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion
        
        # Initialize tokenizer
        self.embedding_model_name = "text-embedding-3-large"
        self.tiktokenizer = tiktoken.encoding_for_model(model_name)
        self.tiktokenizer_emb = tiktoken.encoding_for_model(self.embedding_model_name)

        self.in_token = 0
        self.out_token = 0
        self.emb_token = 0

    def calculate_token_count(self, tokenizer, prompt):
        """Calculate and return the token count for input and output."""
        token_count = len(tokenizer.encode(prompt)) 
        return token_count    
    
    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to openai chat API and retrieve the answers.
        """
        responses = []
        for prompt in batch_prompts:
            response = self.gpt_chat_completion(prompt=prompt)
            responses.append(response)
        return responses
                    
    def gpt_chat_completion(self, prompt):
        messages = [{"role": "user", "content": prompt},]
        backoff_time = 1
        while True:
            try:
                response = self.model.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature).choices[0].message.content.strip()
 
                self.in_token += self.calculate_token_count(self.tiktokenizer, prompt)
                self.out_token += self.calculate_token_count(self.tiktokenizer, response)

                return response
            except Exception as e:
                print(e, f' Sleeping {backoff_time} seconds...')
                time.sleep(backoff_time)
                backoff_time *= 1.5

    def get_text_embeddings(self, texts):
        try:  
            self.emb_token +=  sum([self.calculate_token_count(self.tiktokenizer_emb, text) for text in  texts])
            # self.calculate_token_count(self.tiktokenizer_emb, texts)
            response = self.model.embeddings.create(
                model=self.embedding_model_name,
                input=texts
            )
            embeddings = [i.embedding for i in response.data]
            return embeddings
        except AuthenticationError as e:
            print(f"Error fetching embeddings: {e}")
            return None
        
        