import requests
import logging
from ..LLMInterface import LLMInterface
from ..LLMEnums import DeepSeekEnums  

class DeepSeekProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1",
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        
        if not self.generation_model_id:
            self.logger.error("Generation model for DeepSeek was not set")
            return None
        
        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        chat_history.append(self.construct_prompt(prompt=prompt, role=DeepSeekEnums.USER.value))

        payload = {
            "model": self.generation_model_id,
            "messages": chat_history,
            "max_tokens": max_output_tokens,
            "temperature": temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(f"{self.api_url}/chat/completions", json=payload, headers=headers)

        if response.status_code != 200:
            self.logger.error(f"Error while generating text with DeepSeek: {response.text}")
            return None

        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", None)

    def embed_text(self, text: str, document_type: str = None):
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for DeepSeek was not set")
            return None

        payload = {
            "model": self.embedding_model_id,
            "input": text
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(f"{self.api_url}/embeddings", json=payload, headers=headers)

        if response.status_code != 200:
            self.logger.error(f"Error while embedding text with DeepSeek: {response.text}")
            return None

        result = response.json()
        return result.get("data", [{}])[0].get("embedding", None)

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }
