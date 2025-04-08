from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging
import math

class OpenAIProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        api_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1
    ):
        self.api_key = api_key

        # Ollama endpoint from .env (e.g. http://localhost:11434/v1/)
        self.ollama_base_url = api_url if api_url and len(api_url) else None

        # Official OpenAI endpoint (hardcoded or could come from another .env variable)
        self.openai_official_url = "https://api.openai.com/v1/"

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # We will initialize the client as None; we set base_url in each method via if–else.
        self.client = None

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None
    ):
        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None

        # Decide which base URL to use (official if model starts with "gpt", else local)
        if self.generation_model_id.startswith("gpt"):
            base_url = self.openai_official_url
        else:
            base_url = self.ollama_base_url

        # Create the client with appropriate base_url
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        chat_history.append(self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=chat_history,
                max_tokens=max_output_tokens,
                temperature=temperature
            )
        except Exception as e:
            self.logger.error(f"Error while calling generate_text: {e}")
            return None

        if not response or not response.choices or not response.choices[0].message:
            self.logger.error("Error while generating text (no valid response).")
            return None

        return response.choices[0].message.content

    def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None

        # Decide which base URL to use (example: if it starts with "gpt", use official)
        # Adjust as needed, e.g., if "text-embedding-ada" or "ada" in self.embedding_model_id => official, else local
        if self.embedding_model_id.startswith("gpt") or "ada" in self.embedding_model_id.lower():
            base_url = self.openai_official_url
        else:
            base_url = self.ollama_base_url

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=text
            )
        except Exception as e:
            self.logger.error(f"Error while calling embed_text: {e}")
            return None

        if not response or not response.data or not response.data[0].embedding:
            self.logger.error("Error while embedding text (no valid response).")
            return None

        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }
