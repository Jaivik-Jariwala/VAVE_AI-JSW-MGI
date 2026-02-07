import os
import logging
import time
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
from PIL import Image

# Load env vars
load_dotenv()

logger = logging.getLogger(__name__)

# Configure API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY is missing!")
else:
    genai.configure(api_key=api_key)

# Allowed Models in Priority Order
PRIORITY_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash", 
    "gemini-3-flash"
]

class LLMClient:
    def __init__(self):
        self.working_model = None

    def _get_model(self, model_name=None):
        """Returns a GenerativeModel instance."""
        name = model_name or self.working_model or PRIORITY_MODELS[0]
        return genai.GenerativeModel(name)

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception)) # Broad retry for API flakiness + 429
    )
    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generates text with retry logic. 
        Iterates through priority models if one fails permanently (after retries).
        """
        last_error = None
        
        for model_name in PRIORITY_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                
                final_prompt = prompt
                if system_prompt:
                    final_prompt = f"System Instruction: {system_prompt}\n\nUser Query: {prompt}"
                
                logger.info(f"Generating with {model_name}...")
                response = model.generate_content(final_prompt)
                
                if response.text:
                    self.working_model = model_name # Cache success
                    return response.text
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                last_error = e
                continue

        logger.error(f"All models failed. Last error: {last_error}")
        raise last_error # Trigger Tenacity retry

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception))
    )
    def generate_vision(self, prompt: str, image_input) -> str:
        """
        Generates content from image + text.
        image_input can be: path (str/Path), PIL.Image, or list of them.
        """
        inputs = []
        if isinstance(image_input, list):
            for img in image_input:
                inputs.append(self._prepare_image(img))
        else:
            inputs.append(self._prepare_image(image_input))
            
        inputs.append(prompt)

        last_error = None
        for model_name in PRIORITY_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Vision Generation with {model_name}...")
                response = model.generate_content(inputs)
                if response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Vision Model {model_name} failed: {e}")
                last_error = e
                continue
        
        raise last_error

    def _prepare_image(self, img_obj):
        """Standardize image input for Gemini."""
        if isinstance(img_obj, (str, os.PathLike)):
            path = str(img_obj)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path)
        return img_obj

# Singleton instance
llm_client = LLMClient()

def generate_text(prompt: str, system_prompt: str = None) -> str:
    return llm_client.generate_text(prompt, system_prompt)

def generate_vision(prompt: str, image_input) -> str:
    return llm_client.generate_vision(prompt, image_input)
