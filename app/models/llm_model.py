import os
import json
import openai
from typing import Dict, Any
from app.utils.logger import get_logger


logger = get_logger()


class LLMService:
    """Service for interacting with Large Language Models."""
    
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.debug(f"OpenAI API was set successfully")
        else:
            self.openai_client = None
    
    def analyze_with_openai(self, text: str) -> Dict[str, Any]:
        """Analyze text using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        system_prompt = """
        You are a sentiment analysis expert. Analyze the provided text and output a JSON response with the following fields:
        - sentiment: The overall sentiment (must be one of: "positive", "negative", or "neutral")
        - confidence: A float between 0 and 1 indicating your confidence in the sentiment assessment

        Output ONLY valid JSON. No other text.
        """
        user_prompt = f"Text to analyze: {text}"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            logger.debug(f"Response from OpenAI API for {text=} gated {result=}")
            if "sentiment" not in result or "confidence" not in result:
                raise ValueError("LLM response missing required fields")
            return result

        except Exception as e:
            raise RuntimeError(f"Error analyzing text with OpenAI: {str(e)}")

llm_model = LLMService()