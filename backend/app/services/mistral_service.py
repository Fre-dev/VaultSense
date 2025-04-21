import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk

from app.core.config import settings

class MistralService:
    """Service for interacting with Mistral AI API"""
    
    def __init__(self):
        """Initialize Mistral client with API key from settings"""
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.ocr_model = settings.OCR_MODEL
        self.extraction_model = settings.EXTRACTION_MODEL
    
    async def process_image(self, image_content: bytes, image_name: str) -> Dict[str, Any]:
        """
        Process an image using Mistral OCR and extract structured data
        
        Args:
            image_content: Binary content of the image
            image_name: Name of the image file
            
        Returns:
            Structured data extracted from the image
        """
        # Encode image as base64 for API
        encoded = base64.b64encode(image_content).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        
        # Process image with OCR
        image_response = self.client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model=self.ocr_model
        )
        
        # Get OCR results for processing
        if not image_response.pages:
            return {"error": "No text detected in image"}
            
        image_ocr_markdown = image_response.pages[0].markdown
        
        # Get structured response from model
        chat_response = self.client.chat.complete(
            model=self.extraction_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        ImageURLChunk(image_url=base64_data_url),
                        TextChunk(
                            text=(
                                f"This is image's OCR in markdown:\n\n{image_ocr_markdown}\n.\n"
                                "Convert this into a sensible structured json response. "
                                "The output should be strictly be json with no extra commentary"
                            )
                        ),
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        # Parse and return JSON response
        try:
            response_dict = json.loads(chat_response.choices[0].message.content)
            return response_dict
        except json.JSONDecodeError:
            # If the response is not valid JSON, return the raw content
            return {"raw_content": chat_response.choices[0].message.content} 