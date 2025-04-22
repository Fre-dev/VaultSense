import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import os
import sys
import asyncio

from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk


backend_path = str(Path(__file__).parent.parent.parent)
if backend_path not in sys.path:
    sys.path.append(backend_path)

from app.core.config import settings

class MistralBatchService:
    """Service for batch processing images and saving structured JSON outputs"""
    
    def __init__(self):
        """Initialize Mistral client with API key from settings"""
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.ocr_model = settings.OCR_MODEL
        self.extraction_model = settings.EXTRACTION_MODEL
        self.output_dir = Path("structured_jsons")
        self.output_dir.mkdir(exist_ok=True)
    
    async def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image and return structured data
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Structured data extracted from the image
        """
        # Read image content
        with open(image_path, 'rb') as f:
            image_content = f.read()
            
        # Encode image as base64 for API
        encoded = base64.b64encode(image_content).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        
        # Process image with OCR
        image_response = self.client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model=self.ocr_model
        )
        
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
        
        try:
            response_dict = json.loads(chat_response.choices[0].message.content)
            return response_dict
        except json.JSONDecodeError:
            return {"raw_content": chat_response.choices[0].message.content}
    
    async def process_folder(self, input_folder: Union[str, Path]) -> None:
        """
        Process all images in a folder concurrently and save structured JSON outputs
        
        Args:
            input_folder: Path to folder containing images
        """
        input_path = Path(input_folder)
        
        # Collect all image paths
        image_paths = [
            path for path in input_path.glob("*")
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process all images concurrently
        tasks = [self.process_image(path) for path in image_paths]
        results = await asyncio.gather(*tasks)
        
        # Save all results
        for image_path, result in zip(image_paths, results):
            output_path = self.output_dir / f"{image_path.stem}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved structured data to {output_path}")


if __name__ == "__main__":
    async def main():
        # Initialize the service
        service = MistralBatchService()
        
        # Specify your input folder
        input_folder = Path("/Users/vijayrajgohil/Documents/VaultSense/process_folder/")
        
        # Process all images
        await service.process_folder(input_folder)
    
    asyncio.run(main())