from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Dict, Any
from fastapi import Form

from app.services.mistral_service import MistralService
from app.services.embedding_service import EmbeddingService
from app.utils.image_utils import validate_image, preprocess_image

router = APIRouter()

@router.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    mistral_service: MistralService = Depends(lambda: MistralService()),
    embedding_service: EmbeddingService = Depends(lambda: EmbeddingService())
) -> Dict[str, Any]:
    """
    API endpoint to process an image using Mistral OCR, extract structured data,
    and generate embeddings
    
    Args:
        file: Uploaded image file
        mistral_service: MistralService instance for OCR processing
        embedding_service: EmbeddingService instance for generating embeddings
        
    Returns:
        Dictionary containing OCR results and embeddings
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Read file content
    file_content = await file.read()
    
    # Validate image
    is_valid, error_message = validate_image(file_content)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Preprocess image
    processed_image = preprocess_image(file_content)
    
    try:
        # Process image with Mistral OCR
        ocr_result = await mistral_service.process_image(
            image_content=processed_image,
            image_name=file.filename
        )
        
        # Generate embeddings from OCR result
        result = embedding_service.process_ocr_result(ocr_result)
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        ) 