from io import BytesIO
from typing import Tuple
from PIL import Image

def validate_image(file_content: bytes) -> Tuple[bool, str]:
    """
    Validate if the file is a supported image type
    
    Args:
        file_content: Binary content of the file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to open the image with PIL
        img = Image.open(BytesIO(file_content))
        img.verify()  # Verify it's an image
        
        # Check supported formats
        if img.format not in ['JPEG', 'PNG', 'TIFF', 'BMP', 'GIF']:
            return False, f"Unsupported image format: {img.format}"
            
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def preprocess_image(file_content: bytes, max_size: int = 4000) -> bytes:
    """
    Preprocess image if needed (resize, format conversion, etc.)
    
    Args:
        file_content: Binary content of the image
        max_size: Maximum dimension (width or height) for the image
        
    Returns:
        Processed image content as bytes
    """
    try:
        img = Image.open(BytesIO(file_content))
        
        # Resize if image is too large
        width, height = img.size
        if width > max_size or height > max_size:
            # Calculate new dimensions while preserving aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
                
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to RGB if needed (removes alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save to buffer
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        return buffer.getvalue()
    except Exception:
        # If preprocessing fails, return original content
        return file_content 