from pydantic import Field
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "VaultSense OCR API"
    
    # Mistral AI settings
    MISTRAL_API_KEY: str = Field(default=os.getenv("MISTRAL_API_KEY", ""))
    
    # OCR model settings
    OCR_MODEL: str = Field(default=os.getenv("OCR_MODEL", "mistral-ocr-latest"))
    EXTRACTION_MODEL: str = Field(default=os.getenv("EXTRACTION_MODEL", "pixtral-12b-latest"))

settings = Settings() 