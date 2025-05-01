import os
from typing import Iterator
from .openai import OpenAI
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.messages import BaseMessageChunk
from src.utils.logger import setup_logger
logger = setup_logger(__name__)


def get_superior_llm_model():
    return OpenAI(model_id=os.getenv("SUPERIOR_OPENAI_MODEL_NAME")).llm

def get_llm_model():
    return OpenAI().llm


def call_llm(prompt: str, model_id: str = None) -> Iterator[BaseMessageChunk]:
    kwargs = {"model_id": model_id} if model_id else {}

    openai = OpenAI(**kwargs)
    llm_response: LLMResult = openai.call_llm(prompt)
    return llm_response


def call_llm_sync(prompt: str, model_id: str = None) -> LLMResult:
    kwargs = {"model_id": model_id} if model_id else {}
    
    # Initialize the LLM response variable
    llm_response = None
    
    # Create OpenAI instance
    openai = OpenAI(**kwargs)
    
    # Call the synchronous LLM method
    llm_response = openai.call_llm_sync(prompt)
    
    logger.info(f"Sync LLM call completed with response type: {type(llm_response)}")
    
    return llm_response


def call_superior_llm(prompt: str, model_id: str = None) -> Iterator[BaseMessageChunk]:
    kwargs = {"model_id": model_id} if model_id else {}

    openai = OpenAI(**kwargs, model_id=os.getenv("SUPERIOR_OPENAI_MODEL_NAME"))
    llm_response: LLMResult = openai.call_llm(prompt)
    return llm_response


def call_superior_llm_sync(prompt: str, model_id: str = None) -> LLMResult:
    kwargs = {"model_id": model_id} if model_id else {}
    
    # Initialize the LLM response variable
    llm_response = None
    
    # Create OpenAI instance
    openai = OpenAI(**kwargs, model_id=os.getenv("SUPERIOR_OPENAI_MODEL_NAME"))
    
    # Call the synchronous LLM method
    llm_response = openai.call_llm_sync(prompt)
    
    logger.info(f"Sync LLM call completed with response type: {type(llm_response)}")
    
    return llm_response
