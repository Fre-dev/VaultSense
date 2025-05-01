import os
from typing import Iterator, Dict, Any, List, Union
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.messages import BaseMessageChunk
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

class OpenAI:
    def __init__(self, model_id: str = os.getenv("OPENAI_MODEL_NAME")):
        self.llm = ChatOpenAI(
            model=model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True,
            temperature=0,
        )

    def _format_messages(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Union[HumanMessage, SystemMessage]]:
        """Format messages to be compatible with the OpenAI API"""
        if isinstance(prompt, str):
            # If prompt is a string, create a simple HumanMessage
            return [HumanMessage(content=prompt)]
        elif isinstance(prompt, list):
            # If prompt is a list of message dicts
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    messages.append(SystemMessage(content=msg.get("content", "")))
                elif msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                # Add more roles as needed
            return messages
        else:
            # Fallback for any other format
            logger.warning(f"Unexpected prompt format: {type(prompt)}")
            return [HumanMessage(content=str(prompt))]

    def call_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Iterator[BaseMessageChunk]:
        """Call LLM with streaming enabled"""
        try:
            messages = self._format_messages(prompt)
            llm_response = self.llm.stream(messages)
            return llm_response
        except Exception as e:
            logger.error(f"Error calling LLM with streaming: {str(e)}")
            # Return a simple error message
            class ErrorMessage:
                @property
                def content(self):
                    return f"Error calling LLM: {str(e)}"
            return ErrorMessage()

    def call_llm_sync(self, prompt: Union[str, List[Dict[str, str]]]) -> LLMResult:
        """Call LLM synchronously"""
        try:
            messages = self._format_messages(prompt)
            llm_response = self.llm.invoke(messages)
            return llm_response
        except Exception as e:
            logger.error(f"Error calling LLM synchronously: {str(e)}")
            # Return a simple error message
            class ErrorMessage:
                @property
                def content(self):
                    return f"Error calling LLM: {str(e)}"
            return ErrorMessage()
