from datetime import datetime
import os
import json
from time import sleep
import uuid
import requests
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from src.agents.supervisor import SupervisorAgent
from src.dto.chat import ChatInput, ChatOutput
from pydantic import BaseModel
from src.utils.logger import setup_logger
logger = setup_logger(__name__)


router = APIRouter()


@router.post("/chat")
async def chat(
    input: ChatInput,
):
    async def chat_stream(input: ChatInput):


        response_start = f'{{"session_id":"{input.session_id}", "thread_id":"{input.thread_id}","response":'
        response_end = f"}}"

        response_complete: str = response_start
        error_msg = f'{{"error": "Error processing response", "partial_response": true}}'

        try:
            yield response_start

            async for event in SupervisorAgent().app.astream_events(
                {
                    "task": {
                        "original": input.question,
                    },
                },
                version="v2",
                config={"configurable": {"question": input.question}},
            ):
                try:
                    kind = event["event"]

                    # stream response from llm in final node which is generate_answer
                    if kind == "on_chat_model_stream" and event["metadata"][
                        "langgraph_node"
                    ].startswith("generate_"):
                        ev = event["data"]

                        if ev["chunk"].content:
                            response_complete += ev["chunk"].content
                            yield ev["chunk"].content

                    # get response_id from state on chain_end for generate_answer
                    if kind == "on_chain_end" and event["name"] == "generate_answer":
                        if "output" in event["data"] and "answer" in event["data"]["output"]:
                            answer_data = event["data"]["output"]["answer"]
                            if "summary" in answer_data:
                                response_complete += answer_data["summary"]
                                yield answer_data["summary"]
                            elif "response" in answer_data:
                                # Fallback to response if summary doesn't exist
                                response_complete += answer_data["response"]
                                yield answer_data["response"]

                    if kind == "on_chain_end" and event["name"] == "save_memories":
                        if "output" in event["data"] and "answer" in event["data"]["output"] and "response_id" in event["data"]["output"]["answer"]:
                            chain_end_str = f',"response_id":"{event["data"]["output"]["answer"]["response_id"]}"'
                            response_complete += chain_end_str
                            yield chain_end_str

                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    logger.error(f"Error processing event: {str(e)}")
                    logger.error(f"Error traceback: {error_trace}")
                    yield error_msg + "}"
                    return

            yield response_end

        except Exception as e:            
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Fatal error in chat stream: {str(e)}")
            logger.error(f"Error traceback: {error_trace}")
            yield error_msg + "}"

    return StreamingResponse(chat_stream(input), media_type="text/event-stream")