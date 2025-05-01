from typing import AsyncGenerator, Literal, Dict, Any
import os
from datetime import datetime
import uuid
from langgraph.graph import StateGraph, END
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

# Import tools for Document operations
from src.tools.document import (
    create_document,
    update_document,
    assign_document,
    fetch_documents,
    analyze_documents,
    search_documents,
    comment_on_document,
    delete_document,
    store_document_in_milvus,
    search_document_in_milvus
)

from src.states.document import DocumentState
from src.llm.runner import get_llm_model, call_llm, call_llm_sync

class JiraAgent:
    """
    Agent that interacts with JIRA for ticket management and data retrieval.
    This agent can create, update, fetch and search JIRA tickets, as well as
    store and retrieve JIRA data from Milvus vector database.
    """

    def __init__(self):
        self.model = get_llm_model()
        self.tools = [
            create_document,
            update_document,
            assign_document,
            fetch_documents,
            analyze_documents,
            search_documents,
            comment_on_document,
            delete_document,
            store_document_in_milvus,
            search_document_in_milvus
        ]

        self.bound_model = self.model.bind_tools(self.tools)
        self.app = self._create_workflow()

    def _should_store_in_milvus(self, state: DocumentState) -> Literal["store_document_data", "generate_response"]:
        """Determine if we should store the result in Milvus"""
        operation = state.get("operation", "")
        
        if state.get("found_in_milvus") == True:
            logger.info("Document already found in Milvus, skipping storage")
            return "generate_response"
            
        if "error" in state:
            logger.info("Error in state, skipping Milvus storage")
            return "generate_response"

        if "document_data" in state:
            document_data = state.get("document_data")
            if isinstance(document_data, dict) and ("message" in document_data or "error" in document_data):
                logger.info("Document data contains message/error, skipping Milvus storage")
                return "generate_response"
            elif isinstance(document_data, list) and len(document_data) == 0:
                logger.info("Empty document list, skipping Milvus storage")
                return "generate_response"
        
        if operation in ["create", "update", "fetch"] and "document_data" in state:
            logger.info(f"Storing document data in Milvus for operation: {operation}")
            return "store_document_data"
            
        return "generate_response"

    def _determine_next_action(self, state: DocumentState) -> Literal["create_document", "update_document", "fetch_documents", "search_documents", "comment_on_document", "search_milvus", "analyze_documents"]:
        """Determine the next action based on the operation requested"""
        operation = state.get("operation", "")
        
        if operation == "create":
            return "create_document"
        elif operation == "update":
            return "update_document"
        elif operation == "assign":
            return "assign_document"
        elif operation == "fetch":
            return "fetch_documents"
        elif operation == "search":
            return "search_documents"
        elif operation == "comment":
            return "comment_on_document"
        elif operation == "delete":
            return "delete_document"
        elif operation == "analyze":
            return "analyze_documents"
        else:
            return "search_milvus"

    def _create_workflow(self):
        workflow = StateGraph(DocumentState)
        
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("create_document", create_document)
        workflow.add_node("update_document", update_document)
        workflow.add_node("assign_document", assign_document)
        workflow.add_node("fetch_documents", fetch_documents)
        workflow.add_node("analyze_documents", analyze_documents)
        workflow.add_node("search_documents", search_documents)
        workflow.add_node("comment_on_document", comment_on_document)
        workflow.add_node("search_milvus", search_document_in_milvus)
        workflow.add_node("store_document_data", store_document_in_milvus)
        workflow.add_node("delete_document", delete_document)
        workflow.add_node("generate_response", self._generate_response)

        operation_map = {
            "create_document": "create_document",
            "update_document": "update_document",
            "assign_document": "assign_document",
            "fetch_documents": "fetch_documents",
            "analyze_documents": "analyze_documents",
            "search_documents": "search_documents",
            "comment_on_document": "comment_on_document",
            "search_milvus": "search_milvus",
            "delete_document": "delete_document"
        }
        
        workflow.add_conditional_edges(
            "analyze_request",
            self._determine_next_action,
            operation_map
        )
        
        storage_map = {
            "store_document_data": "store_document_data",
            "generate_response": "generate_response"
        }
        
        for node in ["create_document", "update_document", "assign_document", "fetch_documents", "search_documents", "comment_on_document", "search_milvus", "delete_document", "analyze_documents"]:
            workflow.add_conditional_edges(
                node,
                self._should_store_in_milvus,
                storage_map
            )
        
        workflow.add_edge("store_document_data", "generate_response")
        
        workflow.set_entry_point("analyze_request")
        workflow.set_finish_point("generate_response")
        
        return workflow.compile()

    def _analyze_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the user request to determine the operation"""
        task = state.get("task", "")
        
        # Use LLM to determine what operation the user wants to perform
        messages = [
            {"role": "system", "content": "You are an AI assistant that analyzes user requests related to documents."},
            {"role": "user", "content": f"Based on this request: '{task}', determine which operation the user wants to perform: create, update, assign, delete, analyze, fetch, search, or comment on a document. Return ONLY the operation name."}
        ]
        
        try:
            logger.info("Calling LLM to analyze document request")
            response = call_llm_sync(messages)
            
            # Extract content from response
            operation = ""
            if hasattr(response, "content"):
                operation = response.content
            else:
                operation = str(response)
                
            logger.info(f"LLM returned operation: {operation}")
            
            # Normalize the operation
            if "create" in operation.lower():
                operation = "create"
                
            elif "update" in operation.lower():
                operation = "update"
            elif "assign" in operation.lower():
                operation = "assign"
            elif "fetch" in operation.lower() or "get" in operation.lower():
                operation = "fetch"
            elif "search" in operation.lower():
                operation = "search"
            elif "comment" in operation.lower():
                operation = "comment"
            elif "delete" in operation.lower():
                operation = "delete"
            elif "analyze" in operation.lower():
                operation = "analyze"
            else:
                operation = "search"  # Default to search
                
            logger.info(f"Normalized operation: {operation} for task: {task}")
            
        except Exception as e:
            logger.error(f"Error analyzing request: {str(e)}")
            # Default to search as fallback
            operation = "search"
        
        return {"task": task, "operation": operation, **state}

    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Generating final response: {state}")

        """Generate a human-readable response based on the state"""
        ticket_data = state.get("ticket_data", {})
        operation = state.get("operation", "")
        task = state.get("task", "")
        answer = state.get("answer", "")
        
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps with document management."},
            {"role": "user", "content": f"Generate a concise, helpful response about the {operation} operation for the document. The original request was: '{task}'. The document data is: {document_data}"}
        ]
        
        try:
            logger.info("Calling LLM to generate document response")
            response = call_llm_sync(messages)
            
            # Extract content from response
            response_content = ""
            if hasattr(response, "content"):
                response_content = response.content
            else:
                response_content = str(response)
                
            logger.info("Successfully generated document response")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response_content = f"I encountered an error while processing your request related to {operation} operation. Please try again."
        
        # Generate a unique response ID
        response_id = str(uuid.uuid4())
        
        # Create answer structure
        answer = {
            "response": response_content,
            "summary": response_content,  # Ensure summary is also populated for chat.py
            "response_id": response_id,
            "answer": answer, # If answer is not empty, use it
            "timestamp": datetime.now().isoformat()
        }
        
        # Set the answer in the state
        state["answer"] = answer
        
        # Mark this as task_complete to signal to the supervisor that no further processing is needed
        state["task_complete"] = True
        
        logger.info(f"Generated document response with ID: {response_id}")
        
        return state

    def run(self, input: str):
        logger.info(f"Document Agent received input: {input}")
        if isinstance(input, str):
            input = {"task": input}
        
        result = self.app.invoke(input)
        
        # Ensure we're correctly setting the answer in the output
        # This is critical for the supervisor to recognize completion
        if "answer" in result:
            logger.info("Document Agent completed task successfully")
        else:
            logger.warning("Document Agent did not set answer in result")
            
        return result
