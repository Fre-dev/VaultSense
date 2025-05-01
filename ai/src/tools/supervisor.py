import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.llm.runner import get_llm_model, call_llm, call_llm_sync
from src.memory.long import LongTermMemory
from src.utils.logger import setup_logger
logger = setup_logger(__name__)
ltm = LongTermMemory(customer="default")

def retrieve_memories(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant memories from long-term memory.
    
    This tool queries the long-term memory to find relevant information
    based on the user's task.
    """

    logger.info(f"Retrieving memories for task: {state.get('original', '')}")
    original_task = state.get("original", "")
    if original_task:
        try:
            memories = ltm.get_similar_questions(original_task)
            state["memories"] = memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            state["memories"] = []
    
    return state

def create_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a plan for handling the task.
    """
    # Check if we already have enough information to give an answer directly
    if "ticket_data" in state or "answer" in state:
        logger.info("Already have ticket data or answer, skipping to generate_answer")
        state["next_step"] = "generate_answer"
        state["plan"] = "Answer directly with the information already available."
        
        # Add to executed steps
        if "executed_steps" not in state:
            state["executed_steps"] = []
        state["executed_steps"].append("create_plan")
        
        return state
    
    logger.info(f"Creating plan for task: {state.get('task')}")
    
    # Get the task in the most usable format
    task = state.get("task", {})
    enriched_task = ""
    
    if isinstance(task, dict):
        # If task is a dictionary, try to get original key or use the whole dict
        enriched_task = task.get("original", str(task))
    elif isinstance(task, str):
        enriched_task = task
    else:
        # Convert anything else to string
        enriched_task = str(task)
    
    # Enrich with memories if available
    memories = state.get("memories", [])
    if memories:
        enriched_task += "\n\nRelevant context from memories:"
        for i, memory in enumerate(memories[:3], 1):  # Use top 3 memories
            if isinstance(memory, dict):
                q = memory.get("question", "")
                a = memory.get("answer", "")
                if q and a:
                    enriched_task += f"\n{i}. Q: {q}\nA: {a}"
    
    # Determine the appropriate agent
    prompt = f"""
    Task: {enriched_task}
    
    Available agents:
    1. Document Agent - For analyzing documents, managing documents, etc.
    
    Determine which agent is most appropriate for this task. If none is clearly appropriate, choose 'generate_answer'.
    
    Return ONLY one of these values: 'document', or 'generate_answer'.
    """
    
    # Get initial response
    try:
        logger.info("Calling LLM to determine agent")
        response = call_llm_sync(prompt)
        next_step = ""
        
        # Extract content from response
        if hasattr(response, "content"):
            next_step = response.content
        else:
            next_step = str(response)
        
        # Normalize the response
        if "document" in next_step.lower():
            next_step = "document"
        else:
            next_step = "generate_answer"
            
        logger.info(f"Determined next_step: {next_step}")
        
    except Exception as e:
        logger.error(f"Error determining agent: {str(e)}")
        # Default to generate_answer if there's an error
        next_step = "generate_answer"
    
    # Create a detailed plan
    prompt_plan = f"""
    Create a detailed plan for handling this task:
    
    Task: {enriched_task}
    
    Selected agent: {next_step}
    
    Provide a step-by-step plan for how this agent should address the task.
    """
    
    # Get plan
    try:
        logger.info("Calling LLM to create plan")
        response_plan = call_llm_sync(prompt_plan)
        plan = ""
        
        # Extract content from response
        if hasattr(response_plan, "content"):
            plan = response_plan.content
        else:
            plan = str(response_plan)
            
        logger.info("Plan created successfully")
        
    except Exception as e:
        logger.error(f"Error creating plan: {str(e)}")
        # Use a simple fallback plan
        plan = f"Process the request using the {next_step} agent."
    
    # Update the state
    state["next_step"] = next_step
    state["plan"] = plan
    
    # Add to executed steps
    if "executed_steps" not in state:
        state["executed_steps"] = []
    state["executed_steps"].append("create_plan")
    
    logger.info(f"Created plan with next step: {next_step}")
    
    return state

def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an answer to the user's task.
    """
    try:
        logger.info(f"Generating answer with state keys: {list(state.keys())}")
        
        task = state.get("task", {})
        memories = state.get("memories", [])
        plan = state.get("plan", "")
        
        # Extract the original question from task
        original_question = ""
        if isinstance(task, dict):
            original_question = task.get("original", "")
        elif isinstance(task, str):
            original_question = task
        
        # If we still don't have a question, check if it's in the configurable
        if not original_question and "configurable" in state:
            original_question = state.get("configurable", {}).get("question", "")
        
        if not original_question:
            logger.warning("No question found in task for generate_answer")
            original_question = "No question provided"
        else:
            logger.info(f"Found question: {original_question}")
        
        # Format memories as relevant context
        memory_context = ""
        if memories:
            memory_context = "Based on previous interactions:\n"
            for i, memory in enumerate(memories[:3], 1):  # Use top 3 memories
                if isinstance(memory, dict):
                    q = memory.get("question", "")
                    a = memory.get("answer", "")
                    if q and a:
                        memory_context += f"{i}. Question: {q}\n   Answer: {a}\n\n"
        
        # Generate a response
        prompt = f"""
        You are an AI assistant. Please respond to the following question:
        
        Question: {original_question}
        
        {memory_context}
        
        {plan}
        
        Provide a helpful, accurate, and concise response.
        """
        
        logger.info("Calling LLM to generate answer")
        
        # Use try-except block for LLM call
        try:
            # Use call_llm_sync instead of call_llm to get a direct result instead of a generator
            response = call_llm_sync(prompt)
            
            # Extract the content from the response
            response_content = ""
            if hasattr(response, "content"):
                response_content = response.content
            else:
                # If we're dealing with a message object instead
                response_content = str(response)
                
            logger.info("Successfully generated response content")
                
        except Exception as e:
            logger.error(f"Error in LLM call: {str(e)}")
            # Provide a fallback response
            response_content = "I'm sorry, I wasn't able to process your request at this time. Please try again later."
        
        # Generate a unique response ID
        response_id = str(uuid.uuid4())
        
        # Create answer structure based on ChatOutput model
        answer = {
            "response": response_content,
            "summary": response_content,  # Needed for chat.py to display
            "response_id": response_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update the state
        state["answer"] = answer
        
        logger.info(f"Generated answer with response_id: {response_id}")
        
    except Exception as e:
        # Catch any exceptions in the generate_answer function
        logger.error(f"Fatal error in generate_answer: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create a fallback answer
        response_id = str(uuid.uuid4())
        fallback_content = "I encountered an error while processing your request. Please try again."
        
        state["answer"] = {
            "response": fallback_content,
            "summary": fallback_content,
            "response_id": response_id,
            "timestamp": datetime.now().isoformat()
        }
    
    return state

def save_memories(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save the conversation and extracted information to long-term memory.
    
    This tool saves the user's task, the system's response, and any extracted
    entities and context to long-term memory for future reference.
    """
    try:
        task = state.get("task", {})
        answer = state.get("answer", {})
        entities = state.get("entities", {})
        context = state.get("context", "")
        
        # Log state for debugging
        logger.info(f"Save memories called with state keys: {list(state.keys())}")
        
        # Extract the original task, handling different possible structures
        original_task = ""
        if isinstance(task, dict):
            original_task = task.get("original", "")
        elif isinstance(task, str):
            original_task = task
        
        # If we still don't have a task, check if it's in the configurable
        if not original_task and "configurable" in state:
            original_task = state.get("configurable", {}).get("question", "")
        
        # Handle the case when answer is missing or has a different structure
        response = ""
        response_id = str(uuid.uuid4())
        
        if isinstance(answer, dict):
            response = answer.get("response", "")
            if not response:
                response = answer.get("summary", "")
            response_id = answer.get("response_id", response_id)
        
        if not original_task or not response:
            # Log the state for debugging
            logger.warning(f"Missing task or response for memory saving. State keys: {list(state.keys())}")
            logger.warning(f"Task type: {type(task)}, content: {task}")
            logger.warning(f"Answer type: {type(answer)}, content: {answer}")
            
            # Don't fail the chain, just return the state as is
            return state
        
        # Initialize long-term memory
        ltm = LongTermMemory(customer="default")
        
        # Save to long-term memory
        try:
            ltm.save_question_answer(
                question=original_task,
                answer=response,
                response_id=response_id,
                metadata={
                    "entities": entities,
                    "context": context,
                    "session_id": task.get("session_id", "") if isinstance(task, dict) else "",
                    "thread_id": task.get("thread_id", "") if isinstance(task, dict) else ""
                }
            )
            
            logger.info(f"Saved memory with response_id: {response_id}")
            
            # Update the answer in the state with the response_id
            if isinstance(answer, dict):
                answer["response_id"] = response_id
                state["answer"] = answer
            else:
                state["answer"] = {
                    "response": response,
                    "response_id": response_id,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error saving memory to LongTermMemory: {str(e)}")
            logger.error(f"Error details: {str(e)}")
    
    except Exception as e:
        # Catch any other exceptions during the memory saving process
        logger.error(f"Fatal error in save_memories: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return state 