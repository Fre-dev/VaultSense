from typing import Dict, Any, Optional
from src.llm.runner import get_llm_model, call_llm, call_llm_sync
from src.agents.document import DocumentAgent
from src.tools.supervisor import (
    create_plan,
    generate_answer,
    retrieve_memories,
    save_memories,
)
from src.states.supervisor import SupervisorState

from langgraph.graph import StateGraph

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

class SupervisorAgent:
    """
    Supervisor agent that coordinates between different agents.
    This agent analyzes user requests and routes them to the appropriate agent.
    """
    def __init__(self):
        self.model = get_llm_model()
        self.document_agent = DocumentAgent()
        self.app = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("retrieve_memories", retrieve_memories)
        workflow.add_node("document_agent", self.document_agent.app)
        workflow.add_node("create_plan", create_plan)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("save_memories", save_memories)

        # Define conditional routing map
        conditional_map = {
            "document": "document_agent",
            "generate_answer": "generate_answer"
        }

        # Set up the workflow
        workflow.add_edge("retrieve_memories", "create_plan")
        
        # Add conditional routing based on the next_step determined in create_plan
        workflow.add_conditional_edges(
            "create_plan", lambda x: x.get("next_step", ""), conditional_map
        )
        
        # Connect agent outputs with conditional routing based on task completion
        workflow.add_conditional_edges(
            "document_agent", check_agent_completion, 
            {
                "create_plan": "create_plan", 
                "generate_answer": "generate_answer",
                "save_memories": "save_memories"
            }
        )
        
        workflow.add_conditional_edges(
            "document_agent", check_agent_completion, 
            {
                "create_plan": "create_plan", 
                "generate_answer": "generate_answer",
                "save_memories": "save_memories"
            }
        )
        
        # Connect generate_answer to save_memories
        workflow.add_edge("generate_answer", "save_memories")

        # Set entry and exit points
        workflow.set_entry_point("retrieve_memories")
        workflow.set_finish_point("save_memories")

        return workflow.compile()

    def _determine_agent(self, task: str) -> str:
        """Determine which agent should handle the task"""
        messages = [
            {"role": "system", "content": "You are an AI assistant that determines which agent should handle a specific task."},
            {"role": "user", "content": f"Based on this request: '{task}', determine which agent should handle it: 'document'. If none apply, say 'generate_answer'. Return ONLY the agent name."}
        ]
        
        response = call_llm(messages)
        agent = response.content.lower().strip()
        
        # Normalize the agent name
        if "document" in agent:
            return "document"
        else:
            return "generate_answer"

    def run(self, input: Dict[str, Any]):
        """
        Run the supervisor agent to process the user's request.
        
        Args:
            input: Dictionary containing user and task information
        
        Returns:
            Streaming response from the agent workflow
        """
        logger.info(f"Supervisor Agent received input: {input}")
        return self.app.stream(input)
