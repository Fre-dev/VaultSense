from typing import Dict, Any, TypedDict, Optional, List, Literal
from langgraph.graph.message import add_messages
from typing import Annotated

class SupervisorState(TypedDict, total=False):
    """
    State model for the SupervisorAgent workflow.
    
    Attributes:
        user: Information about the user making the request
        task: The original task from the user and its enriched form
        memories: Relevant memories retrieved from long-term storage
        entities: Entities extracted from the conversation
        context: Context information extracted from the conversation
        next_step: The next step in the workflow (document, generate_answer)
        document_result: Result from the Document agent
        plan: The plan created for addressing the user's task
        answer: The final answer to return to the user
        messages: Messages for langgraph communication
        executed_steps: Steps that have been executed in the workflow
    """
    user: Dict[str, Any]
    task: Dict[str, str]
    memories: Optional[List[Dict[str, Any]]]
    entities: Optional[Dict[str, Any]]
    context: Optional[str]
    next_step: Optional[str]
    document_result: Optional[Dict[str, Any]]
    plan: Optional[str]
    answer: Optional[Dict[str, Any]]
    messages: Annotated[list, add_messages]
    executed_steps: Annotated[
        Literal["document", "generate_answer"], add_messages
    ] 