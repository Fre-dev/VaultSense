from typing import Dict, Any, TypedDict, Optional, List

class DocumentState(TypedDict, total=False):
    """
    State model for the DocumentAgent workflow.
    
    Attributes:
        task: The original task or query from the user
        operation: The operation to perform (create, update, fetch, search, comment)
        document_data: Data for documents
        document_id: ID of the specific document to operate on
        project_key: Project key
        issue_type: Type of document
        summary: Summary of the document
        description: Description of the document
        priority: Priority of the document
        assignee: User to assign the document to
        status: Status of the document
        comment: Comment to add to the document
        search_query: Query for searching documents
        milvus_search_results: Results from searching Milvus
        error: Error message if an operation fails
        result: Result of the operation
        answer: Final answer to return to the user
    """
    task: str
    operation: str
    document_data: Optional[Dict[str, Any]]
    document_id: Optional[str]
    project_key: Optional[str]
    issue_type: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    priority: Optional[str]
    assignee: Optional[str]
    status: Optional[str]
    comment: Optional[str]
    search_query: Optional[str]
    milvus_search_results: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    result: Optional[Dict[str, Any]]
    answer: Optional[Dict[str, Any]] 