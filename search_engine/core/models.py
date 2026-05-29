from typing import Optional, Dict
from pydantic import BaseModel, Field

class Document(BaseModel):
    """
    Represents the full content and metadata of a searchable document.
    """
    id: Optional[int] = Field(default=None, description="Auto-incrementing primary key from the database")
    title: str = Field(..., description="The title of the blog, project, journal, or page")
    content: str = Field(..., description="The body content text of the document")
    type: str = Field(..., description="Type of document (e.g., Blog, Project, Journal, Page)")
    url: Optional[str] = Field(default=None, description="Optional URL to the document source")

class ScoredCandidate(BaseModel):
    """
    Lightweight intermediate structure representing a candidate matched during retrieval.
    Avoids loading full document contents into memory during early ranking phases.
    Uses an extensible dictionary to support adding future scoring features dynamically.
    """
    doc_id: int = Field(..., description="The database primary key of the candidate document")
    scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Dynamic mapping of signal names (e.g., 'prefix', 'trigram') to raw float scores"
    )
    score: float = Field(default=0.0, description="The final combined and weighted rank score")


class SearchResult(BaseModel):
    """
    Represents the final ranked search result returned to the user,
    hydrated with the original Document data.
    """
    document: Document = Field(..., description="The fully hydrated document from the storage layer")
    score: float = Field(..., description="The final rank score of this document for the search query")
