from pydantic import BaseModel


class Candidate(BaseModel):
    document_id: int
    title: str


class RankedCandidate(Candidate):
    score: float