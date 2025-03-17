from pydantic import BaseModel


class QueryResponse(BaseModel):
    response: str


class QueryRequest(BaseModel):
    user_id: int
    group_id: str
    query: str
    session_id: str