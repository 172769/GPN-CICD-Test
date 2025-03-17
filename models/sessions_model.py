from pydantic import BaseModel
import datetime
from typing import List, Optional, Any

# Pydantic models for creating a session
class CreateSessionRequest(BaseModel):
    user_id: int
    title: Optional[str] = "New Chat"


class CreateSessionResponse(BaseModel):
    session_id: str


# Pydantic model for a session
class Session(BaseModel):
    id: str
    user_id: int
    title: str
    updated_at: datetime.datetime


class GetSessionsResponse(BaseModel):
    sessions: List[Session]

class HistoryResponse(BaseModel):
    history: List[Any]