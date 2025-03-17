from fastapi import APIRouter,HTTPException
import datetime
from models.sessions_model import CreateSessionRequest, CreateSessionResponse, GetSessionsResponse, Session, HistoryResponse
from services.session_service import SessionService

sessions_router = APIRouter()


@sessions_router.post("/create", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    session = await SessionService.create_session(request.user_id, request.title)
    return CreateSessionResponse(session_id=str(session["_id"]))


@sessions_router.get("/get_all_sessions/{user_id}", response_model=GetSessionsResponse)
async def get_sessions(user_id: int):
    sessions = await SessionService.get_sessions_for_user(user_id)
    # Sort sessions by updated_at (latest first)
    sorted_sessions = sorted(
        sessions,
        key=lambda s: s.get("updated_at", datetime.datetime.min),
        reverse=True,
    )
    formatted_sessions = []
    for s in sorted_sessions:
        formatted_sessions.append(
            Session(
                id=str(s["_id"]),
                user_id=s["user_id"],
                title=s["title"],
                updated_at=s["updated_at"],
            )
        )
    return GetSessionsResponse(sessions=formatted_sessions)

@sessions_router.get("/get_session/{session_id}", response_model=HistoryResponse)
async def get_session_history(session_id: str):
    session = await SessionService.get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return HistoryResponse(history=session.get("history", []))
