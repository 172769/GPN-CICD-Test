from fastapi import FastAPI
from api.query_inference import query_inference_router
from api.session import sessions_router
app = FastAPI()

app.include_router(query_inference_router,prefix="/query")
app.include_router(sessions_router,prefix="/sessions")