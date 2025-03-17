from fastapi import APIRouter,HTTPException
import asyncio
import datetime
import os
from dotenv import load_dotenv
from models.query_model import QueryRequest, QueryResponse

from services.session_service import SessionService
from services.query_reformulation import QueryReformulationService
from services.intent_classifier import IntentClassificationService
from services.vector_search import VectorSearchService
from services.reranker import Reranker
from services.response_generator import ResponseGeneratorService
from services.hallucination_checker import HallucinationCheckService
from services.query_embedding import EmbeddingClient

load_dotenv()

query_inference_router = APIRouter()


LLM_VENDOR = os.getenv("LLM_VENDOR")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDING_VENDOR = os.getenv("EMBEDDING_VENDOR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
RERANKER_VENDOR = os.getenv("RERANKER_VENDOR")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY")

# Instantiate our service objects.
query_reformulation_service = QueryReformulationService(LLM_VENDOR, LLM_MODEL_NAME, LLM_API_KEY)
intent_classification_service = IntentClassificationService(LLM_VENDOR, LLM_MODEL_NAME, LLM_API_KEY)
vector_search_service = VectorSearchService()  
reranker = Reranker(RERANKER_VENDOR, RERANKER_API_KEY, RERANKER_MODEL_NAME)
response_generator_service = ResponseGeneratorService(LLM_VENDOR, LLM_MODEL_NAME, LLM_API_KEY)
hallucination_check_service = HallucinationCheckService(LLM_VENDOR, LLM_MODEL_NAME, LLM_API_KEY)
embedding_client = EmbeddingClient(EMBEDDING_VENDOR, EMBEDDING_MODEL_NAME, EMBEDDING_API_KEY)


@query_inference_router.post("/infer", response_model=QueryResponse)
async def infer(request: QueryRequest):
    # Step 1: Retrieve session history from MongoDB.
    session = await SessionService.get_session_by_id(request.session_id)
    print("got the session")
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Extract the last 3 interactions as short-term memory.
    history = session.get("history", [])
    short_term_memory = [
    f"Query: {interaction.get('reformulated_query', '')} | Response: {interaction.get('response', '')}"
    for interaction in history[-3:]
    ]
    print("got the short term memory")
    
    # Step 2: Query Reformulation.
    reformulated_query = await query_reformulation_service.reformulate_query(request.query, short_term_memory)
    print("got the reformulated query : ", reformulated_query)
    
    # Step 3: Intent Classification.
    intent = await intent_classification_service.classify_intent(reformulated_query)
    print("got the intent : ", intent)
    if intent.lower() == "non-domain":
        generated_response = "Sorry, I'm a bot specialized in banking and global payments."
        new_history_entry = {
        "query": request.query,
        "reformulated_query": reformulated_query,
        "response": generated_response,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        await SessionService.update_session_history(request.session_id, new_history_entry)
        return QueryResponse(response=generated_response)
    elif intent.lower() == "greeting":
        generated_response = "Hello and welcome to GPN chatbot!"
        new_history_entry = {
        "query": request.query,
        "reformulated_query": reformulated_query,
        "response": generated_response,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        await SessionService.update_session_history(request.session_id, new_history_entry)
        return QueryResponse(response=generated_response)
    
    # Step 4: Generate query embeddings (wrap the synchronous call in a thread).
    query_embedding = await asyncio.to_thread(embedding_client.generate_embedding, reformulated_query)
    print("got the query embedding")
    
    # Step 5: Vector Search: retrieve top 20 candidate documents.
    # (Assumes authorization filter is the group_id.)
    vector_results, _ = await vector_search_service.search(query_embedding, request.group_id, limit=10)
    print("got the vector results")
    if not vector_results:
        generated_response = "Sorry, I could not find relevant documents."
        new_history_entry = {
        "query": request.query,
        "reformulated_query": reformulated_query,
        "response": generated_response,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        await SessionService.update_session_history(request.session_id, new_history_entry)
        return QueryResponse(response=generated_response)
    
    # Step 6: Extract content strings from retrieved documents while keeping full metadata.
    documents = vector_results  # Each result is a dict with keys: content, document_name, page_number, file_link, etc.
    content_list = [doc.get("content", "") for doc in documents]
    
    # Step 7: Reranking: rank candidate documents using the reformulated query.
    rerank_results = reranker.rerank(reformulated_query, content_list, top_n=10)
    print("rerank results : \n", rerank_results)
    # Filter for top 3 documents with a relevance score of at least 50%.
    top_indices = [res["index"] for res in rerank_results if res["relevance_score"] >= 0.4][:3]
    print("got the top indices")
    print("top indices : \n", top_indices)
    if not top_indices:
        generated_response = "Sorry, I couldn't find a sufficiently relevant answer."
        new_history_entry = {
        "query": request.query,
        "reformulated_query": reformulated_query,
        "response": generated_response,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        await SessionService.update_session_history(request.session_id, new_history_entry)
        return QueryResponse(response=generated_response)
    
    # Map indices back to the full document metadata.
    top_documents = [documents[i] for i in top_indices]
    print("got the top documents")
    print("top documents : \n", top_documents)
    
    # Step 8: Generate the final response using the top documents.
    generated_response = await response_generator_service.generate_response(reformulated_query, top_documents)
    print("got the generated response")
    print("generated response : \n", generated_response)
    
    # Step 9: Hallucination Check: ensure factual consistency of the generated response.
    doc_texts = [doc.get("content", "") for doc in top_documents]
    consistency_score = await hallucination_check_service.check_hallucination(reformulated_query, generated_response, doc_texts)
    print("consistency score : ", consistency_score)
    if consistency_score < 90:
        # Regenerate response if factual consistency is low.
        generated_response = await response_generator_service.generate_response(reformulated_query, top_documents)
    
    # Step 10: Update session history with the new interaction.
    new_history_entry = {
        "query": request.query,
        "reformulated_query": reformulated_query,
        "response": generated_response,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        # "sources": [
        #     {
        #         "document_name": doc.get("document_name", "N/A"),
        #         "page_number": doc.get("page_number", "N/A"),
        #         "file_link": doc.get("file_link", "N/A")
        #     } for doc in top_documents
        # ]
    }
    await SessionService.update_session_history(request.session_id, new_history_entry)
    
    return QueryResponse(response=generated_response)