import os
import asyncio
from services.query_reformulation import QueryReformulationService
from services.intent_classifier import IntentClassificationService
from services.hallucination_checker import HallucinationCheckService
from services.response_generator import ResponseGeneratorService
from dotenv import load_dotenv

load_dotenv()


api_key_value = os.getenv('API_KEY')

async def test_query_reformulation():
    # Replace these with your actual vendor credentials.
    vendor = "openai"
    model_name = "gpt-4o"  # Example model, adjust as needed.
    api_key = api_key_value      # Replace with your actual API key.
    
    # Create an instance of the QueryReformulationService.
    service = QueryReformulationService(vendor, model_name, api_key)
    
    # Test input: a sample query and an empty short-term memory list.
    query = "why the sky is blue?"
    short_term_memory = []  # For testing, this is empty; add previous messages if available.
    
    reformulated_query = await service.reformulate_query(query, short_term_memory)
    print("Original Query:", query)
    print("Reformulated Query:", reformulated_query)


async def test_intent_classification():
    # Replace these with your actual credentials.
    vendor = "openai"
    model_name = "gpt-4o"  # Example model, adjust as needed.
    api_key = api_key_value      # Replace with your actual API key.
    
    # Instantiate the IntentClassificationService.
    service = IntentClassificationService(vendor, model_name, api_key)
    
    # Test query: one expected to be domain-related.
    query = "What are the latest changes in credit card policies?"
    
    intent = await service.classify_intent(query)
    print("Query:", query)
    print("Classified Intent:", intent)


async def test_hallucination_check():
    vendor = "openai"
    model_name = "gpt-4o"  # Example model, adjust as needed.
    api_key = api_key_value     # Replace with your actual API key.
    
    service = HallucinationCheckService(vendor, model_name, api_key)
    
    query = "What improvements does our latest payment gateway bring?"
    response = (
        "Our new payment gateway increases transaction speeds by 50% and introduces an AI-powered fraud detection system "
        "that virtually eliminates fraud, making transactions nearly flawless."
    )
    context = [
        "The new payment gateway offers a moderate improvement in transaction speed over the previous system.",
    "While the fraud detection system uses AI, it does not completely eliminate fraud.",
    "Improvements are notable but still fall within the range of typical industry upgrades."
]


    
    score = await service.check_hallucination(query, response, context)
    print("Factual Consistency Score:", score)

async def test_response_generation():
    # Replace these with your actual credentials.
    vendor = "openai"
    model_name = "gpt-4o"  # Example model, adjust as needed.
    api_key = api_key_value      # Replace with your actual API key.
    
    # Create a sample list of documents (simulate output from vector search).
    documents = [
        {
            "content": "Our global payment system now supports digital currencies and blockchain technology has been integrated.",
            "document_name": "Global Payments Overview",
            "page_number": 1,
            "file_link": "https://example.com/doc1"
        },
        {
            "content": "Credit card policies and banking guidelines are updated quarterly to comply with regulations.",
            "document_name": "Banking Regulations Q2",
            "page_number": 3,
            "file_link": "https://example.com/doc2"
        }
    ]
    
    query = "What are the recent updates to our global payment system?"
    
    # Instantiate the response generator service.
    response_service = ResponseGeneratorService(vendor, model_name, api_key)
    generated_response = await response_service.generate_response(query, documents)
    
    print("Generated Response:\n", generated_response)

if __name__ == "__main__":
    #asyncio.run(test_query_reformulation())
    #asyncio.run(test_intent_classification())
    #asyncio.run(test_hallucination_check())
    asyncio.run(test_response_generation())


   