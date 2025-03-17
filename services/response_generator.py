from typing import List, Dict, Optional
from utils.langchain_client import LangChainClient

class ResponseGeneratorService:
    """
    Service to generate an answer with references using LangChainClient.
    
    The generated response includes inline reference citations (e.g., [1], [2])
    and a separate references section at the bottom that lists the document name,
    page number, and file link for each cited document.
    """
    
    def __init__(self, vendor: str, model_name: str, api_key: str):
        """
        Initializes the ResponseGeneratorService with a LangChainClient.
        
        Parameters:
          - vendor: The LLM vendor (e.g., "openai").
          - model_name: The model name to be used.
          - api_key: The API key for the vendor.
        """
        self.langchain_client = LangChainClient(llm_vendor=vendor, model_name=model_name, api_key=api_key)
    
    async def generate_response(self, query: str, documents: List[Dict], prompt: Optional[str] = None) -> str:
        """
        Generates a response based on the query and provided documents.
        
        The output will include inline citations (like [1], [2], etc.) within the answer,
        and then a "References:" section listing each reference with document name, page number, and link.
        
        Parameters:
          - query: The user's query.
          - documents: A list of dictionaries, each containing keys such as:
                - content
                - document_name
                - page_number
                - file_link
          - prompt: Optional custom prompt template. If not provided, a default prompt is used.
        
        Returns:
          - A string containing the generated answer with inline references and a references section.
        """
        if prompt is None:
            prompt = (
                "# Expert Response Generation Task\n\n"
                "## Context and Role\n"
                "You are an expert AI assistant tasked with providing accurate, concise answers based solely on the provided documents. Your response must be fully grounded in these documents with no external knowledge or speculation.\n\n"
                "## Documents\n"
                "{documents}\n\n"
                "## User Query\n"
                "{query}\n\n"
                "## Response Requirements\n"
                "1. Provide a clear, direct answer to the query based only on the provided documents\n"
                "2. Use inline citations in the format [1], [2], etc. after each sentence or claim\n"
                "3. If multiple documents support a claim, include all relevant citations: [1][3]\n"
                "4. If information is not found in the documents, state this clearly rather than speculating\n"
                "5. Synthesize information across documents when relevant\n"
                "6. Prioritize the most relevant information to the query\n"
                "7. Make sure to display any list or table information in a pretty way (markdown) \n"
                "8. Keep your answer concise and focused \n"
                "9. Use objective, factual language\n\n"
                "## References Section Format\n"
                "After your answer, include a 'References:' section with each cited source listed as:\n"
                "[#] Document: [document_name] | Page: [page_number] | Source: [file_link]\n\n"
                "## Answer Format\n"
                "[Your concise, well-structured answer with proper citations]\n\n"
                "References:\n"
                "[1] Document: [document_name] | Page: [page_number] | Source: [file_link]\n"
                "[2] Document: [document_name] | Page: [page_number] | Source: [file_link]\n"
                "...\n\n"
                "## Answer:"
            )
        
        # Prepare the document context as a single string, enumerating each document with an index.
        doc_context_lines = []
        for i, doc in enumerate(documents, start=1):
            # Assuming each document has 'content', 'document_name', 'page_number', and 'file_link'
            doc_line = (
                f"[{i}] Document Name: {doc.get('document_name', 'N/A')}, "
                f"Page: {doc.get('page_number', 'N/A')}, "
                f"Link: {doc.get('document_url', 'N/A')}\n"
                f"Content: {doc.get('content', '')}"
            )
            doc_context_lines.append(doc_line)
        doc_context = "\n\n".join(doc_context_lines)
        
        # Use the LangChainClient's generate_response method.
        # The chain expects a list of strings for the 'documents' parameter.
        response = await self.langchain_client.generate_response(query, [doc_context], prompt)
        return response


    
