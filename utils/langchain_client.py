from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class LangChainClient:
    """
    A client to interact with different LLM vendors using LangChain as an orchestrator.
    It supports hallucination checks, document relevancy checks, intent classification, 
    response generation, and query reformulation.
    """
    def __init__(self, llm_vendor: str, model_name: str, api_key: str , temperature: float = 0.01, max_tokens: int  = 1000):
        """
        Initializes the LangChainClient with a specific LLM vendor and model.
        """
        if llm_vendor.lower() == "openai":
            self.llm = ChatOpenAI(model_name=model_name, api_key=api_key,temperature=temperature, max_tokens = max_tokens)
        else:
            raise ValueError(f"Unsupported LLM vendor: {llm_vendor}")
    
    async def generate_response(self, query: str, documents: List[str], prompt: str) -> str:
        """Generates a response using the LLM based on the user query and retrieved documents."""
        prompt_template = PromptTemplate.from_template(prompt)
        
        chain = (
            {"query": RunnablePassthrough(), "documents": lambda x: "\n".join(documents)}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        response = await chain.ainvoke(query)
        return response
    
    async def hallucination_check(self, query: str, response: str, context: List[str], prompt: str) -> bool:
        """Checks if the generated response contains hallucinations by comparing it against retrieved context."""
        prompt_template = PromptTemplate.from_template(prompt)
        
        chain = (
            {
                "query": lambda x: query, 
                "response": lambda x: response, 
                "context": lambda x: "\n".join(context)
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        validation_result = await chain.ainvoke({})
        return validation_result.strip()
    
    async def classify_intent(self, query: str, prompt: str) -> str:
        """Classifies the intent of the user query."""
        prompt_template = PromptTemplate.from_template(prompt)
        
        chain = (
            {"query": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        intent = await chain.ainvoke(query)
        return intent.strip()
    
    async def reformulate_query(self, query: str, short_term_memory: List[str], prompt: str) -> str:
        """
        Reformulates a user query based on short-term memory to improve retrieval accuracy.
        Short-term memory contains recent interactions to provide context for refinement.
        """
        prompt_template = PromptTemplate.from_template(prompt)
        
        chain = (
            {
                "query": lambda x: query, 
                "history": lambda x: "\n".join(short_term_memory)
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        reformulated_query = await chain.ainvoke({})
        return reformulated_query.strip()