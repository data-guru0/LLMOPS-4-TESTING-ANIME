from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        # Initialize the ChatGroq LLM with the API key, model name, and zero temperature for deterministic answers
        self.llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
        
        # Get the prompt template specific for anime recommendation
        self.prompt = get_anime_prompt()
        
        # Create a RetrievalQA chain with the LLM and retriever
        # 'stuff' chain_type means it puts all retrieved documents into the prompt as context
        # return_source_documents=True lets us get the documents used for the answer (optional)
        # We pass our custom prompt template here
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def get_recommendation(self, query: str):
        # Run the query through the QA chain and get the answer
        result = self.qa_chain({"query": query})
        # Return just the answer text from the result dictionary
        return result['result']
