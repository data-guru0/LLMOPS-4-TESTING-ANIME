from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

# Set up logger for this module to record info and errors
logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self, persist_dir="chroma_db"):
        try:
            logger.info("Initializing AnimeRecommendationPipeline...")

            # Create a VectorStoreBuilder instance (empty CSV path because we only want to load)
            vector_builder = VectorStoreBuilder(csv_path="", persist_dir=persist_dir)
            
            # Load the existing vector store from disk and create a retriever interface
            retriever = vector_builder.load_vectorstore().as_retriever()
            
            # Initialize the AnimeRecommender with the retriever, API key, and model name
            self.recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)

            logger.info("AnimeRecommendationPipeline initialized successfully.")
        except Exception as e:
            # Log any error during initialization and raise a custom exception
            logger.error(f"Failed to initialize AnimeRecommendationPipeline: {e}")
            raise CustomException("Error during AnimeRecommendationPipeline initialization", e)

    def recommend(self, query: str) -> str:
        try:
            logger.info(f"Received recommendation query: {query}")
            
            # Use the recommender to get an anime recommendation for the input query
            recommendation = self.recommender.get_recommendation(query)
            
            logger.info("Recommendation generated successfully.")
            return recommendation
        except Exception as e:
            # Log any error during recommendation and raise a custom exception
            logger.error(f"Failed to get recommendation: {e}")
            raise CustomException("Error while getting recommendation", e)
