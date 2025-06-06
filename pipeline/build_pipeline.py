from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)

def main():
    try:
        logger.info("[🚀] Starting pipeline build...")

        loader = AnimeDataLoader("data/anime_with_synopsis.csv", "data/anime_updated.csv")
        processed_csv = loader.load_and_process()
        logger.info(f"Data loaded and processed. Output CSV: {processed_csv}")

        vector_builder = VectorStoreBuilder(processed_csv)
        vector_builder.build_and_save_vectorstore()
        logger.info("Vector store built and saved successfully.")

        logger.info("[✅] Pipeline build complete. Ready for app use.")

    except Exception as e:
        raise CustomException("Error occurred during pipeline build", e)

if __name__ == "__main__":
    main()
