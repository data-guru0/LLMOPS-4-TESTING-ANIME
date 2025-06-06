from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir: str = "chroma_db"):
        # Store the CSV file path and directory to save the vector store
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        
        # Initialize the HuggingFace embedding model to convert text into vectors
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_and_save_vectorstore(self):
        # Load data from CSV file without extracting any extra metadata columns
        loader = CSVLoader(
            file_path=self.csv_path,
            encoding="utf-8",
            metadata_columns=[]  # Do not include any metadata columns
        )
        data = loader.load()  # Load the CSV content as documents

        # Split the documents into smaller chunks of 1000 characters without overlap
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(data)  # Split loaded documents into chunks

        # Create a Chroma vector store from the text chunks using the embedding model
        db = Chroma.from_documents(texts, self.embedding, persist_directory=self.persist_dir)
        db.persist()  # Save the vector store to disk in the persist directory
        
        print(f"[✅] VectorStore built and saved to `{self.persist_dir}`")

    def load_vectorstore(self):
        # Load the saved vector store from disk using the same embedding function
        return Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding)
