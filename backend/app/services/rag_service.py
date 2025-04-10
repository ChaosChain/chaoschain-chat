from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import subprocess
from pathlib import Path
import os
import time
import shutil
from typing import Dict, Any, List

from app.core.config import settings

# Langchain imports
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
from app.core.config import settings

# --- Enhanced Logging --- 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ----------------------

def update_repository(repo_dir: Path) -> bool:
    """Pull latest changes from the repository."""
    try:
        # Change to the repository directory
        current_dir = os.getcwd()
        os.chdir(repo_dir)
        
        # Pull latest changes
        subprocess.run(["git", "pull"], check=True)
        
        # Change back to original directory
        os.chdir(current_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error updating repository: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error updating repository: {e}")
        return False

class RAGService:
    def __init__(self):
        logger.info("--- RAGService Initialization Started ---")
        self.embedding_model_name = settings.embedding_model_name
        self.vector_db_dir = Path(settings.vector_db_dir)
        self.openai_embedding_model = settings.openai_embedding_model
        self.openai_chat_model = settings.openai_chat_model
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.search_k = settings.search_k
        self.litepaper_src_dir = settings.litepaper_src_dir

        self.embedding = OpenAIEmbeddings(
            model=self.openai_embedding_model, 
            openai_api_key=settings.openai_api_key
        )
        logger.info(f"Initialized Embedding Model: {self.openai_embedding_model}")

        self.vector_store = self._load_or_create_vector_store()
        logger.info(f"Vector Store Ready. Path: {self.vector_db_dir}")

        self.llm = ChatOpenAI(
            model_name=self.openai_chat_model,
            temperature=0, 
            openai_api_key=settings.openai_api_key
        )
        logger.info(f"Initialized LLM: {self.openai_chat_model}")

        self.chain = self._create_rag_chain()
        logger.info("RAG Chain Created.")
        logger.info("--- RAGService Initialization Complete ---")

    def _load_or_create_vector_store(self) -> Chroma:
        logger.info(f"Attempting to load vector store from: {self.vector_db_dir}")
        if self.vector_db_dir.exists() and any(self.vector_db_dir.iterdir()):
            logger.info("Existing vector store found. Loading...")
            try:
                vector_store = Chroma(persist_directory=str(self.vector_db_dir), embedding_function=self.embedding)
                logger.info("Vector store loaded successfully.")
                return vector_store
            except Exception as e:
                logger.error(f"Error loading existing vector store: {e}. Attempting to recreate.")
                # Optionally remove corrupted dir: shutil.rmtree(self.vector_db_dir)
        
        logger.warning(f"Vector store not found or empty at {self.vector_db_dir}. Creating new store...")
        docs = self._load_documents()
        logger.info(f"Loaded {len(docs)} documents.")
        if not docs:
            logger.error("No documents loaded, cannot create vector store.")
            raise ValueError("No documents were loaded from the source.")
        
        chunks = self._split_documents(docs)
        logger.info(f"Split documents into {len(chunks)} chunks.")

        logger.info("Creating vector store with new chunks...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=str(self.vector_db_dir)
        )
        logger.info("New vector store created and persisted.")
        return vector_store

    def _load_documents(self) -> List[Any]: # Return type depends on loader
        logger.info(f"Loading documents from source: {self.litepaper_src_dir}")
        if settings.is_github_repo:
            repo_name = settings.repo_name
            if not repo_name:
                raise ValueError("Could not determine repository name from source URL.")
            repo_dir = Path("data/repos") / repo_name
            logger.info(f"GitHub repository detected. Cloning/updating to: {repo_dir}")
            
            if not repo_dir.exists():
                logger.info("Cloning repository...")
                try:
                    # Ensure the target directory exists
                    repo_dir.parent.mkdir(parents=True, exist_ok=True)
                    # Clone the repository
                    logger.info(f"Cloning repository {self.litepaper_src_dir} to {repo_dir}...")
                    loader = GitLoader(clone_url=self.litepaper_src_dir, repo_path=str(repo_dir), branch="master")
                    docs = loader.load()
                    logger.info(f"Repository cloned. Loaded {len(docs)} files.")
                    return docs
                except Exception as e:
                    logger.error(f"Error cloning repository: {e}")
                    raise
            else:
                logger.info("Repository exists. Attempting to update...")
                update_repository(repo_dir) # Assumes update_repository logs its own errors
                logger.info("Loading from existing repository path...")
                # Pass the branch name when loading from existing repo too
                loader = GitLoader(repo_path=str(repo_dir), branch="master")
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} files from existing repo.")
                return docs
        else:
            # Handle local directory loading if needed (adjust loader)
            logger.error(f"Source is not a GitHub repo URL: {self.litepaper_src_dir}. Local directory loading not fully implemented.")
            raise NotImplementedError("Loading from local directories requires a different DocumentLoader configuration.")

    def _split_documents(self, docs: List[Any]) -> List[Any]:
        logger.info(f"Splitting {len(docs)} documents. Chunk Size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

    def _create_rag_chain(self):
        logger.info("Creating RAG chain...")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.search_k})
        logger.info(f"Retriever configured with search_k={self.search_k}")
        
        template = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        Context: {context} 
        Answer:"""
        prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain setup complete.")
        return rag_chain

    async def get_answer(self, query: str) -> Dict[str, Any]:
        logger.info(f"--- RAGService.get_answer called with query: '{query}' ---")
        if not self.chain:
            logger.error("RAG chain is not initialized.")
            return {"answer": "Error: RAG chain not initialized.", "sources": []}
        if not self.vector_store:
             logger.error("Vector store is not initialized.")
             return {"answer": "Error: Vector store not initialized.", "sources": []}

        try:
            # Retrieve relevant documents first for source tracking
            retriever = self.vector_store.as_retriever(search_kwargs={"k": self.search_k})
            logger.info(f"Retrieving documents for query: '{query}'")
            retrieved_docs = retriever.invoke(query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            
            # Log retrieved doc metadata (e.g., source file) - adjust based on Document structure
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Unknown')
                # Try to get a snippet of content
                content_snippet = doc.page_content[:100].replace("\n", " ") + "..."
                logger.info(f"  Doc {i+1}: Source='{source}', Snippet='{content_snippet}'")

            # Generate the answer using the RAG chain
            logger.info(f"Invoking RAG chain with query: '{query}'")
            # Note: The context is handled internally by the chain definition
            answer = await self.chain.ainvoke(query)
            logger.info(f"RAG chain generated answer: '{answer}'")

            # Format sources (adjust structure as needed based on Document metadata)
            sources_metadata = []
            for doc in retrieved_docs:
                 source_path = doc.metadata.get("source", "Unknown")
                 # Attempt to create a URL if it's a GitHub source
                 url = "#" # Default URL
                 if settings.is_github_repo and "data/repos" in source_path:
                     relative_path = Path(source_path).relative_to(Path("data/repos") / settings.repo_name)
                     # Remove .md extension for URL typically
                     url_path = str(relative_path).replace("\\", "/") # Ensure forward slashes
                     base_url = settings.litepaper_src_dir.replace(".git", "") # Base repo URL
                     # Construct a likely URL (might need adjustment based on site structure)
                     url = f"{base_url}/blob/main/{url_path}"
                 
                 sources_metadata.append({
                     "name": Path(source_path).name,
                     "url": url,
                     "page_content": doc.page_content # Include full content if needed downstream
                 })

            return {"answer": answer, "sources": sources_metadata}
        except Exception as e:
            logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
            return {"answer": f"An error occurred: {e}", "sources": []}

# Singleton instance
_rag_service_instance: RAGService | None = None

def get_rag_service() -> RAGService:
    # Indent the following lines correctly
    global _rag_service_instance
    logger.info("get_rag_service called")
    if _rag_service_instance is None:
        logger.info("No existing RAGService instance found, creating new one...")
        _rag_service_instance = RAGService() # This line and subsequent lines were likely unindented
        logger.info("New RAGService instance created.")
    else:
        logger.info("Returning existing RAGService instance.")
    return _rag_service_instance