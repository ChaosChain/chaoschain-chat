import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

# --- Path Configuration ---
# Get the absolute path of the project root directory (assuming script is in backend/scripts)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data" # Store data within backend for simplicity now
VECTOR_STORE_PATH = str(DATA_DIR / "vector_store")
REPO_PARENT_PATH = DATA_DIR / "repos"
# --------------------------

# Add the backend directory to the Python path
sys.path.append(str(BACKEND_DIR))

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings

def clone_or_pull_repository(repo_url: str, target_dir: Path) -> bool:
    """Clone a repository if it doesn't exist, or pull latest changes if it does."""
    try:
        if not target_dir.exists():
            print(f"Cloning repository to {target_dir}")
            os.makedirs(target_dir.parent, exist_ok=True)
            subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
        else:
            print(f"Updating existing repository at {target_dir}")
            current_dir = os.getcwd()
            os.chdir(target_dir)
            subprocess.run(["git", "pull"], check=True)
            os.chdir(current_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error with repository operation: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def load_documents(source_dir: str) -> List[Document]:
    """Load documents from multiple file types using manual glob for md/mdx."""
    documents: List[Document] = []
    source_path = Path(source_dir)

    # 1. Manual Loading for .md and .mdx files using rglob
    print(f"Manually searching for .md and .mdx files in: {source_path}")
    md_files = list(source_path.rglob('*.md'))
    mdx_files = list(source_path.rglob('*.mdx'))
    print(f"Found {len(md_files)} .md files and {len(mdx_files)} .mdx files.")

    for file_path in md_files + mdx_files:
        try:
            print(f"Loading: {file_path}")
            loader = UnstructuredMarkdownLoader(str(file_path))
            loaded_docs = loader.load() # Returns a list
            if loaded_docs:
                # Add metadata
                doc = loaded_docs[0] # Assume one doc per file
                ext = file_path.suffix
                doc.metadata['type'] = 'mdx' if ext == '.mdx' else 'markdown'
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    # 2. DirectoryLoader for .mmd files (keep as it seems to work)
    print(f"Attempting to load .mmd from: {source_dir} using DirectoryLoader")
    try:
        mmd_loader = DirectoryLoader(
            path=source_dir,
            glob="**/*.mmd",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True,
            recursive=True
        )
        mmd_docs = mmd_loader.load()
        print(f"Found {len(mmd_docs)} .mmd documents via DirectoryLoader.")
        for doc in mmd_docs:
            doc.metadata['type'] = 'mermaid'
        documents.extend(mmd_docs)
    except Exception as e:
        print(f"Error loading .mmd files via DirectoryLoader: {e}")

    # 3. Explicit loading for README.md (already handled, keep)
    # The previous edit already added logic for README.md which is good.
    # We might need to ensure it's not double-loaded by the .md rglob. 
    # Let's refine the rglob logic slightly to exclude the exact README path if loaded separately.
    # However, the current code adds it separately and extends, which is fine too.

    # Check if README was loaded by rglob and avoid double adding if necessary
    # (Simpler: the loop above already handled it if found by rglob)

    print(f"Total documents loaded: {len(documents)}")
    return documents

def create_vector_store(documents: List, persist_dir: str, embedding_model_name: str):
    """Create and persist the vector store from documents."""
    print(f"Splitting documents into chunks (size: {settings.chunk_size}, overlap: {settings.chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Initialize embedding model
    print(f"Using Local HuggingFace Embeddings: {embedding_model_name}")
    embedding_function = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'}
    )

    # Create or clear vector store
    persist_path = Path(persist_dir)
    if persist_path.exists():
        print(f"Clearing existing vector store at: {persist_path}")
        shutil.rmtree(persist_path)
    os.makedirs(persist_path, exist_ok=True)

    print(f"Creating vector store using embeddings: {embedding_function.__class__.__name__}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    vector_store.persist()
    print(f"Vector store created and persisted at: {persist_dir}")

def ingest_repository(
    source: str,
    # target_dir is now calculated internally based on PROJECT_ROOT
    vector_store_dir: str = VECTOR_STORE_PATH, # Use constant path
    embedding_model: Optional[str] = None
) -> bool:
    """
    Ingest content from a git repository and create a vector store.
    
    Args:
        source: Either a GitHub repository URL or owner/repo string
        vector_store_dir: Directory to store the vector database (Defaults to backend/data/vector_store)
        embedding_model: Name of the embedding model to use
    """
    # Use constant/default paths unless overridden 
    vector_store_dir = vector_store_dir or VECTOR_STORE_PATH
    embedding_model = embedding_model or settings.embedding_model_name

    # --- Calculate target_dir based on project root --- 
    if "/" in source and not os.path.exists(source):
        if not source.startswith(("http://", "https://", "git@")):
            source = f"https://github.com/{source}.git"
        repo_name = source.split("/")[-1].replace(".git", "")
        target_dir = REPO_PARENT_PATH / repo_name # Use constant parent path
    elif os.path.exists(source):
        # Handle case where source is a local path
        target_dir = Path(source).resolve() # Use resolved absolute path
    else:
        print(f"Error: Invalid source specified: {source}")
        return False
    # --------------------------------------------------

    # Clone or update the repository
    success = clone_or_pull_repository(source, target_dir)
    if not success:
        return False

    # Load documents from the repository (use absolute path)
    try:
        print(f"Loading documents from: {target_dir.absolute()}")
        # Pass the absolute path to load_documents
        documents = load_documents(str(target_dir.absolute())) 
        if not documents:
            print("No documents found to ingest.")
            return False
        
        print(f"Loaded {len(documents)} documents.")
        # Extract unique source types for better logging
        doc_types = set()
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            doc_types.add(doc_type)
        print(f"Document types found: {doc_types}")

        # Create vector store
        create_vector_store(documents, vector_store_dir, embedding_model)
        return True

    except Exception as e:
        print(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest content from a git repository and create a vector store")
    parser.add_argument("source", help="GitHub repository URL, owner/repo, or local repository path")
    # Remove target-dir argument as it's now calculated
    # parser.add_argument("--target-dir", help="Directory to store the cloned repository") 
    parser.add_argument("--vector-store-dir", default=VECTOR_STORE_PATH, help=f"Directory to store the vector database (default: {VECTOR_STORE_PATH})")
    parser.add_argument("--embedding-model", help="Name of the embedding model to use")
    
    args = parser.parse_args()
    
    success = ingest_repository(
        args.source,
        # No target_dir argument needed here
        args.vector_store_dir,
        args.embedding_model
    )
    
    if success:
        print(f"Successfully ingested repository: {args.source}")
    else:
        print(f"Failed to ingest repository: {args.source}")
        sys.exit(1) 