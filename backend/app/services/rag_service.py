from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
import subprocess
from pathlib import Path
import os
import time

from app.core.config import settings

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
        print("Initializing RAGService...")
        
        # Handle GitHub repository if needed
        if settings.is_github_repo:
            print(f"GitHub repository detected: {settings.litepaper_src_dir}")
            repo_dir = Path("data/repos") / settings.repo_name
            if not repo_dir.exists():
                print(f"Cloning repository to {repo_dir}")
                os.makedirs(repo_dir.parent, exist_ok=True)
                subprocess.run(["git", "clone", settings.litepaper_src_dir, str(repo_dir)], check=True)
            else:
                print(f"Updating existing repository at {repo_dir}")
                update_repository(repo_dir)
            settings.litepaper_src_dir = str(repo_dir)
        
        # Initialize embedding model
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )

        # Load vector store
        print(f"Loading vector store from: {settings.vector_db_dir}")
        self.vector_store = Chroma(
            persist_directory=settings.vector_db_dir,
            embedding_function=self.embedding_function
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={'k': settings.search_k}
        )
        print("Vector store loaded.")

        # Initialize LLM
        print(f"Initializing LLM: {settings.openai_chat_model}")
        self.llm = ChatOpenAI(
            model_name=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1
        )

        # Create prompt template
        template = """You are an assistant for question-answering tasks based ONLY on the provided context.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Keep the answer concise and relevant to the question.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        self.prompt = ChatPromptTemplate.from_template(template)

        # Create RAG chain
        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        print("Creating RAG chain...")
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Chain with source tracking
        self.rag_chain_with_sources = RunnablePassthrough.assign(
            context=(lambda x: x['question']) | self.retriever
        ).assign(
            answer=(lambda x: {"context": format_docs(x['context']), "question": x['question']})
                   | self.prompt
                   | self.llm
                   | StrOutputParser()
        )
        print("RAGService Initialized.")

    async def get_answer(self, query: str) -> dict:
        print(f"Processing query: {query}")
        result = await self.rag_chain_with_sources.ainvoke({"question": query})

        sources_data = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result.get('context', [])
        ]

        print(f"Generated answer: {result.get('answer', 'No answer found')}")
        return {"answer": result.get('answer', 'Sorry, I could not find an answer.'), "sources": sources_data}

# Create a single instance
rag_service = RAGService()

async def get_rag_service():
    return rag_service 