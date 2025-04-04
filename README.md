# Chaos Chain Litepaper RAG Chat

A chat interface allowing users to ask questions about the ChaosChain Litepaper, powered by Retrieval-Augmented Generation (RAG).

This project consists of:

*   **Backend:** A FastAPI application using LangChain to process documents, create embeddings, store them in ChromaDB, and generate answers using an OpenAI model (like GPT-4o-mini). It provides a streaming API compatible with the Vercel AI SDK.
*   **Frontend:** A Next.js application using the `assistant-ui` library for the chat interface, consuming the backend's streaming API.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd litepaper-chat
    ```
2.  **Backend Setup:**
    *   Navigate to `backend/`.
    *   Create a Python virtual environment: `python -m venv venv`
    *   Activate it: `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate.ps1` (Windows PowerShell)
    *   Install dependencies: `pip install -r requirements.txt` (You might need to create requirements.txt first: `pip freeze > requirements.txt`)
3.  **Frontend Setup:**
    *   Navigate to `frontend/`.
    *   Install dependencies: `npm install`
4.  **Environment Variables:**
    *   Create a `.env` file in the project root.
    *   Copy contents from `.env.example` and fill in your `OPENAI_API_KEY`.
    *   Ensure `LITEPAPER_SRC_DIR` points to the correct litepaper source (e.g., the GitHub repo URL).
    *   Update `CORS_ORIGINS` if your frontend runs on a different port during development or after deployment.

## Running Locally

1.  **Start Backend:**
    ```bash
    cd backend
    # Activate venv if needed
    python -m uvicorn app.main:app --reload --port 8000
    ```
    *The first time, it will clone the repo and build the vector store.*

2.  **Start Frontend:**
    ```bash
    cd frontend
    npm run dev
    ```
    *Access the chat at http://localhost:3000 (or the port Next.js indicates).*

## Deployment

*   **Frontend:** Deploy the `frontend` directory as a Next.js application (e.g., on Vercel). Set the `NEXT_PUBLIC_API_URL` environment variable to the deployed backend URL.
*   **Backend:** Deploy the `backend` directory as a Python application (e.g., on Render, Fly.io). Set environment variables (`OPENAI_API_KEY`, `CORS_ORIGINS`, etc.) on the hosting platform.

## Project Structure

```
litepaper-chat/
├── backend/                 # FastAPI backend
│   ├── app/                # Main application code
│   ├── scripts/            # Data ingestion scripts
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   ├── src/               # Source code
│   └── public/            # Static assets
└── data/                  # Data directory
    └── vector_store/      # ChromaDB vector store
```

## Prerequisites

- Python 3.9+
- Node.js 18+
- Git

## Development

- Backend API documentation: http://localhost:8000/docs
- Frontend development server: http://localhost:5173

## License

MIT