# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install git (required by LangChain GitLoader)
# Also install build-essential for packages that need compilation
# Add clang compiler
RUN apt-get update && \
    apt-get install -y git build-essential clang --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
# Note: Context is project root, so paths are relative to that
COPY ./backend/requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir --upgrade pip
# Install wheel first to ensure packages using setup.py can build wheels
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

# Copy the rest of the backend application code into the container at /app
COPY ./backend /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables (can be overridden by docker-compose)
# Ensure your actual keys/paths are provided via .env file and docker-compose
# Example:
# ENV OPENAI_API_KEY=your_openai_api_key_here
# ENV LITEPAPER_SRC_DIR=https://github.com/ChaosChain/chaoschain-lite-paper.git
# ENV VECTOR_DB_DIR=/app/data/vector_store
# ENV CORS_ORIGINS=http://localhost:3000,http://frontend:3000
# These are best set via the .env file and docker-compose env_file directive

# Create the data directory if it potentially doesn't exist (volume mounting recommended)
RUN mkdir -p /app/data/vector_store

# Run uvicorn server when the container launches
# Use 0.0.0.0 to listen on all interfaces within the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 