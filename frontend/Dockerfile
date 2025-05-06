# Use an official Node.js runtime as a parent image
# Use a version compatible with your Next.js version (e.g., 18 or 20)
FROM node:20-alpine AS base

# Set the working directory
WORKDIR /app

# Install dependencies first, in a separate step to take advantage of Docker's layer caching.
# Copy package.json and package-lock.json (or yarn.lock)
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend application code
COPY frontend/ .

# Build the Next.js application for production
RUN npm run build

ARG NEXT_PUBLIC_API_URL
# Define environment variable for the API URL
# This will be overridden by docker-compose to point to the backend service
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
# Add any other public ENV vars needed by the frontend build or runtime

# Expose port 3000
EXPOSE 3000

# Set the command to start the production server
# Next.js automatically looks for the built app in .next
CMD ["npm", "start"] 