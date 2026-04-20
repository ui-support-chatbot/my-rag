#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found."
    exit 1
fi

# Basic validation
if [ -z "$SERVER_HOST" ] || [ -z "$SERVER_USER" ] || [ -z "$PROJECT_PATH" ]; then
    echo "Error: Missing required variables (SERVER_HOST, SERVER_USER, or PROJECT_PATH) in .env"
    exit 1
fi

# Connect and run commands
echo "Connecting to $SERVER_USER@$SERVER_HOST..."
ssh "$SERVER_USER@$SERVER_HOST" << 'EOF'
set -e
echo "Navigating to $PROJECT_PATH..."
cd "$PROJECT_PATH" || { echo "Error: Directory $PROJECT_PATH does not exist."; exit 1; }

echo "--- Docker Status ---"
docker compose ps

echo "--- GPU Status ---"
nvidia-smi

echo "--- System Load (Top) ---"
top -bn1 | head -20

echo "--- Last 50 lines of rag-api logs ---"
docker compose logs --tail=50 rag-api
EOF
