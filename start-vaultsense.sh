#!/bin/bash
# Install python dependencies
pip install -r ai/requirements.txt

# Find the local machine IP address
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}')
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32" ]]; then
    # Windows with Git Bash or similar
    LOCAL_IP=$(ipconfig | grep -i "IPv4 Address" | head -n 1 | awk -F ": " '{print $2}' | tr -d '\r\n')
else
    echo "Unsupported operating system. Please manually set your IP address in docker-compose.yaml."
    exit 1
fi

if [ -z "$LOCAL_IP" ]; then
    echo "Could not determine local IP address. Please manually set your IP address in docker-compose.yaml."
    exit 1
fi

echo "Detected local IP address: $LOCAL_IP"

# Create temporary docker-compose file and replace the MILVUS_URL line
# This approach works on both macOS and Linux
sed "s/- MILVUS_URL=.*$/- MILVUS_URL=$LOCAL_IP:19530/g" docker-compose.yaml > docker-compose.yaml.new
mv docker-compose.yaml.new docker-compose.yaml

echo "Updated docker-compose.yaml with your local IP address ($LOCAL_IP)"

# Install sqlite for mac if not installed
if ! command -v sqlite3 &> /dev/null; then
    echo "sqlite3 could not be found, installing..."
    brew install sqlite
fi

# Start local SQLite database
sqlite3 ./sqlite/vaultsense.db < ./sqlite/vaultsense.sql

# Start the docker containers
echo "Starting VaultSense services..."
docker-compose up -d


# Create the following collections in milvus by running the following python script
echo "Creating Milvus collections..."
python scripts/create_milvus_collections.py

echo "VaultSense services started successfully!"
echo "You can access the Attu UI at: http://localhost:8001" 