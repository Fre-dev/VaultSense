#!/bin/bash

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

# Start the docker containers
echo "Starting VaultSense services..."
docker-compose up -d

echo "VaultSense services started successfully!"
echo "You can access the Attu UI at: http://localhost:8001" 