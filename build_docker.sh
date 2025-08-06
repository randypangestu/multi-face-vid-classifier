#!/bin/bash

# Build script for Multi-Face Video Classifier Docker image
set -e

IMAGE_NAME="multi-face-video-classifier:latest"

echo "Building Docker image: $IMAGE_NAME"
echo "This may take 10-15 minutes due to large dependencies..."
echo ""

# Build the image
docker build -t "$IMAGE_NAME" .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully: $IMAGE_NAME"
    echo ""
    echo "You can now run the container with:"
    echo "  docker run --rm -it $IMAGE_NAME"
else
    echo ""
    echo "❌ Failed to build Docker image"
    exit 1
fi