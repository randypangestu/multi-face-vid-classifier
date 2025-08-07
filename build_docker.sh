#!/bin/bash

# Script to automatically build Docker image based on GPU availability

set -e

echo "Multi-Face Video Classifier - Docker Build Script"
echo "=================================================="

# Function to check if NVIDIA GPU is available
check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to check if Docker supports NVIDIA runtime
check_docker_gpu() {
    if docker info 2>/dev/null | grep -q "nvidia"; then
        return 0
    fi
    return 1
}

# Detect GPU availability
if check_gpu && check_docker_gpu; then
    echo "✓ NVIDIA GPU detected and Docker GPU support available"
    echo "Building GPU-optimized Docker image..."
    
    # Build GPU version
    docker build -f Dockerfile.gpu -t multi-face-classifier-gpu .
    
    echo "✓ Successfully built multi-face-classifier-gpu"
    echo "Image: multi-face-classifier-gpu:latest"
    
else
    echo "⚠ No GPU detected or Docker GPU support unavailable"
    echo "Building CPU-only Docker image..."
    
    # Build CPU version  
    docker build -f Dockerfile -t multi-face-classifier .
    
    echo "✓ Successfully built multi-face-classifier"
    echo "Image: multi-face-classifier:latest"
fi

echo ""
echo "Build completed successfully!"
echo "Use ./run_docker.sh <folder_path> to run the classifier"