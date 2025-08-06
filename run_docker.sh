#!/bin/bash

# Run script for Multi-Face Video Classifier Docker container
set -e

# Default values
IMAGE_NAME="multi-face-video-classifier:latest"
INPUT_DIR="./input"
OUTPUT_DIR="./output"
DEFAULT_MODE="rc6"
DEFAULT_DEVICE="0"
DEFAULT_MAX_FRAMES="50"

# Help function
show_help() {
    echo "Usage: $0 <video_path> [OPTIONS]"
    echo ""
    echo "Run Multi-Face Video Classifier in Docker container"
    echo ""
    echo "Arguments:"
    echo "  video_path              Path to video file (relative to input directory)"
    echo ""
    echo "Options:"
    echo "  --mode MODE            Classification mode (rc1, rc2v1, rc2v2, rc3, rc4, rc5, rc6) [default: $DEFAULT_MODE]"
    echo "  --device DEVICE        GPU device ID (-1 for CPU) [default: $DEFAULT_DEVICE]"
    echo "  --max-frames FRAMES    Maximum frames to process [default: $DEFAULT_MAX_FRAMES]"
    echo "  --frame-skip SKIP      Process every nth frame [default: 1]"
    echo "  --output OUTPUT        Output JSON file path [default: auto-generated]"
    echo "  --visualize            Create visualization video"
    echo "  --output-video PATH    Output path for visualization video"
    echo "  --verbose              Enable verbose logging"
    echo "  --cpu                  Force CPU inference (equivalent to --device -1)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 sample_video.mp4"
    echo "  $0 sample_video.mp4 --mode rc6 --device 0 --visualize"
    echo "  $0 sample_video.mp4 --cpu --verbose"
    echo ""
    echo "Note: Place your video files in the './input' directory"
    echo "      Results will be saved to the './output' directory"
}

# Parse command line arguments
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

VIDEO_PATH="$1"
shift

# Create input and output directories if they don't exist
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Check if video file exists
if [ ! -f "$INPUT_DIR/$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $INPUT_DIR/$VIDEO_PATH"
    echo "Please place your video file in the '$INPUT_DIR' directory"
    exit 1
fi

# Build Docker run command
DOCKER_ARGS=()
SCRIPT_ARGS=("/app/input/$VIDEO_PATH")

# Parse options
while [ $# -gt 0 ]; do
    case $1 in
        --mode)
            SCRIPT_ARGS+=("--mode" "$2")
            shift 2
            ;;
        --device)
            SCRIPT_ARGS+=("--device" "$2")
            shift 2
            ;;

        --dev)
            SCRIPT_ARGS+=("--dev")
            shift
            ;;
        --max-frames)
            SCRIPT_ARGS+=("--max-frames" "$2")
            shift 2
            ;;
        --frame-skip)
            SCRIPT_ARGS+=("--frame-skip" "$2")
            shift 2
            ;;
        --output)
            SCRIPT_ARGS+=("--output" "/app/output/$2")
            shift 2
            ;;
        --output-video)
            SCRIPT_ARGS+=("--output-video" "/app/output/$2")
            shift 2
            ;;
        --visualize)
            SCRIPT_ARGS+=("--visualize")
            shift
            ;;
        --verbose)
            SCRIPT_ARGS+=("--verbose")
            shift
            ;;
        --cpu)
            SCRIPT_ARGS+=("--device" "-1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if Docker image exists, load or build if not
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    IMAGE_FILE="../docker-images/multi-face-video-classifier.tar"
    
    # Try to load from saved file first
    if [ -f "$IMAGE_FILE" ]; then
        echo "Docker image '$IMAGE_NAME' not found locally. Loading from $IMAGE_FILE..."
        if docker load -i "$IMAGE_FILE"; then
            echo "✅ Docker image loaded successfully!"
            echo ""
        else
            echo "❌ Failed to load image from file. Building new image..."
            if ! ./build_docker.sh; then
                echo "Error: Failed to build Docker image"
                echo ""
                echo "Troubleshooting:"
                echo "1. Ensure you have at least 10GB free disk space"
                echo "2. Try running: docker system prune -a"
                echo "3. Run ./build_docker.sh manually to see detailed errors"
                exit 1
            fi
        fi
    else
        echo "Docker image '$IMAGE_NAME' not found. Building..."
        if ! ./build_docker.sh; then
            echo "Error: Failed to build Docker image"
            echo ""
            echo "Troubleshooting:"
            echo "1. Ensure you have at least 10GB free disk space"
            echo "2. Try running: docker system prune -a"
            echo "3. Run ./build_docker.sh manually to see detailed errors"
            exit 1
        fi
    fi
fi

# Check if NVIDIA Docker runtime is available
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
elif docker info | grep -q "nvidia"; then
    DOCKER_CMD="docker"
    DOCKER_ARGS+=("--gpus" "all")
else
    DOCKER_CMD="docker"
    echo "Warning: NVIDIA Docker runtime not detected. Using CPU inference."
    # Override device to CPU if no GPU support
    for i in "${!SCRIPT_ARGS[@]}"; do
        if [[ "${SCRIPT_ARGS[$i]}" == "--device" ]] && [[ "${SCRIPT_ARGS[$((i+1))]}" != "-1" ]]; then
            SCRIPT_ARGS[$((i+1))]="-1"
            echo "Switched to CPU inference due to no GPU support"
            break
        fi
    done
fi

echo "Running Multi-Face Video Classifier..."
echo "Input video: $INPUT_DIR/$VIDEO_PATH"
echo "Docker command: $DOCKER_CMD"
echo ""

# Run the container
$DOCKER_CMD run --rm \
    "${DOCKER_ARGS[@]}" \
    -v "$(pwd)/$INPUT_DIR:/app/input:ro" \
    -v "$(pwd)/$OUTPUT_DIR:/app/output" \
    "$IMAGE_NAME" \
    python bin/run_multi_face.py "${SCRIPT_ARGS[@]}"

echo ""
echo "✅ Processing completed!"
echo "Check the '$OUTPUT_DIR' directory for results."