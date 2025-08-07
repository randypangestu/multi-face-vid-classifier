#!/bin/bash

# Run script for Multi-Face Video Classifier Docker container
set -e

# Default values
IMAGE_NAME="multi-face-classifier"
OUTPUT_DIR="./output"

# Help function
show_help() {
    echo "Usage: $0 <folder_path> [OPTIONS]"
    echo ""
    echo "Run Multi-Face Video Classifier on all videos in a folder"
    echo ""
    echo "Arguments:"
    echo "  folder_path            Path to folder containing videos (relative to input/)"
    echo ""
    echo "Options:"
    echo "  --device DEVICE        GPU device ID (-1 for CPU) [default: auto-detect]"
    echo "  --cpu                  Force CPU inference"
    echo "  --gpu                  Force GPU inference"
    echo "  --verbose              Enable verbose logging"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 additional_vid      # Process all videos in input/additional_vid/"
    echo "  $0 veriff_videos --cpu # Process input/veriff_videos/ with CPU"
    echo "  $0 test_folder --gpu   # Process input/test_folder/ with GPU"
    echo ""
    echo "Note: Videos should be placed in input/<folder_path>/"
    echo "      Results will be saved to the './output' directory"
}

# Parse command line arguments
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

FOLDER_PATH="$1"
shift

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Default device selection
DEVICE_MODE="auto"
VERBOSE=false

# Parse options
while [ $# -gt 0 ]; do
    case $1 in
        --device)
            DEVICE_MODE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE_MODE="cpu"
            shift
            ;;
        --gpu)
            DEVICE_MODE="gpu"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if input folder exists
INPUT_FOLDER="./input/$FOLDER_PATH"
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder not found: $INPUT_FOLDER"
    echo "Please create the folder and place your video files there"
    exit 1
fi

# Check if there are video files in the folder
VIDEO_COUNT=$(find "$INPUT_FOLDER" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "Error: No video files found in $INPUT_FOLDER"
    echo "Supported formats: mp4, avi, mov, mkv"
    exit 1
fi

echo "Found $VIDEO_COUNT video file(s) in $INPUT_FOLDER"

# Determine device and image selection
DOCKER_ARGS=()
DEVICE_ARG="-1"  # Default to CPU

case $DEVICE_MODE in
    "auto")
        if check_gpu; then
            echo "Auto-detected GPU, using GPU acceleration"
            DEVICE_ARG="0"
        else
            echo "No GPU detected, using CPU"
            DEVICE_ARG="-1"
        fi
        ;;
    "gpu")
        if check_gpu; then
            echo "Forced GPU mode"
            DEVICE_ARG="0"
        else
            echo "Warning: GPU requested but not available, falling back to CPU"
            DEVICE_ARG="-1"
        fi
        ;;
    "cpu")
        echo "Forced CPU mode"
        DEVICE_ARG="-1"
        ;;
    *)
        # Numeric device ID
        if [[ "$DEVICE_MODE" =~ ^[0-9]+$ ]]; then
            DEVICE_ARG="$DEVICE_MODE"
            echo "Using GPU device: $DEVICE_ARG"
        else
            echo "Invalid device: $DEVICE_MODE"
            exit 1
        fi
        ;;
esac

# Select appropriate Docker image and runtime
if [ "$DEVICE_ARG" == "-1" ]; then
    # CPU mode
    if docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo "Using CPU image: $IMAGE_NAME"
    else
        echo "Error: CPU Docker image '$IMAGE_NAME' not found."
        echo "Please run: ./build_docker.sh"
        exit 1
    fi
else
    # GPU mode - prefer GPU image, fallback to CPU image
    if docker image inspect "multi-face-classifier-gpu" &> /dev/null; then
        IMAGE_NAME="multi-face-classifier-gpu"
        DOCKER_ARGS+=("--runtime=nvidia")
        echo "Using GPU image: $IMAGE_NAME"
    elif docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo "GPU requested but GPU image not available, using CPU image: $IMAGE_NAME"
        echo "Warning: This may not work optimally for GPU inference"
    else
        echo "Error: No Docker image found."
        echo "Please run: ./build_docker.sh"
        exit 1
    fi
fi

# Prepare script arguments
SCRIPT_ARGS=("/app/input/$FOLDER_PATH" "--output" "/app/output" "--device" "$DEVICE_ARG")

if $VERBOSE; then
    SCRIPT_ARGS+=("--verbose")
fi

echo ""
echo "Running Multi-Face Video Classifier..."
echo "Input folder: $INPUT_FOLDER"
echo "Processing $VIDEO_COUNT video file(s)"
echo "Device: $DEVICE_ARG"
echo ""

# Run the container
docker run --rm \
    "${DOCKER_ARGS[@]}" \
    -v "$(pwd)/input:/app/input:ro" \
    -v "$(pwd)/$OUTPUT_DIR:/app/output" \
    "$IMAGE_NAME" \
    python bin/run_multi_face.py "${SCRIPT_ARGS[@]}"

echo ""
echo "✅ Processing completed!"
echo "Check the '$OUTPUT_DIR' directory for results."