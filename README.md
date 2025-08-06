# Multi-Face Video Classifier

A robust binary classifier that analyzes videos to detect the presence of multiple live faces, distinguishing between faces on identity documents and live faces in the scene.

## Overview

This classifier analyzes video content and returns:
- **Class 0**: Single or no live faces detected (faces on ID cards are filtered out)
- **Class 1**: Multiple live faces detected simultaneously

The system uses advanced face detection (SCRFD), object detection for ID cards, and face recognition embeddings to accurately classify videos while filtering out faces appearing on identity documents.

## Features

- **Advanced Face Detection**: Uses InsightFace SCRFD models with ONNX Runtime
- **ID Card Filtering**: Automatically detects and filters faces on identity documents  
- **Face Recognition**: Employs embedding similarity to identify unique individuals
- **Batch Processing**: Process entire folders of videos efficiently
- **GPU Acceleration**: CUDA support for faster processing
- **Flexible Input**: Supports single videos or directory batch processing

## Installation

### Method 1: Virtual Environment (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multi-face-video-classifier
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv/face-multi-det2
   source venv/face-multi-det2/bin/activate  # Linux/Mac
   # or
   venv\face-multi-det2\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Method 2: Docker Setup

For containerized deployment with GPU support:

1. **Build Docker image**
   ```bash
   ./build_docker.sh
   ```

2. **Run with Docker**
   ```bash
   # Single video
   ./run_docker.sh /path/to/video.mp4
   
   # Batch processing (recommended)
   ./run_docker.sh /path/to/video/folder/
   ```

## Usage

### Activate Virtual Environment (if using venv)

Before running any commands, activate the virtual environment:
```bash
source venv/multi-face-classifier/bin/activate
```

### Basic Usage

**Single Video:**
```bash
python bin/run_multi_face.py /path/to/video.mp4
```

**Batch Processing (Recommended):**
```bash
python bin/run_multi_face.py /path/to/video/folder/
```

The script automatically detects if the input is a file or directory and processes accordingly.

### Advanced Options

```bash
# Process with custom settings
python bin/run_multi_face.py /path/to/videos/ \
    --max-frames 50 \
    --device 0 \
    --output results/ \

# CPU-only processing
python bin/run_multi_face.py /path/to/videos/ --device -1


### Command Line Arguments

- `video_path`: Path to video file or directory containing videos
- `--model`: InsightFace model (`buffalo_l`, `buffalo_m`, `buffalo_s`) - default: `buffalo_l`
- `--mode`: Processing mode (`rc1` to `rc6`) - default: `rc6` (most advanced)
- `--max-frames`: Maximum frames to process per video - default: `50`
- `--frame-skip`: Process every nth frame - default: `1`
- `--det-size`: Detection input size [width height] - default: `640 640`
- `--device`: GPU device ID (`-1` for CPU) - default: `0`
- `--output`: Output directory for JSON results - default: same as input
- `--visualize`: Create visualization videos with detections
- `--output-video`: Custom path for visualization video
- `--verbose`: Enable detailed logging
- `--dev`: Use development pipeline (if available)

## Output Format

The classifier outputs JSON files with the following structure:

**Single Face Example (Class 0):**
```json
{
    "contain_multiple_live_faces": false,
    "n_faces": 1,
    "class": 0
}
```

**Multiple Faces Example (Class 1):**
```json
{
    "contain_multiple_live_faces": true,
    "n_faces": 2,
    "class": 1
}
```

### Output Files

For each processed video, the system generates:
- `{video_name}_out.json`: Classification results
- `{video_name}_detected.mp4`: Visualization video (if `--visualize` is used)

**Processing Modes: Only available with command `--dev`**
- `rc1`: Basic face detection
- `rc3`: Face detection with card counting
- `rc4`: Face detection with card filtering
- `rc5`: Advanced face detection with card filtering  
- `rc6`: Most advanced with face recognition (recommended + default)

## Example Workflows

### Batch Processing Multiple Videos

```bash
# Process all videos in a directory
python bin/run_multi_face.py /path/to/video/dataset/ \
    --output /path/to/results/ \
    --max-frames 200 \
    --verbose

# Results will be saved as:
# /path/to/results/video1_out.json
# /path/to/results/video2_out.json
# ...
```

### Development and Testing

```bash
# Quick test with fewer frames
python bin/run_multi_face.py test_video.mp4 --max-frames 20 --verbose

# CPU testing (no GPU required)
python bin/run_multi_face.py test_video.mp4 --device -1

# Create visualization for analysis
python bin/run_multi_face.py test_video.mp4 --visualize --verbose
```

## Hardware Requirements

**Recommended for Performance:**
- GPU: NVIDIA GPU with CUDA support
- VRAM: 4GB+ for optimal performance
- RAM: 16GB+ for large batch processing

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Install all dependencies: `pip install -r requirements.txt`

2. **CUDA/GPU Issues**
   - Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Use `--device -1` for CPU-only processing

3. **Memory Errors**
   - Reduce `--max-frames` parameter
   - Use `--frame-skip 2` or higher to process fewer frames

4. **Permission Errors**
   - Ensure read/write permissions for input/output directories
   - Check that virtual environment is properly activated

### Performance Optimization

- **Use folder input**: Batch processing is more efficient than individual files
- **GPU acceleration**: Use CUDA when available for 5-10x speedup
- **Frame optimization**: Adjust `--max-frames` and `--frame-skip` based on video length default (50 max frames)
- **Output location**: Use fast storage (SSD) for output directory

## Docker Usage

### Building the Container

```bash
# Build image
./build_docker.sh

# Or manually
docker build -t multi-face-video-classifier .
```

### Running with Docker

```bash
# Process single video
docker run --gpus all -v /path/to/videos:/input multi-face-video-classifier /input/video.mp4

# Process directory (recommended)
docker run --gpus all -v /path/to/videos:/input -v /path/to/output:/output multi-face-video-classifier /input/ --output /output/
```

## Project Structure

```
multi-face-video-classifier/
├── bin/
│   └── run_multi_face.py          # Main executable script
├── src/
│   └── multi_face_video_classifier/
│       ├── video_pipeline.py      # Core classification logic
│       ├── card_detector.py       # ID card detection
│       └── utils/                 # Utility modules
├── data/
│   └── output_test/              # Example outputs
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml          # Docker Compose setup
└── README.md                    # This file
```

## License

This project is for internal use and contains proprietary algorithms for face detection and recognition.