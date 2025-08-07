# Multi-Face Video Classifier

A robust binary classifier that analyzes videos to detect the presence of multiple live faces, distinguishing between faces on identity documents and live faces in the scene.

## Overview

This classifier analyzes video content and returns:
- **Class 0**: Single or no live faces detected (faces on ID cards are filtered out)
- **Class 1**: Multiple live faces detected simultaneously

![Process Flow](assets/multiple-dark.jpg)

The system uses advanced face detection (SCRFD), object detection for ID cards, and face recognition embeddings to accurately classify videos while filtering out faces appearing on identity documents.

## Method Evaluation Results

Short Report can be found here: [Google docs](https://docs.google.com/document/d/1XB7ve1DSaOkBgEwqTEHAK--lw9-cPpKEtHt0TXbcgak/edit?tab=t.0)
or on `assets/multiple_face_detection_report.pdf`

| Method Name                                                   | Main Benchmark (Veriff, 19 videos)     |                                      | Additional Edge Cases (4 videos)     |                                      |
|---------------------------------------------------------------|----------------------------------------|--------------------------------------|----------------------------------------|--------------------------------------|
|                                                               | **Precision**                          | **Recall**                           | **Precision**                          | **Recall**                           |
| RC1 - Baseline                                                | 58.30%                                 | 100%                                 | -                                      | -                                    |
| RC2v1 - One-shot Clip                                         | -                                      | -                                    | -                                      | -                                    |
| RC2v2 - One-shot Clip (face removed)                          | -                                      | -                                    | -                                      | -                                    |
| RC3 - Grounding DINO                                          | 75.00%                                 | 42.90%                               | -                                      | -                                    |
| RC4 - Grounding DINO + ID Card Classification                | 87.50%                                 | 100%                                 | -                                      | -                                    |
| RC5 - RC4 + Additional Text Query                             | **100%**                               | **100%**                             | **0%**                                 | **0%**                               |
| RC6 - RC5 + Face Recognition + Face Occlusion Check           | **100%**                               | **100%**                             | **50%**                                | **33.30%**                           |



## Features

- **Advanced Face Detection**: Uses InsightFace SCRFD models with ONNX Runtime
- **ID Card Filtering**: Automatically detects and filters faces on identity documents  
- **Face Recognition**: Employs embedding similarity to identify unique individuals
- **GPU Acceleration**: CUDA support for faster processing
- **Flexible Input**: Supports single videos or directory batch processing

## Installation

### Method 1: Virtual Environment Setup

This method provides full control over the Python environment and is ideal for development work.

#### Prerequisites

- **Python 3.8-3.11** (Python 3.10 recommended)
- **pip** package manager
- **Git** for repository cloning
- **CUDA toolkit** (optional, for GPU acceleration)

#### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multi-face-video-classifier
   ```

2. **Create virtual environment**
   ```bash
   # Use the provided setup script
   ./setup_venv.sh  # Linux/Mac only
   
   # Alternative: setup by yourself
   python3 -m venv venv/multi-face-classifier
   pip3 install -r requirements.txt
   ```

3. **Activate virtual environment**
   ```bash
   # Linux/Mac
   source venv/multi-face-classifier/bin/activate
   
   ```
   **Verification**: Your prompt should show `(multi-face-classifier)` prefix when activated.

4. **Upgrade pip and install dependencies**
   ```bash
   # Upgrade pip to latest version
   pip install --upgrade pip
   
   # Install core dependencies
   pip install -r requirements.txt
   
   # For GPU support (optional)
   pip install -r requirements.gpu.txt
   ```

#### GPU Setup (Optional)

For NVIDIA GPU acceleration:

1. **Install NVIDIA drivers** and **CUDA toolkit 12.1+**
2. **Verify GPU availability**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
3. **Use GPU requirements**:
   ```bash
   pip install -r requirements.gpu.txt
   ```

#### Virtual Environment Management

**Daily usage workflow**:
```bash
# Activate environment (do this every time)
source venv/multi-face-classifier/bin/activate

# Run your processing
python bin/run_multi_face.py input/my_videos/ --output output/

# Deactivate when done
deactivate
```

**Environment maintenance**:
```bash
# Update packages
pip install --upgrade -r requirements.txt

# Check installed packages
pip list

# Save current environment
pip freeze > requirements-current.txt

# Remove environment (if needed)
rm -rf venv/multi-face-classifier
```

#### Troubleshooting Virtual Environment

**Common Issues**:

1. **Permission errors**:
   ```bash
   # Linux/Mac: Fix permissions
   chmod +x setup_venv.sh
   
   # Windows: Run as administrator
   ```

2. **Python version conflicts**:
   ```bash
   # Specify Python version explicitly
   python3.10 -m venv venv/multi-face-classifier
   ```

3. **Package installation failures**:
   ```bash
   # Clear pip cache
   pip cache purge
   
   # Install with no cache
   pip install --no-cache-dir -r requirements.txt
   ```

4. **CUDA/GPU issues**:
   ```bash
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   
   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

#### Performance Tips

- **Use SSD storage** for faster model loading
- **Allocate 16GB+ RAM** for large batch processing
- **Monitor GPU memory** usage during processing
- **Use CPU fallback** (`--device -1`) if GPU memory insufficient

### Method 2: Docker Setup (Recommended)

For containerized deployment with automatic GPU/CPU detection:

1. **Build Docker image**
   ```bash
   ./build_docker.sh
   ```
   The script automatically detects your system and builds the appropriate image:
   - **GPU system**: Builds optimized GPU image with CUDA support
   - **CPU system**: Builds lightweight CPU-only image

2. **Run with Docker**
   ```bash
   # Process all videos in a folder (recommended)
   ./run_docker.sh folder_name
   
   # Examples:
   ./run_docker.sh additional_vid    # Process input/additional_vid/
   ./run_docker.sh veriff_videos     # Process input/veriff_videos/
   ```

## Usage

### Using Virtual Environment

When using the virtual environment setup, always activate it before running commands:

```bash
# Activate virtual environment
source venv/multi-face-classifier/bin/activate

# Verify activation (should show environment name in prompt)
which python  # Should point to venv/multi-face-classifier/bin/python
```

**Note**: Remember to activate the virtual environment in each new terminal session.

### Basic Usage

**Single Video:**
```bash
python bin/run_multi_face.py /path/to/video.mp4 --output /path/to/output_folder/
```
**Batch Processing (Recommended):**
```bash
python bin/run_multi_face.py /path/to/video/folder/ --output /path/to/output_folder
```
**Batch Processing test on old version:**
```bash
python bin/run_multi_face.py /path/to/video/folder/ --output /path/to/output_folder_rc5 --dev --mode rc5
```

The script automatically detects if the input is a file or directory and processes accordingly.
it will also save a json file of each video, which contains the basic info and classification results

## Evaluation

To evaluate model performance with precision, recall, accuracy, FAR, and FRR:

```bash
# Basic evaluation (uses default labels and results)
cd tools/
python evaluate_model_performance.py

# Custom evaluation
python evaluate_model_performance.py --labels-file /path/to/labels.txt --predictions-dir /path/to/results/
```

**Label file format (TSV):**
```
video	label
video1	0
video2	1
```

**Available datasets:**
- Main benchmark: `assets/labels/labels.txt` (19 videos)
- Edge cases: `assets/labels/labels_edge.txt` (4 videos)

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
    --max-frames 50 \ 
    
# best value is 50

# Results will be saved as:
# /path/to/results/video1_out.json
# /path/to/results/video2_out.json
# ...
```

### Development and Testing

```bash
# Quick test with fewer frames
python bin/run_multi_face.py test_video.mp4 --max-frames 20

# CPU testing (no GPU required)
python bin/run_multi_face.py test_video.mp4 --device -1

# Create visualization for analysis (don't use this, currently will make the folder messy, require a lot of fixing)
python bin/run_multi_face.py test_video.mp4 --visualize 
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

## Docker Usage Guide

### Quick Start

The Docker setup provides automated GPU/CPU detection and simplified folder-based processing:

```bash
# 1. Build the appropriate Docker image for your system
./build_docker.sh

# 2. Place videos in input folders
mkdir -p input/my_videos
cp /path/to/videos/* input/my_videos/

# 3. Process all videos in the folder
./run_docker.sh my_videos
```

### Automated Build Script

The `build_docker.sh` script automatically:
- **Detects GPU availability** and Docker NVIDIA runtime support
- **Builds GPU image** (`multi-face-classifier-gpu`) if GPU detected
- **Builds CPU image** (`multi-face-classifier`) if no GPU available
- **Uses optimized Dockerfiles** for each configuration

```bash
./build_docker.sh
```

**Example Output:**
```
Multi-Face Video Classifier - Docker Build Script
==================================================
✓ NVIDIA GPU detected and Docker GPU support available
Building GPU-optimized Docker image...
✓ Successfully built multi-face-classifier-gpu
Image: multi-face-classifier-gpu:latest

Build completed successfully!
Use ./run_docker.sh <folder_path> to run the classifier
```

### Automated Run Script

The `run_docker.sh` script provides intelligent processing with folder-based input:

#### Basic Usage
```bash
# Process all videos in input/folder_name/
./run_docker.sh <folder_name>

# Examples:
./run_docker.sh additional_vid     # Process input/additional_vid/
./run_docker.sh test_videos       # Process input/test_videos/
./run_docker.sh veriff_dataset    # Process input/veriff_dataset/
```

#### Advanced Options
```bash
# Force CPU processing
./run_docker.sh my_videos --cpu

# Force GPU processing  
./run_docker.sh my_videos --gpu

# Specific GPU device
./run_docker.sh my_videos --device 1

# Enable verbose logging
./run_docker.sh my_videos --verbose
```

### Directory Structure

Organize your videos using the following structure:

```
multi-face-video-classifier/
├── input/                          # Input folder for all video datasets
│   ├── additional_vid/             # Folder 1: Edge case videos
│   │   ├── edge1.mp4
│   │   ├── edge2.mp4
│   │   └── edge3.mp4
│   ├── veriff_videos/              # Folder 2: Main dataset
│   │   ├── veriff1.mp4
│   │   ├── veriff2.mp4
│   │   └── ...
│   └── test_videos/                # Folder 3: Test videos
│       ├── sample1.mp4
│       └── sample2.mp4
├── output/                         # Results automatically saved here
│   ├── edge1_out.json
│   ├── edge2_out.json
│   ├── veriff1_out.json
│   └── ...
├── build_docker.sh                 # Automated build script
└── run_docker.sh                   # Automated run script
```

### Processing Examples

#### Example 1: Process Edge Cases
```bash
# Copy videos to input folder
mkdir -p input/edge_cases
cp /path/to/edge_videos/* input/edge_cases/

# Process with GPU acceleration
./run_docker.sh edge_cases

# Results saved to output/ folder:
# - output/edge1_out.json
# - output/edge2_out.json
# - ...
```

#### Example 2: Process Large Dataset on CPU
```bash
# Copy dataset
mkdir -p input/large_dataset
cp /path/to/videos/* input/large_dataset/

# Force CPU processing (useful for servers without GPU)
./run_docker.sh large_dataset --cpu --verbose

# Monitor progress with verbose output
```

#### Example 3: Multi-GPU Server
```bash
# Process on specific GPU device
./run_docker.sh dataset1 --device 0
./run_docker.sh dataset2 --device 1

# Or let the script auto-detect best GPU
./run_docker.sh dataset1 --gpu
```

### Docker Image Selection Logic

The run script intelligently selects images based on your requirements:

1. **Auto-detection mode** (default):
   - Detects GPU availability
   - Uses GPU image if available
   - Falls back to CPU image if needed

2. **Forced GPU mode** (`--gpu`):
   - Prefers GPU-optimized image
   - Falls back to CPU image with warning if GPU image unavailable
   - Uses NVIDIA Docker runtime (`--runtime=nvidia`)

3. **Forced CPU mode** (`--cpu`):
   - Always uses CPU image
   - No GPU runtime flags

### Performance Comparison

| Configuration | Image | Runtime | Speed | Memory |
|--------------|-------|---------|-------|---------|
| GPU Mode | `multi-face-classifier-gpu` | `nvidia` | ~4x faster | Higher VRAM |
| CPU Mode | `multi-face-classifier` | `runc` | Baseline | Lower RAM |

### Supported Video Formats

The Docker setup automatically detects these video formats:
- `.mp4` (recommended)
- `.avi`
- `.mov` 
- `.mkv`

### Troubleshooting Docker

#### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Check Docker NVIDIA runtime
docker info | grep nvidia

# Force rebuild if needed
docker rmi multi-face-classifier-gpu
./build_docker.sh
```

#### No Videos Found Error
```bash
# Error: No video files found in ./input/folder_name
# Solution: Check folder exists and contains supported video files
ls -la input/folder_name/
```

#### Permission Issues
```bash
# Fix permissions for input/output folders
chmod -R 755 input/
chmod -R 755 output/
```

### Manual Docker Commands (Advanced)

If you prefer manual control over the Docker setup:

```bash
# Manual GPU build
docker build -f Dockerfile.gpu -t multi-face-classifier-gpu .

# Manual CPU build  
docker build -f Dockerfile -t multi-face-classifier .

# Manual run with GPU
docker run --rm --runtime=nvidia \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  multi-face-classifier-gpu \
  python bin/run_multi_face.py /app/input/folder_name --output /app/output --device 0

# Manual run with CPU
docker run --rm \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  multi-face-classifier \
  python bin/run_multi_face.py /app/input/folder_name --output /app/output --device -1
```

## Project Structure

```
multi-face-video-classifier/
├── assets/                              # Documentation and evaluation assets
│   ├── Veriff - Multiple Face Classifier Report.pdf  # Detailed evaluation report
│   ├── labels/                          # Ground truth labels for evaluation
│   │   ├── labels.txt                   # Main benchmark dataset labels
│   │   └── labels_edge.txt              # Edge case dataset labels
│   └── multiple-dark.jpg                # Process flow visualization
├── bin/
│   └── run_multi_face.py                # Main executable script
├── src/
│   └── multi_face_video_classifier/
│       ├── video_pipeline.py            # Core classification logic (RC6)
│       ├── video_pipeline_dev.py        # Development pipeline (RC1-RC5)
│       ├── card_detector.py             # ID card detection using Grounding DINO
│       └── utils/                       # Utility modules
│           ├── face_tracking_utils.py   # Face tracking and recognition
│           ├── pose_estimator.py        # Face pose estimation
│           └── visualization_utils.py   # Visualization helpers
├── tools/
│   └── evaluate_model_performance.py    # Evaluation script with metrics
├── deprecated_script/                   # Legacy code and experiments
├── input/                              # Input video directories
├── venv/                               # Virtual environment
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── docker-compose.yml                 # Docker Compose setup
├── build_docker.sh                    # Docker build script
├── run_docker.sh                      # Docker run script
├── setup_venv.sh                      # Virtual environment setup
└── README.md                          # This documentation
```

### Advanced Options (not recommended to use, might have certain bugs and issues)

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

## License

This project is for internal use and contains proprietary algorithms for face detection and recognition.