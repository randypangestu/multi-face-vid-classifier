#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path
import torch

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Multi-Face Video Classifier, a binary classifier to detect if a video contains multiple faces."
                                "0: only one face or less throughout the video, 1: multiple faces throughout the video.")
    
    parser.add_argument("video_path", help="Path to the video file")
    
    parser.add_argument("--output", type=str, default="output/results/",
                       help="Output JSON file path")

    parser.add_argument("--model", default="buffalo_l",
                       choices=["buffalo_l", "buffalo_m", "buffalo_s"],
                       help="InsightFace model name (default: buffalo_l)")
    
    parser.add_argument("-d", "--dev", action="store_true", help="Use VideoClassifierDevelopment instead of VideoClassifier")
    
    parser.add_argument("--mode", default="rc6", choices=["rc1", "rc2v1", "rc2v2", "rc3", "rc4", "rc5", "rc6"],
                       help="Inference mode (default: rc6)")

    parser.add_argument("--max-frames", type=int, default=50,
                       help="Maximum number of frames to process (default: 50)")
    
    parser.add_argument("--frame-skip", type=int, default=1,
                       help="Process every nth frame (default: 1)")
    
    parser.add_argument("--det-size", nargs=2, type=int, default=[640, 640],
                       help="Detection input size [width height] (default: 640 640)")
    
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (use -1 for CPU, default: 0)")
    
    
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization video with detections")
    
    parser.add_argument("--output-video", type=str, default=None,
                       help="Output path for visualization video")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    args = get_args()
    if args.dev:
        logger.info("Using VideoClassifierDevelopment mode")
        from multi_face_video_classifier.video_pipeline_dev import VideoClassifierDev as VideoClassifier
    else:
        logger.info("Using VideoClassifier mode, will always run rc6 version")
        from multi_face_video_classifier.video_pipeline import VideoClassifier
    # Validate and get video files
    video_path = Path(args.video_path)
    if video_path.is_dir():
        # Process all video files in directory
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_path.glob(ext))
            video_files.extend(video_path.glob(ext.upper()))  # Also check uppercase extensions
        
        if not video_files:
            logger.error(f"No video files found in directory: {args.video_path}")
            logger.info("Supported formats: mp4, avi, mov, mkv, wmv")
            sys.exit(1)

        logger.info(f"Found {len(video_files)} video files in directory: {args.video_path}")
        
    elif video_path.is_file():
        # Process single video file
        video_files = [video_path]
        logger.info(f"Processing single video file: {args.video_path}")
        
    else:
        logger.error(f"Path does not exist: {args.video_path}")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print system information
    if torch is not None:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
    
    # Initialize VideoClassifier
    logger.info("Initializing VideoClassifier...")
    try:
        detector = VideoClassifier(
            device='cuda' if args.device >= 0 else 'cpu',
            mode=args.mode,
            ctx_id=args.device,
            det_size=tuple(args.det_size),
            visualize=args.visualize,
        )
        logger.info("VideoClassifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize VideoClassifier: {e}")
        sys.exit(1)
    
    # Process videos    
    successful_processes = 0
    failed_processes = 0
    
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"=============================================")
        logger.info(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        
        results = {
            'transaction_id': video_file.stem,
            'code': "0000", # 0000 is a placeholder for success
            'message': "Success" # Placeholder message keys
        }

        classification_results = detector.classify_video(
            video_path=str(video_file),
            max_frames=args.max_frames,
            frame_skip=args.frame_skip
        )
        results.update(classification_results)
        logger.info(f"Results for {video_file.name}: {results.get('class', 'Unknown')}")
        
        # Determine output path
        if args.output is None:
            output_path = video_file.with_name(video_file.stem + "_out.json")
        else:
            output_base = Path(args.output)
            # If output is a directory or has no extension, treat as directory
            if output_base.is_dir() or not output_base.suffix:
                output_base.mkdir(parents=True, exist_ok=True)
                output_path = output_base / f"{video_file.stem}_out.json"
            else:
                # Use the specified file path, ensure .json extension
                output_path = output_base.with_suffix('.json')
                output_path.parent.mkdir(parents=True, exist_ok=True)
           
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results to JSON
        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results for {video_file.name}: {e}")
            failed_processes += 1
            continue
        
        # Create visualization if requested
        # Set to False due to issues on saving videos
        if False:
            try:
                if args.output_video:
                    if len(video_files) == 1:
                        output_video_path = args.output_video
                    else:
                        # Multiple files: modify output video name
                        output_video_base = Path(args.output_video)
                        output_video_path = str(output_video_base.parent / f"{output_video_base.stem}_{video_file.stem}{output_video_base.suffix}")
                else:
                    # Default: save next to original video with '_detected' suffix
                    output_video_path = str(video_file.with_name(video_file.stem + "_detected.mp4"))
                
                detector.visualize_detections(
                    video_path=str(video_file),
                    results=results,
                    output_video_path=output_video_path
                )
                logger.info(f"Visualization saved to: {output_video_path}")
            except Exception as e:
                logger.error(f"Failed to create visualization for {video_file.name}: {e}")
        successful_processes += 1

    # Print summary
    logger.info(f"Processing complete: {successful_processes} successful, {failed_processes} failed")
    
    if failed_processes > 0:
        sys.exit(1) 
    


def _print_results_summary(results: dict):
    """Print a formatted summary of detection results."""
    print("\n" + "="*60)
    print("SCRFD FACE DETECTION RESULTS")
    print("="*60)
    
    #video_info = results['video_info']
    detection_summary = results['detection_summary']
    timing = results['timing']
    
    print(f"Video: {video_info['filename']}")
    print(f"Resolution: {video_info['resolution']}")
    print(f"Duration: {video_info['duration_seconds']:.2f} seconds")
    print(f"Total frames: {video_info['total_frames']}")
    print(f"Frames processed: {results['processing_info']['frames_processed']}")
    print(f"")
    print(f"Classification: {detection_summary['classification']}")
    print(f"Binary Label: {detection_summary['binary_label']}")
    print(f"Confidence: {detection_summary['confidence']:.2%}")
    print(f"Summary: {detection_summary['summary']}")
    print(f"")
    print(f"Total faces detected: {detection_summary['total_faces_detected']}")
    print(f"Average faces per frame: {detection_summary['avg_faces_per_frame']:.2f}")
    print(f"Max faces in single frame: {detection_summary['max_faces_in_frame']}")
    print(f"Frames with faces: {detection_summary['frames_with_faces']}/{detection_summary['total_frames_processed']}")
    print(f"Face detection rate: {detection_summary['face_detection_rate']:.2%}")
    print(f"")
    print(f"Model: {results['processing_info']['model_name']}")
    print(f"Detection size: {results['processing_info']['detection_size']}")
    print(f"Providers: {results['processing_info']['providers']}")
    print(f"")
    print("Timing Information:")
    print(f"Model initialization: {timing.get('model_initialization', 0):.3f}s")
    print(f"Total processing: {timing.get('total_processing', 0):.3f}s")


if __name__ == "__main__":
    main()