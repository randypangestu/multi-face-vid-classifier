#!/usr/bin/env python3
"""
Utility script to encode video files to base64 for SageMaker inference testing.
"""
import json
import base64
import argparse
from pathlib import Path


def encode_video_to_base64(video_path: str) -> str:
    """Encode a video file to base64 string."""
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        encoded_string = base64.b64encode(video_bytes).decode('utf-8')
        return encoded_string


def create_sagemaker_input(video_path: str, video_id: str = None, 
                          max_frames: int = 50) -> dict:
    """Create SageMaker input JSON from video file."""
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if video_id is None:
        video_id = video_path.stem
    
    # Encode video to base64
    encoded_video = encode_video_to_base64(str(video_path))
    
    # Create input JSON
    input_json = {
        "video_id": video_id,
        "video_data": encoded_video,
        "max_frames": max_frames,
        "mode": mode
    }
    
    return input_json


def main():
    parser = argparse.ArgumentParser(
        description="Encode video file to base64 for SageMaker inference"
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--video-id", help="Video ID (default: filename stem)")
    parser.add_argument("--max-frames", type=int, default=50,
                       help="Maximum frames to process (default: 50)")
    parser.add_argument("--frame-skip", type=int, default=50,
                       help="Frame skip value (default: 50)")
    parser.add_argument("--mode", default="rc6", 
                       choices=["rc1", "rc3", "rc4", "rc5", "rc6"],
                       help="Processing mode (default: rc6)")
    
    args = parser.parse_args()
    
    try:
        # Create SageMaker input JSON
        input_json = create_sagemaker_input(
            video_path=args.video_path,
            video_id=args.video_id,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            mode=args.mode
        )
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            video_path = Path(args.video_path)
            output_path = f"{video_path.stem}_sagemaker_input.json"
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        print(f"SageMaker input JSON created: {output_path}")
        print(f"Video ID: {input_json['video_id']}")
        print(f"Video size (base64): {len(input_json['video_data'])} characters")
        print(f"Parameters: max_frames={input_json['max_frames']}, "
              f"frame_skip={input_json['frame_skip']}, mode={input_json['mode']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())