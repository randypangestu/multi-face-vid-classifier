#!/usr/bin/env python3
"""
SageMaker inference script for Multi-Face Video Classifier.
Accepts JSON input with base64-encoded video data and returns classification results.
(Currently not tested, and only serves as an example for how to use the model in SageMaker)
"""
import json
import logging
import os
import sys
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, Union
import torch

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SageMakerInferenceHandler:
    """SageMaker inference handler for multi-face video classification."""
    
    def __init__(self):
        """Initialize the inference handler with default settings."""
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Default settings optimized for SageMaker
        self.default_settings = {
            'mode': 'rc6',  # Most advanced mode
            'max_frames': 50,  # Optimal frame count
            'frame_skip': 2,  # Process every 2nd frame for speed
            'model_name': 'buffalo_l',
            'det_size': (640, 640),
            'device': self.device,
            'ctx_id': 0 if self.device == 'cuda' else -1,
            'visualize': False
        }
        
        logger.info(f"Handler initialized with device: {self.device}")
        
    def model_fn(self, model_dir: str = None):
        """Load the model for SageMaker inference."""
        try:
            logger.info("Loading VideoClassifier for inference...")
            
            # Import the main classifier (RC6 version)
            from multi_face_video_classifier.video_pipeline import VideoClassifier
            
            # Initialize with default settings
            classifier = VideoClassifier(
                device=self.default_settings['device'],
                mode=self.default_settings['mode'],
                ctx_id=self.default_settings['ctx_id'],
                det_size=self.default_settings['det_size'],
                visualize=self.default_settings['visualize']
            )
            
            logger.info("VideoClassifier loaded successfully")
            return classifier
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def input_fn(self, request_body: Union[str, bytes], content_type: str = 'application/json') -> Dict[str, Any]:
        """Parse input data from SageMaker request."""
        try:
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            
            if content_type == 'application/json':
                input_data = json.loads(request_body)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Validate required fields
            if 'video_data' not in input_data:
                raise ValueError("Missing required field 'video_data' in input JSON")
            
            if 'video_id' not in input_data:
                input_data['video_id'] = 'unknown'
            
            # Extract optional parameters with defaults
            params = {
                'max_frames': input_data.get('max_frames', self.default_settings['max_frames']),
                'frame_skip': input_data.get('frame_skip', self.default_settings['frame_skip']),
                'mode': input_data.get('mode', self.default_settings['mode'])
            }
            
            return {
                'video_data': input_data['video_data'],
                'video_id': input_data['video_id'],
                'params': params
            }
            
        except Exception as e:
            logger.error(f"Error parsing input: {str(e)}")
            raise
    
    def predict_fn(self, input_data: Dict[str, Any], model) -> Dict[str, Any]:
        """Run inference on the input data."""
        video_id = input_data['video_id']
        video_data = input_data['video_data']
        params = input_data['params']
        
        logger.info(f"Processing video_id: {video_id}")
        
        try:
            # Decode base64 video data
            video_bytes = base64.b64decode(video_data)
            
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_bytes)
                temp_video_path = temp_video.name
            
            try:
                # Run classification
                logger.info(f"Running classification with params: {params}")
                
                classification_results = model.classify_video(
                    video_path=temp_video_path,
                    max_frames=params['max_frames'],
                    frame_skip=params['frame_skip']
                )
                
                # Prepare response
                response = {
                    'video_id': video_id,
                    'status': 'success',
                    'code': '0000',
                    'message': 'Classification completed successfully',
                    'results': classification_results,
                    'processing_params': params
                }
                
                logger.info(f"Classification completed for {video_id}: class {classification_results.get('class', 'unknown')}")
                return response
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_video_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e}")
        
        except Exception as e:
            logger.error(f"Error during prediction for {video_id}: {str(e)}")
            return {
                'video_id': video_id,
                'status': 'error',
                'code': '1001',
                'message': f'Classification failed: {str(e)}',
                'results': None,
                'processing_params': params
            }
    
    def output_fn(self, prediction: Dict[str, Any], accept: str = 'application/json') -> Union[str, bytes]:
        """Format the prediction output."""
        try:
            if accept == 'application/json':
                return json.dumps(prediction, indent=2)
            else:
                raise ValueError(f"Unsupported accept type: {accept}")
        except Exception as e:
            logger.error(f"Error formatting output: {str(e)}")
            raise


# Global handler instance for SageMaker
_handler = SageMakerInferenceHandler()

# SageMaker entry points
def model_fn(model_dir: str = None):
    """SageMaker model loading function."""
    return _handler.model_fn(model_dir)

def input_fn(request_body: Union[str, bytes], content_type: str = 'application/json'):
    """SageMaker input parsing function."""
    return _handler.input_fn(request_body, content_type)

def predict_fn(input_data: Dict[str, Any], model):
    """SageMaker prediction function."""
    return _handler.predict_fn(input_data, model)

def output_fn(prediction: Dict[str, Any], accept: str = 'application/json'):
    """SageMaker output formatting function."""
    return _handler.output_fn(prediction, accept)


def main():
    """Main function for testing the inference script locally."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SageMaker inference script locally")
    parser.add_argument("--input-json", required=True, help="Path to input JSON file")
    parser.add_argument("--output-json", help="Path to save output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load model
    logger.info("Loading model...")
    model = model_fn()
    
    # Load input data
    logger.info(f"Loading input from: {args.input_json}")
    with open(args.input_json, 'r') as f:
        request_body = f.read()
    
    # Parse input
    input_data = input_fn(request_body, 'application/json')
    
    # Run prediction
    logger.info("Running prediction...")
    prediction = predict_fn(input_data, model)
    
    # Format output
    output = output_fn(prediction, 'application/json')
    
    # Save or print output
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(output)
        logger.info(f"Output saved to: {args.output_json}")
    else:
        print("\nInference Result:")
        print(output)


if __name__ == "__main__":
    main()