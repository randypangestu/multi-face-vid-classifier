#!/usr/bin/env python3
"""
SCRFD Face Detector Module

This module provides the main SCRFD face detection functionality using InsightFace
with ONNX Runtime backend and comprehensive face analysis capabilities.
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
# Import required libraries
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import pdb

import onnxruntime as ort

# Import local modules
from .card_detector import CardDetector
from .utils.pose_estimator import PoseEstimator
from .utils.face_tracking_utils import FaceTracker, FaceStatisticsCalculator
from .utils.visualization_utils import VideoVisualizationProcessor, ResultsAnalyzer
import random
#ignore warnings
import warnings
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)
# Set up logging level
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
ort.set_default_logger_severity(4)   

MODELS_DICT = {
    'face_analysis': 'buffalo_l'
}
# ignore warnings 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='onnxruntime')

VERSION_DICT = {}

class VideoClassifier:
    """
    Video Classifier to classify videos based on the presence of multiple live faces.
    0 - No multiple live faces detected
    1 - Multiple live faces detected
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 mode: str = 'rc6',
                 ctx_id: int = 0,
                 det_size: Tuple[int, int] = (640, 640),
                 visualize: bool = False):
        """
        
        Args:
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            ctx_id: GPU device ID (0 for first GPU, -1 for CPU)
            det_size: Detection input size (width, height)
        """
        # if deivice is 'cpu', or not cuda or not 0, or 1, set providers to ['CPUExecutionProvider'
        if device == 'cpu' or not torch.cuda.is_available() or ctx_id < 0:
            providers = ['CPUExecutionProvider']
            logger.info("Using CPU for inference")
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info(f"Using CUDA with ONNX Runtime on device {ctx_id}")

        self.model_name = MODELS_DICT['face_analysis']
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.mode = mode
        self.visualize = visualize
        # Set default providers based on available hardware
        if providers is None:
            providers = self._get_default_providers()
        
        self.providers = providers
        self.face_analysis = None
        self.timing_data = {}
        self.card_detector = CardDetector(
            device='cuda' if ctx_id >= 0 else 'cpu',
            mode = mode,
        )
        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.face_tracker = FaceTracker()
        self.stats_calculator = FaceStatisticsCalculator()
        self.visualization_processor = VideoVisualizationProcessor()
        self.results_analyzer = ResultsAnalyzer()
        # Initialize face analysis app
        self._initialize_face_app()
    
    def _get_default_providers(self) -> List[str]:
        """Get default ONNX Runtime providers based on available hardware."""
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers and self.ctx_id >= 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info(f"Using CUDA with ONNX Runtime on device {self.ctx_id}")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("Using CPU for inference")
        return providers
    
    def _initialize_face_app(self):
        """Initialize the InsightFace application."""
        try:
            start_time = time.time()
            
            # Initialize FaceAnalysis app
            self.face_analysis = FaceAnalysis(
                name=self.model_name,
                providers=self.providers
            )
            
            # Prepare the app with detection size
            self.face_analysis.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            
            init_time = time.time() - start_time
            self.timing_data['model_initialization'] = init_time
            
            logger.debug(f"SCRFD model '{self.model_name}' initialized successfully in {init_time:.3f}s")
            logger.debug(f"Detection size: {self.det_size}")
            logger.info(f"Providers: {self.providers}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SCRFD model: {str(e)}")
            raise
    
    def detect_faces_in_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single image with pose estimation.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detection results with bounding boxes, landmarks, and pose
        """
        if self.face_analysis is None:
            raise RuntimeError("Face detection app not initialized")
        
        start_time = time.time()
        image_height, image_width = image.shape[:2]
        
        try:
            # Run face detection and analysis
            faces = self.face_analysis.get(image)
            detection_time = time.time() - start_time
            
            # Process results
            face_results = []
            for i, face in enumerate(faces):
                face_result = self._process_single_face_detection(
                    face, i, image_width, image_height, detection_time
                )
                face_results.append(face_result)
            
            return face_results
            
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return []
    
    def _process_single_face_detection(self, face, face_id: int, image_width: int, 
                                     image_height: int, detection_time: float) -> Dict[str, Any]:
        """Process a single face detection result."""
        # Extract bounding box
        bbox = face.bbox.astype(int) if hasattr(face, 'bbox') else None
        
        # Extract facial landmarks (5 points: eyes, nose, mouth corners)
        landmarks = face.kps.astype(int) if hasattr(face, 'kps') else None
        
        # Estimate head pose if landmarks are available
        pose = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'valid': False}
        if landmarks is not None:
            pose = self.pose_estimator.estimate_pose(landmarks, image_width, image_height)
        
        # Extract additional attributes if available
        age = getattr(face, 'age', None)
        gender = getattr(face, 'gender', None)
        embedding = getattr(face, 'embedding', None)
        
        return {
            'face_id': face_id,
            'bbox': bbox.tolist() if bbox is not None else None,
            'landmarks': landmarks.tolist() if landmarks is not None else None,
            'confidence': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
            'pose': pose,
            'age': int(age) if age is not None else None,
            'gender': int(gender) if gender is not None else None,
            'embedding_available': embedding is not None,
            'embedding': embedding.tolist() if embedding is not None else None,
            'detection_time': detection_time
        }
    
    def _get_card_bounding_boxes(self, image: np.ndarray) -> List:
        """
        Using grounding dino to detect card bounding box.
        This is a placeholder function, as the actual implementation would depend on the specific model and method
        used for card detection.
        """
        card_bboxes = self.card_detector.detect(image)
        return card_bboxes

    def _is_bbox_inside(self, face_bbox: List[float], card_bbox: List[float]) -> bool:
        """
        Check if the face bounding box is inside the card bounding box.
        
        Args:
            face_bbox: List of [x1, y1, x2, y2] for face bounding box
            card_bbox: List of [x1, y1, x2, y2] for card bounding box
            
        Returns:
            True if face_bbox is inside card_bbox, False otherwise
        """
       
        offset = 25
        bbox_inside_flag = (face_bbox[0] >= card_bbox[0] - offset and
                face_bbox[1] >= card_bbox[1] - offset and
                face_bbox[2] <= card_bbox[2] + offset )
                #face_bbox[3] <= card_bbox[3] + offset)
        if bbox_inside_flag:
            return True
        
        return (face_bbox[0] >= card_bbox[0] and
                face_bbox[1] >= card_bbox[1] and
                face_bbox[2] <= card_bbox[2] and
                face_bbox[3] <= card_bbox[3])

    def _check_fr_similarity(self, face: Dict[str, Any], embedding_live_face_list: List[np.ndarray]):
        """
        Check and update the similarity of face embeddings.

        Args:
            embedding_live_face_list (List[np.ndarray]): List of live face embeddings
            face (Dict[str, Any]): Face information dictionary containing 'embedding'

        Returns:
            List[np.ndarray]: Updated list of live face embeddings
            bool: True if the face is similar to existing embeddings, False otherwise
        """
        if len(embedding_live_face_list) == 0:
            embedding_live_face_list.append(face['embedding'])
            return embedding_live_face_list, False
        if 'embedding' in face:
            embedding = face['embedding']
            # Compare with existing embeddings and keep the most similar ones
            # This is a placeholder for the actual similarity check
            similar_embeddings = [emb for emb in embedding_live_face_list if self._is_embedding_similar(emb, embedding)]
            if similar_embeddings:
                return embedding_live_face_list, True
            else:
                embedding_live_face_list.append(embedding)
        return embedding_live_face_list, False

    def _is_embedding_similar(self, reference: list, embedding: list, threshold=0.5) -> bool:
        # cosine similarity, should be above threshold to be similar
        cos_sim = np.dot(reference, embedding) / (np.linalg.norm(reference) * np.linalg.norm(embedding))
        flag_same_face = cos_sim > threshold
        return flag_same_face


    def _get_unique_live_faces(self, frame, card_bboxes, not_live_face_idx, embedding_live_face_list, threshold=0.7) -> List[Dict[str, Any]]:
        """
        Get a list of unique live faces from the current frame.

        Returns:
            List[Dict[str, Any]]: List of unique live face information dictionaries.
        """
        for idx_j, face in enumerate(frame['faces']):
            if idx_j not in not_live_face_idx:
                # ignore face that has less than 70% visible area (occluded by cards)
                occluded_face = self._check_face_occlusion(face['bbox'], card_bboxes, threshold=threshold)
                if not occluded_face:  # Only process non-occluded faces
                    embedding_live_face_list, flag_similar = self._check_fr_similarity(face, embedding_live_face_list)
        return embedding_live_face_list

    def _remove_faces_inside_card(self, frame, card_bboxes, number_live_faces):
        """
        Remove faces that are inside the card bounding boxes.

        Args:
            frame: Frame dictionary containing 'faces' key with face information
            card_bboxes: List of card bounding boxes
            number_live_faces: Current count of live faces

        Returns:
            Tuple: Updated number of live faces and indices of non-live faces
        """
        not_live_face_idx = []
        for idx, face in enumerate(frame['faces']):
            face_bbox = face['bbox']
            for card_bbox in card_bboxes:
                if self._is_bbox_inside(face_bbox, card_bbox):
                    number_live_faces -= 1
                    not_live_face_idx.append(idx)
                    break
        return number_live_faces, not_live_face_idx

    def _classify_multiple_live_faces(self, 
                                     frames_list: List[dict],
                                    ):
        """
        Classify whether the videos (all frames) contain multiple live faces.

        Args:
            frames_list (List[dict]): List of frame results from video processing
                flagged_frame:  {
                        'frame_index': int(frame_idx),
                        'frame': frame,
                        'num_faces': len(faces),
                        'faces': faces
                    }
        """
        number_live_faces = 0
        logger.debug(f"Mode: {self.mode}")
        embedding_live_face_list = []
        for idx, frame in enumerate(frames_list):
            
            card_bboxes = self._get_card_bounding_boxes(frame['frame'])
            number_live_faces = len(frame['faces'])
            # if any of the face[bbox] is inside the card_bboxes, then it is not a live face, if not it is live face

            number_live_faces, not_live_face_idx = self._remove_faces_inside_card(
                frame, card_bboxes, number_live_faces
            )

            embedding_live_face_list = self._get_unique_live_faces(
                frame, card_bboxes, not_live_face_idx, embedding_live_face_list
            )
            # override number of live faces with the number of unique embeddings 
            number_live_faces = max(number_live_faces, len(embedding_live_face_list))
            
            if self.visualize:
                if number_live_faces > 1:
                    prefix_debug = "multiple" 
                else:
                    prefix_debug = "single"
                self._draw_face_card(card_bboxes, frame, idx, prefix_debug)
    
            if number_live_faces > 1:
                logger.info(f"Multiple live faces detected in frame {idx}: {number_live_faces} faces")
                return {
                    'contain_multiple_live_faces': True,
                    'n_faces': number_live_faces,
                    'class': 1,
                    }
        return {
               'contain_multiple_live_faces': False,
                'n_faces': number_live_faces,
               'class': 0,
               }

    def _check_face_occlusion(self, face_bbox, card_bboxes, threshold=0.7):
        """
        Check if face is occluded by calculating the percentage of face area NOT covered by cards.
        
        Args:
            face_bbox: List of [x1, y1, x2, y2] for face bounding box
            card_bboxes: List of card bounding boxes, each as [x1, y1, x2, y2]
            threshold: Minimum percentage of face that must be visible (not occluded) to be considered non-occluded
            
        Returns:
            bool: True if face is occluded (visible percentage < threshold), False otherwise
        """
        if not card_bboxes:
            return False  # No cards means no occlusion
        
        visible_percentage = self._calculate_face_visible_percentage(face_bbox, card_bboxes)
        
        # Face is occluded if visible percentage is below threshold
        occlusion_flag = visible_percentage < threshold
        return occlusion_flag
    
    def _calculate_face_visible_percentage(self, face_bbox, card_bboxes):
        """
        Calculate the percentage of face area that is NOT covered by any card bounding boxes.
        
        Args:
            face_bbox: List of [x1, y1, x2, y2] for face bounding box
            card_bboxes: List of card bounding boxes, each as [x1, y1, x2, y2]
            
        Returns:
            float: Percentage of face area that is visible (0.0 to 1.0)
        """
        x1_face, y1_face, x2_face, y2_face = face_bbox
        face_area = (x2_face - x1_face) * (y2_face - y1_face)
        
        if face_area <= 0:
            return 0.0
        
        # Calculate total overlapped area from all cards
        total_overlap_area = 0
        
        for card_bbox in card_bboxes:
            overlap_area = self._calculate_intersection_area(face_bbox, card_bbox)
            total_overlap_area += overlap_area
        
        # Handle case where overlaps might count the same area multiple times
        # Cap the total overlap at the face area
        total_overlap_area = min(total_overlap_area, face_area)
        
        # Calculate visible area and percentage
        visible_area = face_area - total_overlap_area
        visible_percentage = visible_area / face_area
        
        return max(0.0, visible_percentage)  # Ensure non-negative
    
    def _calculate_intersection_area(self, bbox1, bbox2):
        """
        Calculate intersection area between two bounding boxes.
        
        Args:
            bbox1: List of [x1, y1, x2, y2] for first bounding box
            bbox2: List of [x1, y1, x2, y2] for second bounding box
            
        Returns:
            float: Intersection area
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0
        else:
            return (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1: List of [x1, y1, x2, y2] for first bounding box
            bbox2: List of [x1, y1, x2, y2] for second bounding box
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            intersection_area = 0
        else:
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate areas of both bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0
        
        iou = intersection_area / union_area
        return iou
    def _draw_face_card(self, card_bboxes, frame, idx, prefix):
        # draw all card bounding boxes
        for card_bbox in card_bboxes:
            x1, y1, x2, y2 = map(int, card_bbox)
            cv2.rectangle(frame['frame'], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame['frame'], "Card", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # draw all face bounding boxes
        for face in frame['faces']:
            face_bbox = face['bbox']
            x1, y1, x2, y2 = map(int, face_bbox)
            cv2.rectangle(frame['frame'], (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame['frame'], "Face", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(f"{prefix}_live_faces_{idx}_{random.randint(0,100)}.jpg", frame['frame'])

    def classify_video(self, video_path: str, 
                     max_frames: int = 100,
                     frame_skip: int = 1) -> Dict[str, Any]:
        """
        Process video for face detection across multiple frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_skip: Process every nth frame (1 = process all frames)
            
        Returns:
            Dictionary containing comprehensive face detection results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        start_time = time.time()
        
        # Open and validate video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        try:
            # Get video properties
            video_info = self._extract_video_info(cap, video_path)
            logger.debug(f"Total frames: {video_info['total_frames']}, FPS: {video_info['fps']:.2f}, Duration: {video_info['duration_seconds']:.2f}s")
            
            # Process frames
            frame_results, flagged_frames = self._process_video_frames(cap, video_info, max_frames, frame_skip)
            logger.debug(f'TOTAL FLAGGED FRAMES: {len(flagged_frames)}')

            video_classification = self._classify_multiple_live_faces(flagged_frames)
            logger.info(f"Video classification result, path: {video_path}, contain_multiple_live_faces: {video_classification['contain_multiple_live_faces']}")
            # Analyze results with tracking
            if False:
                analysis = self._analyze_video_results(frame_results, video_path)
                
                # Compile final results
                results = self._compile_video_results(
                    video_info, frame_results, analysis, max_frames, frame_skip, start_time
                )
            logger.info(f"class: {video_classification['class']}")
            logger.info(f"n_faces: {video_classification['n_faces']}")           
            
            return video_classification
            
        finally:
            cap.release()
    
    def _extract_video_info(self, cap: cv2.VideoCapture, video_path: str) -> Dict[str, Any]:
        """
        Extract basic video information.
        will be used to get video properties like total frames, FPS, resolution, etc.
        so that we can select frames to process.
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return {
            'path': video_path,
            'filename': Path(video_path).name,
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration,
            'resolution': f"{width}x{height}",
            'width': width,
            'height': height
        }
    
    def _process_video_frames(self, cap: cv2.VideoCapture, video_info: Dict, 
                            max_frames: int, frame_skip: int) -> List[Dict]:
        """Process video frames for face detection."""
        # Determine frames to process
        frames_to_process = min(max_frames, video_info['total_frames'] // frame_skip)
        frame_indices = np.linspace(0, video_info['total_frames'] - 1, frames_to_process, dtype=int)
        
        frame_results = []
        flagged_frames_list = []
        processed_count = 0
        
        prev_frames_num_faces = 0
        delay_before_flag = 2 # Number of frames to wait before flagging a change in face count
        saving_flagged_frame = False
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect faces in current frame
            faces = self.detect_faces_in_image(frame)
            num_faces = len(faces)
            timestamp = frame_idx / video_info['fps'] if video_info['fps'] > 0 else 0

            if num_faces != prev_frames_num_faces:
                if saving_flagged_frame:
                    flagged_frame['frame'] = temp_frame
                    flagged_frames_list.append(flagged_frame)
                    saving_flagged_frame = False
                    delay_before_flag = 2
                prev_frames_num_faces = num_faces
                flagged_frame = {
                    'frame_index': int(frame_idx),
                    'frame': frame,
                    'num_faces': len(faces),
                    'faces': faces
                }
                saving_flagged_frame = True
                temp_frame = frame.copy()
            

            if saving_flagged_frame:
                delay_before_flag -= 1
                if delay_before_flag <= 0:
                    flagged_frame = {
                        'frame_index': int(frame_idx),
                        'frame': frame,
                        'num_faces': len(faces),
                        'faces': faces
                    }
                    flagged_frames_list.append(flagged_frame)
                    saving_flagged_frame = False
                    delay_before_flag = 2  # Reset delay for next flagging

            frame_result = {
                'frame_index': int(frame_idx),
                'timestamp': timestamp,
                'num_faces': num_faces,
                'faces': faces
                }
            frame_results.append(frame_result)
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.debug(f"Processed {processed_count}/{frames_to_process} frames")
        
        return frame_results, flagged_frames_list
    
    def _analyze_video_results(self, frame_results: List[Dict], video_path: str) -> Dict[str, Any]:
        """Analyze video results with face tracking and statistics."""
        # Basic analysis
        basic_analysis = self.results_analyzer.analyze_detection_results(frame_results, video_path)
        
        # Face tracking analysis
        face_tracks = self.face_tracker.track_faces_across_frames(frame_results)
        face_statistics = self.stats_calculator.calculate_face_statistics(face_tracks, frame_results)
        
        # Combine analyses
        analysis = {
            **basic_analysis,
            'face_statistics': face_statistics,
            'unique_faces_detected': len(face_statistics)
        }
        
        return analysis
    
    def _compile_video_results(self, video_info: Dict, frame_results: List[Dict], 
                             analysis: Dict, max_frames: int, frame_skip: int, 
                             start_time: float) -> Dict[str, Any]:
        """Compile comprehensive video processing results."""
        total_time = time.time() - start_time
        self.timing_data['total_processing'] = total_time
        
        return {
            'video_info': video_info,
            'processing_info': {
                'frames_processed': len(frame_results),
                'frame_skip': frame_skip,
                'max_frames_limit': max_frames,
                'model_name': self.model_name,
                'detection_size': self.det_size,
                'providers': self.providers
            },
            'detection_summary': analysis,
            'frame_results': frame_results,
            'timing': self.timing_data.copy()
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save detection results to JSON file with enhanced face statistics.
        
        Args:
            results: Detection results dictionary
            output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add summary statistics for face tracks
        self._add_face_rankings_to_results(results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Print face statistics summary
        self._print_face_statistics_summary(results)
    
    def _add_face_rankings_to_results(self, results: Dict[str, Any]):
        """Add face rankings and best face summary to results."""
        if 'detection_summary' in results and 'face_statistics' in results['detection_summary']:
            face_stats = results['detection_summary']['face_statistics']
            
            # Create summary of best faces
            face_rankings = []
            for track_id, stats in face_stats.items():
                face_rankings.append({
                    'track_id': track_id,
                    'quality_score': stats['derived_stats']['face_quality_score'],
                    'total_detections': stats['total_detections'],
                    'average_confidence': stats['confidence_stats']['average'],
                    'is_primary_face': stats['derived_stats']['primary_face'],
                    'is_looking_at_camera': stats['derived_stats']['is_looking_at_camera']
                })
            
            # Sort by quality score
            face_rankings.sort(key=lambda x: x['quality_score'], reverse=True)
            results['detection_summary']['face_rankings'] = face_rankings
            
            # Add best face summary
            if face_rankings:
                best_face = face_rankings[0]
                results['detection_summary']['best_face'] = {
                    'track_id': best_face['track_id'],
                    'quality_score': best_face['quality_score'],
                    'statistics': face_stats[best_face['track_id']]
                }
    
    def _print_face_statistics_summary(self, results: Dict[str, Any]):
        """Print detailed face statistics summary to console."""
        if 'detection_summary' not in results or 'face_statistics' not in results['detection_summary']:
            return
        
        face_stats = results['detection_summary']['face_statistics']
        print(f"\nFace Statistics Summary:")
        print(f"Unique faces detected: {len(face_stats)}")
        
        for track_id, stats in face_stats.items():
            print(f"\n{track_id}:")
            print(f"  Detections: {stats['total_detections']}")
            print(f"  Duration: {stats['duration_seconds']:.2f}s")
            print(f"  Avg Confidence: {stats['confidence_stats']['average']:.3f}")
            
            bbox_stats = stats.get('bbox_size_stats', {})
            if 'area' in bbox_stats:
                print(f"  Avg Face Size: {bbox_stats['area']['average']:.0f} pixels²")
            
            print(f"  Looking at Camera: {stats['derived_stats']['is_looking_at_camera']}")
            print(f"  Quality Score: {stats['derived_stats']['face_quality_score']:.3f}")
            
            pose_stats = stats.get('pose_stats', {})
            if 'pose_deviation' in pose_stats:
                avg_pose_dev = pose_stats['pose_deviation']['average']
                print(f"  Avg Pose Deviation: {avg_pose_dev:.1f}°")
            
            position_stats = stats.get('position_stats', {})
            if position_stats:
                total_movement = position_stats.get('total_movement_distance', 0)
                print(f"  Total Movement: {total_movement:.0f} pixels")
                print(f"  Is Stationary: {position_stats.get('is_stationary', False)}")
    
    def visualize_detections(self, video_path: str, results: Dict[str, Any], 
                           output_video_path: str = None, 
                           max_frames_to_save: int = 1000) -> str:
        """
        Create visualization video with face detection overlays and pose information.
        
        Args:
            video_path: Original video path
            results: Detection results
            output_video_path: Path to save visualization video
            max_frames_to_save: Maximum frames to include in output video
            
        Returns:
            Path to the created visualization video
        """
        return self.visualization_processor.create_visualization_video(
            video_path, results, output_video_path, max_frames_to_save
        )