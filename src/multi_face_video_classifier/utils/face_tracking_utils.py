#!/usr/bin/env python3
"""
Face Tracking Utilities Module

This module provides face tracking, enrollment, and verification utilities
for multi-frame face detection systems.
"""

import math
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Face tracking system with enrollment and verification capabilities.
    """
    
    def __init__(self, max_distance_threshold: int = 100, embedding_similarity_threshold: float = 0.4):
        """
        Initialize face tracker.
        
        Args:
            max_distance_threshold: Maximum pixel distance for spatial tracking
            embedding_similarity_threshold: Threshold for face embedding similarity
        """
        self.max_distance_threshold = max_distance_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
    
    def track_faces_across_frames(self, frame_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Track faces across frames with face enrollment and verification.
        
        Args:
            frame_results: List of per-frame detection results
            
        Returns:
            Dictionary mapping track IDs to detection sequences
        """
        face_tracks = {}
        next_track_id = 0
        previous_face_count = 0
        face_embeddings = {}
        enrolled_faces = {}
        
        for frame_idx, frame_result in enumerate(frame_results):
            current_faces = frame_result['faces']
            current_face_count = len(current_faces)
            matched_tracks = set()
            
            # Check if we need to enroll faces
            need_enrollment = self._should_perform_enrollment(
                frame_idx, current_face_count, previous_face_count
            )
            
            if need_enrollment:
                next_track_id = self._perform_face_enrollment(
                    current_faces, frame_result, enrolled_faces, face_embeddings,
                    face_tracks, matched_tracks, next_track_id
                )
            else:
                # Regular spatial tracking for continuous frames
                self._perform_spatial_tracking(
                    current_faces, frame_result, face_tracks, face_embeddings,
                    enrolled_faces, matched_tracks
                )
            
            previous_face_count = current_face_count
        
        # Add enrollment/verification statistics to tracks
        self._add_enrollment_statistics(face_tracks)
        
        return face_tracks
    
    def _should_perform_enrollment(self, frame_idx: int, current_count: int, previous_count: int) -> bool:
        """Determine if face enrollment should be performed for current frame."""
        return (
            frame_idx == 0 or  # First frame
            current_count != previous_count or  # Face count changed
            current_count > 1  # Multiple faces in frame
        )
    
    def _perform_face_enrollment(self, current_faces: List[Dict], frame_result: Dict,
                               enrolled_faces: Dict, face_embeddings: Dict,
                               face_tracks: Dict, matched_tracks: set, next_track_id: int) -> int:
        """
        Perform face enrollment for all faces in current frame.
        
        Returns:
            Updated next_track_id
        """
        for face in current_faces:
            if face.get('embedding') is None:
                continue
            
            face_embedding = face['embedding']
            best_match = self._find_best_embedding_match(face_embedding, enrolled_faces)
            
            if best_match is None:
                # New face detected - create new enrollment
                enrolled_id = f"face_{next_track_id}"
                enrolled_faces[enrolled_id] = face_embedding
                face_embeddings[enrolled_id] = face_embedding
                
                # Create new track
                face_tracks[enrolled_id] = [self._create_detection_entry(
                    frame_result, face, enrollment_data={
                        'enrolled_at': frame_result['timestamp'],
                        'enrollment_frame': frame_result['frame_index'],
                        'embedding_quality': face.get('confidence', 0.0)
                    }
                )]
                next_track_id += 1
                matched_tracks.add(enrolled_id)
            else:
                # Update existing track with verified identity
                similarity_score = self._calculate_embedding_similarity(face_embedding, enrolled_faces[best_match])
                face_tracks[best_match].append(self._create_detection_entry(
                    frame_result, face, verification_data={
                        'verified_id': best_match,
                        'similarity_score': similarity_score
                    }
                ))
                matched_tracks.add(best_match)
                
                # Update embedding if quality is better
                if self._is_better_quality(face, face_tracks[best_match][0]):
                    face_embeddings[best_match] = face_embedding
                    enrolled_faces[best_match] = face_embedding
        
        return next_track_id
    
    def _perform_spatial_tracking(self, current_faces: List[Dict], frame_result: Dict,
                                face_tracks: Dict, face_embeddings: Dict,
                                enrolled_faces: Dict, matched_tracks: set):
        """Perform spatial tracking for continuous frames."""
        for face in current_faces:
            current_center = self.get_bbox_center(face['bbox']) if face['bbox'] else None
            if current_center is None:
                continue
            
            best_match = self._find_best_spatial_match(current_center, face_tracks, matched_tracks)
            
            if best_match:
                # Update existing track
                face_tracks[best_match].append(self._create_detection_entry(frame_result, face))
                matched_tracks.add(best_match)
                
                # Update embedding if available and better quality
                if (face.get('embedding') is not None and 
                    self._is_better_quality(face, face_tracks[best_match][0])):
                    face_embeddings[best_match] = face['embedding']
                    enrolled_faces[best_match] = face['embedding']
    
    def _find_best_embedding_match(self, face_embedding: np.ndarray, enrolled_faces: Dict) -> Optional[str]:
        """Find best matching enrolled face based on embedding similarity."""
        min_distance = float('inf')
        best_match = None
        
        for enrolled_id, enrolled_embedding in enrolled_faces.items():
            similarity = np.dot(face_embedding, enrolled_embedding)
            distance = 1 - similarity
            
            if distance < min_distance and distance < self.embedding_similarity_threshold:
                min_distance = distance
                best_match = enrolled_id
        
        return best_match
    
    def _find_best_spatial_match(self, current_center: Tuple[float, float], 
                                face_tracks: Dict, matched_tracks: set) -> Optional[str]:
        """Find best matching face track based on spatial distance."""
        min_distance = float('inf')
        best_match = None
        
        for track_id, track_data in face_tracks.items():
            if track_id in matched_tracks:
                continue
            
            last_detection = track_data[-1]
            last_center = last_detection.get('bbox_center')
            if last_center is None:
                continue
            
            distance = self.calculate_distance(current_center, last_center)
            if distance < min_distance and distance < self.max_distance_threshold:
                min_distance = distance
                best_match = track_id
        
        return best_match
    
    def _create_detection_entry(self, frame_result: Dict, face: Dict, 
                              enrollment_data: Dict = None, verification_data: Dict = None) -> Dict:
        """Create a standardized detection entry for face tracks."""
        entry = {
            'frame_index': frame_result['frame_index'],
            'timestamp': frame_result['timestamp'],
            'face_data': face,
            'bbox_center': self.get_bbox_center(face['bbox']) if face['bbox'] else None
        }
        
        if enrollment_data:
            entry['enrollment'] = enrollment_data
        if verification_data:
            entry['verification'] = verification_data
        
        return entry
    
    def _calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity score between two face embeddings."""
        return float(np.dot(embedding1, embedding2))
    
    def _is_better_quality(self, new_face: Dict, reference_detection: Dict) -> bool:
        """Check if new face has better quality than reference."""
        new_confidence = new_face.get('confidence', 0)
        ref_confidence = reference_detection.get('face_data', {}).get('confidence', 0)
        return new_confidence > ref_confidence
    
    def _add_enrollment_statistics(self, face_tracks: Dict):
        """Add enrollment/verification statistics to all face tracks."""
        for track_id, detections in face_tracks.items():
            enrollment_info = next((d.get('enrollment') for d in detections if 'enrollment' in d), None)
            if enrollment_info:
                for detection in detections:
                    detection['enrolled_id'] = track_id
                    if 'enrollment' not in detection:
                        detection['verification'] = {
                            'verified_id': track_id,
                            'is_enrolled_face': True
                        }
    
    @staticmethod
    def get_bbox_center(bbox: List[int]) -> Optional[Tuple[float, float]]:
        """Calculate center point of bounding box."""
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def get_bbox_size(bbox: List[int]) -> Tuple[float, float]:
        """Calculate width and height of bounding box."""
        if not bbox or len(bbox) != 4:
            return (0, 0)
        x1, y1, x2, y2 = bbox
        return (x2 - x1, y2 - y1)


class FaceStatisticsCalculator:
    """
    Calculator for detailed face statistics across tracking sequences.
    """
    
    def __init__(self, default_image_width: int = 1920, default_image_height: int = 1080):
        """
        Initialize statistics calculator.
        
        Args:
            default_image_width: Default image width for calculations
            default_image_height: Default image height for calculations
        """
        self.default_image_width = default_image_width
        self.default_image_height = default_image_height
    
    def calculate_face_statistics(self, face_tracks: Dict[str, List[Dict]], 
                                frame_results: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate detailed statistics for each tracked face.
        
        Args:
            face_tracks: Dictionary of face tracks across frames
            frame_results: List of frame results for context
            
        Returns:
            Dictionary containing detailed statistics for each face
        """
        face_statistics = {}
        image_center = self._estimate_image_center(frame_results)
        
        for track_id, detections in face_tracks.items():
            if not detections:
                continue
            
            statistics = self._calculate_single_face_statistics(detections, frame_results, image_center)
            face_statistics[track_id] = statistics
        
        return face_statistics
    
    def _estimate_image_center(self, frame_results: List[Dict]) -> Tuple[float, float]:
        """Estimate image center from face detection data."""
        image_width, image_height = self.default_image_width, self.default_image_height
        
        # Try to infer from first frame with faces
        for frame_result in frame_results:
            if frame_result['faces']:
                first_face = frame_result['faces'][0]
                if first_face['bbox']:
                    # Rough estimation - assume face is about 1/8 of image width
                    face_width = first_face['bbox'][2] - first_face['bbox'][0]
                    image_width = max(self.default_image_width, face_width * 8)
                    image_height = int(image_width * 9 / 16)  # Assume 16:9 aspect ratio
                break
        
        return (image_width / 2, image_height / 2)
    
    def _calculate_single_face_statistics(self, detections: List[Dict], 
                                        frame_results: List[Dict], 
                                        image_center: Tuple[float, float]) -> Dict:
        """Calculate comprehensive statistics for a single face track."""
        # Extract data arrays for analysis
        data_arrays = self._extract_data_arrays(detections, image_center)
        
        # Calculate basic statistics
        basic_stats = self._calculate_basic_statistics(detections, frame_results, data_arrays)
        
        # Calculate derived statistics
        derived_stats = self._calculate_derived_statistics(data_arrays, detections)
        
        # Combine all statistics
        statistics = {
            **basic_stats,
            'bbox_size_stats': self._calculate_bbox_statistics(data_arrays['bbox_sizes']),
            'position_stats': self._calculate_position_statistics(data_arrays),
            'pose_stats': self._calculate_pose_statistics(data_arrays['pose_data']),
            'derived_stats': derived_stats
        }
        
        # Add quality score
        statistics['derived_stats']['face_quality_score'] = self._calculate_face_quality_score(statistics)
        
        return statistics
    
    def _extract_data_arrays(self, detections: List[Dict], image_center: Tuple[float, float]) -> Dict:
        """Extract data arrays from detection sequence."""
        bbox_sizes = []
        pose_data = {'roll': [], 'pitch': [], 'yaw': [], 'pose_deviation': []}
        confidences = []
        timestamps = []
        distances_from_center = []
        movement_distances = []
        total_movement_distance = 0.0
        previous_center = None
        
        for detection in detections:
            face_data = detection['face_data']
            
            # Bbox size statistics
            if face_data['bbox']:
                width, height = FaceTracker.get_bbox_size(face_data['bbox'])
                bbox_sizes.append((width, height))
                
                # Current center position
                current_center = detection['bbox_center']
                if current_center:
                    # Distance from image center
                    distance_from_center = FaceTracker.calculate_distance(current_center, image_center)
                    distances_from_center.append(distance_from_center)
                    
                    # Movement distance calculation
                    if previous_center is not None:
                        movement_dist = FaceTracker.calculate_distance(current_center, previous_center)
                        movement_distances.append(movement_dist)
                        total_movement_distance += movement_dist
                    
                    previous_center = current_center
            
            # Pose statistics
            if face_data['pose']['valid']:
                pose = face_data['pose']
                pose_data['roll'].append(pose.get('roll', 0))
                pose_data['pitch'].append(pose.get('pitch', 0))
                pose_data['yaw'].append(pose.get('yaw', 0))
                pose_data['pose_deviation'].append(pose.get('pose_deviation', 0))
            
            # Confidence and time
            confidences.append(face_data['confidence'])
            timestamps.append(detection['timestamp'])
        
        return {
            'bbox_sizes': bbox_sizes,
            'pose_data': pose_data,
            'confidences': confidences,
            'timestamps': timestamps,
            'distances_from_center': distances_from_center,
            'movement_distances': movement_distances,
            'total_movement_distance': total_movement_distance
        }
    
    def _calculate_basic_statistics(self, detections: List[Dict], frame_results: List[Dict], 
                                  data_arrays: Dict) -> Dict:
        """Calculate basic tracking statistics."""
        timestamps = data_arrays['timestamps']
        confidences = data_arrays['confidences']
        
        return {
            'total_detections': len(detections),
            'first_appearance': timestamps[0] if timestamps else 0,
            'last_appearance': timestamps[-1] if timestamps else 0,
            'duration_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'detection_consistency': len(detections) / len(frame_results) if frame_results else 0,
            'confidence_stats': {
                'average': np.mean(confidences) if confidences else 0,
                'median': np.median(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0
            }
        }
    
    def _calculate_bbox_statistics(self, bbox_sizes: List[Tuple[float, float]]) -> Dict:
        """Calculate bounding box size statistics."""
        if not bbox_sizes:
            return {}
        
        widths = [size[0] for size in bbox_sizes]
        heights = [size[1] for size in bbox_sizes]
        areas = [w * h for w, h in bbox_sizes]
        
        return {
            'width': self._calculate_stats_dict(widths),
            'height': self._calculate_stats_dict(heights),
            'area': self._calculate_stats_dict(areas)
        }
    
    def _calculate_position_statistics(self, data_arrays: Dict) -> Dict:
        """Calculate position and movement statistics."""
        distances_from_center = data_arrays['distances_from_center']
        movement_distances = data_arrays['movement_distances']
        total_movement_distance = data_arrays['total_movement_distance']
        
        if not distances_from_center:
            return {}
        
        return {
            'distance_from_center': self._calculate_stats_dict(distances_from_center),
            'total_movement_distance': total_movement_distance,
            'average_movement_per_frame': np.mean(movement_distances) if movement_distances else 0,
            'max_movement_between_frames': max(movement_distances) if movement_distances else 0,
            'movement_variance': np.var(movement_distances) if movement_distances else 0,
            'is_stationary': np.mean(movement_distances) < 10 if movement_distances else True
        }
    
    def _calculate_pose_statistics(self, pose_data: Dict) -> Dict:
        """Calculate pose-related statistics."""
        pose_stats = {}
        for pose_type, values in pose_data.items():
            if values:
                pose_stats[pose_type] = {
                    **self._calculate_stats_dict(values),
                    'values': values  # Keep raw values for detailed analysis
                }
        return pose_stats
    
    def _calculate_derived_statistics(self, data_arrays: Dict, detections: List[Dict]) -> Dict:
        """Calculate derived/composite statistics."""
        bbox_sizes = data_arrays['bbox_sizes']
        pose_data = data_arrays['pose_data']
        
        return {
            'average_face_size_pixels': np.mean([w * h for w, h in bbox_sizes]) if bbox_sizes else 0,
            'face_size_consistency': self._calculate_size_consistency(bbox_sizes),
            'is_looking_at_camera': np.mean(pose_data.get('pose_deviation', [90])) < 30 if pose_data.get('pose_deviation') else False,
            'pose_stability': self._calculate_pose_stability(pose_data),
            'primary_face': len(detections) == max([len(track) for track in [detections]]),  # Placeholder for max comparison
        }
    
    def _calculate_stats_dict(self, values: List[float]) -> Dict:
        """Calculate standard statistics for a list of values."""
        if not values:
            return {'average': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
        
        return {
            'average': np.mean(values),
            'median': np.median(values),
            'min': min(values),
            'max': max(values),
            'std': np.std(values)
        }
    
    def _calculate_size_consistency(self, bbox_sizes: List[Tuple[float, float]]) -> float:
        """Calculate face size consistency score."""
        if len(bbox_sizes) <= 1:
            return 1.0
        
        areas = [w * h for w, h in bbox_sizes]
        mean_area = np.mean(areas)
        if mean_area == 0:
            return 1.0
        
        return 1 - (np.std(areas) / mean_area)
    
    def _calculate_pose_stability(self, pose_data: Dict) -> float:
        """Calculate pose stability score."""
        pose_deviation_values = pose_data.get('pose_deviation', [0])
        if not pose_deviation_values:
            return 0.0
        
        return 1 - (np.std(pose_deviation_values) / 180)
    
    def _calculate_face_quality_score(self, face_stats: Dict) -> float:
        """Calculate overall quality score for a face based on various metrics."""
        score = 0.0
        weight_sum = 0.0
        
        # Confidence score (weight: 0.3)
        if 'confidence_stats' in face_stats:
            confidence_score = face_stats['confidence_stats']['average']
            score += confidence_score * 0.3
            weight_sum += 0.3
        
        # Detection consistency (weight: 0.2)
        consistency_score = face_stats.get('detection_consistency', 0)
        score += consistency_score * 0.2
        weight_sum += 0.2
        
        # Face size (weight: 0.2) - prefer larger faces
        if 'bbox_size_stats' in face_stats and 'area' in face_stats['bbox_size_stats']:
            avg_area = face_stats['bbox_size_stats']['area']['average']
            size_score = min(1.0, avg_area / 10000)  # Normalize area score
            score += size_score * 0.2
            weight_sum += 0.2
        
        # Pose quality (weight: 0.15) - prefer faces looking at camera
        if 'derived_stats' in face_stats:
            if face_stats['derived_stats']['is_looking_at_camera']:
                score += 1.0 * 0.15
            else:
                score += 0.5 * 0.15
            weight_sum += 0.15
        
        # Stability (weight: 0.15) - prefer stable detections
        if 'derived_stats' in face_stats:
            pose_stability = face_stats['derived_stats'].get('pose_stability', 0)
            face_size_consistency = face_stats['derived_stats'].get('face_size_consistency', 0)
            stability_score = (pose_stability + face_size_consistency) / 2
            score += stability_score * 0.15
            weight_sum += 0.15
        
        return score / weight_sum if weight_sum > 0 else 0.0