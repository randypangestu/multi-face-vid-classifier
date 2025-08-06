#!/usr/bin/env python3
"""
Visualization Utilities Module

This module provides visualization capabilities for face detection results,
including video overlay generation and pose visualization.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from .pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)


class FaceVisualizationRenderer:
    """
    Renderer for face detection visualizations with adaptive styling.
    """
    
    def __init__(self):
        """Initialize visualization renderer."""
        self.pose_estimator = PoseEstimator()
        
        # Color schemes
        self.bbox_color = (0, 255, 0)
        self.landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        self.landmark_names = ['L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth']
        
        # Pose color mapping
        self.pose_colors = {
            'straight': (0, 255, 0),      # Green - looking straight
            'slight_turn': (0, 255, 255), # Yellow - slight turn
            'turned_away': (0, 100, 255)  # Orange/Red - turned away
        }
    
    def render_face_detections(self, frame: np.ndarray, frame_result: Dict,
                              frame_idx: int, timestamp: float) -> np.ndarray:
        """
        Render face detections on a single frame.
        
        Args:
            frame: Input frame as numpy array
            frame_result: Frame detection results
            frame_idx: Frame index for display
            timestamp: Frame timestamp
            
        Returns:
            Frame with rendered detections
        """
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # Render each detected face
        for face in frame_result['faces']:
            self._render_single_face(frame_copy, face, width, height)
        
        # Add frame information
        self._render_frame_info(frame_copy, frame_idx, frame_result['num_faces'], timestamp, height)
        
        return frame_copy
    
    def _render_single_face(self, frame: np.ndarray, face: Dict, width: int, height: int):
        """Render a single face detection with all overlays."""
        if not face['bbox']:
            return
        
        x1, y1, x2, y2 = face['bbox']
        face_size = min(x2 - x1, y2 - y1)
        
        # Adaptive styling based on face size
        font_scale, thickness, box_thickness = self._get_adaptive_styling(face_size)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, box_thickness)
        
        # Draw confidence score
        self._draw_confidence_score(frame, face, x1, y1, font_scale, thickness)
        
        # Draw pose information
        if face['pose']['valid']:
            self._draw_pose_information(frame, face, x1, y1, y2, font_scale, thickness)
            self._draw_pose_axes(frame, face, width, height, face_size)
        
        # Draw landmarks
        if face['landmarks']:
            self._draw_landmarks(frame, face['landmarks'], face_size)
        
        # Draw face ID
        self._draw_face_id(frame, face, x1, y1, x2, font_scale, thickness)
    
    def _get_adaptive_styling(self, face_size: int) -> Tuple[float, int, int]:
        """Get adaptive styling parameters based on face size."""
        base_font_scale = 0.6
        
        if face_size < 80:
            font_scale = base_font_scale * 1.2
            thickness = 2
            box_thickness = 3
        elif face_size < 150:
            font_scale = base_font_scale * 1.0
            thickness = 2
            box_thickness = 2
        else:
            font_scale = base_font_scale * 0.8
            thickness = 1
            box_thickness = 2
        
        return font_scale, thickness, box_thickness
    
    def _draw_confidence_score(self, frame: np.ndarray, face: Dict, x1: int, y1: int,
                              font_scale: float, thickness: int):
        """Draw confidence score with background."""
        conf_text = f"Conf: {face['confidence']:.2f}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Draw text background
        cv2.rectangle(frame, (x1, y1-text_size[1]-10), (x1+text_size[0]+5, y1), (0, 0, 0), -1)
        cv2.putText(frame, conf_text, (x1+2, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.bbox_color, thickness)
    
    def _draw_pose_information(self, frame: np.ndarray, face: Dict, x1: int, y1: int, y2: int,
                              font_scale: float, thickness: int):
        """Draw comprehensive pose information."""
        pose = face['pose']
        
        # Get pose angles and deviation
        roll_abs = pose['roll']
        pitch_abs = pose['pitch']
        yaw_abs = pose['yaw']
        pose_deviation = pose.get('pose_deviation', 0)
        
        # Draw main pose angles
        pose_text = f"R:{roll_abs:.0f}° P:{pitch_abs:.0f}° Y:{yaw_abs:.0f}°"
        pose_color = self._get_pose_color(pose_deviation)
        
        pose_y = y2 + 25
        self._draw_text_with_background(frame, pose_text, (x1, pose_y), font_scale, pose_color, thickness)
        
        # Draw pose deviation
        deviation_text = f"Dev:{pose_deviation:.0f}°"
        dev_y = pose_y + 25
        self._draw_text_with_background(frame, deviation_text, (x1, dev_y), 
                                       font_scale*0.8, pose_color, thickness)
        
        # Draw pose interpretation
        pose_interpretation = self._get_pose_interpretation(roll_abs, pitch_abs, yaw_abs, pose_deviation)
        if pose_interpretation:
            interp_y = dev_y + 25
            self._draw_text_with_background(frame, pose_interpretation, (x1, interp_y), 
                                           font_scale*0.7, (0, 255, 255), thickness)
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: List[List[int]], face_size: int):
        """Draw facial landmarks with adaptive sizing."""
        landmark_radius = max(2, min(6, face_size // 25))
        
        for idx, landmark in enumerate(landmarks):
            color = self.landmark_colors[idx % len(self.landmark_colors)]
            cv2.circle(frame, tuple(landmark), landmark_radius, color, -1)
            
            # Add landmark labels for larger faces
            if face_size > 100:
                cv2.putText(frame, self.landmark_names[idx], 
                           (landmark[0]+5, landmark[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_face_id(self, frame: np.ndarray, face: Dict, x1: int, y1: int, x2: int,
                     font_scale: float, thickness: int):
        """Draw face ID with background."""
        face_id_text = f"ID: {face['face_id']}"
        id_text_size = cv2.getTextSize(face_id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Position at top-right of face
        id_x = x2 - id_text_size[0] - 5
        cv2.rectangle(frame, (id_x-2, y1-id_text_size[1]-10), (x2, y1), (0, 0, 0), -1)
        cv2.putText(frame, face_id_text, (id_x, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    
    def _draw_text_with_background(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                                  font_scale: float, color: Tuple[int, int, int], thickness: int):
        """Draw text with black background for better visibility."""
        x, y = position
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Draw background
        cv2.rectangle(frame, (x, y-text_size[1]-5), (x+text_size[0]+5, y+5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(frame, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def _render_frame_info(self, frame: np.ndarray, frame_idx: int, num_faces: int,
                          timestamp: float, height: int):
        """Render frame information and legend."""
        # Frame info
        info_text = f"Frame: {frame_idx}, Faces: {num_faces}"
        cv2.rectangle(frame, (5, 5), (400, 45), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Timestamp
        time_text = f"Time: {timestamp:.2f}s"
        cv2.putText(frame, time_text, (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Pose legend
        legend_text = "Pose: 0°=Straight, 180°=Max Turn"
        cv2.putText(frame, legend_text, (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _get_pose_color(self, pose_deviation: float) -> Tuple[int, int, int]:
        """Get color based on pose deviation."""
        if pose_deviation < 20:
            return self.pose_colors['straight']
        elif pose_deviation < 45:
            return self.pose_colors['slight_turn']
        else:
            return self.pose_colors['turned_away']
    
    def _get_pose_interpretation(self, roll: float, pitch: float, yaw: float, 
                                pose_deviation: float = None) -> str:
        """Get human-readable interpretation of absolute pose angles."""
        interpretations = []
        
        # Overall pose assessment
        if pose_deviation is not None:
            if pose_deviation < 10:
                interpretations.append("Straight On")
            elif pose_deviation < 30:
                interpretations.append("Slight Turn")
            elif pose_deviation < 60:
                interpretations.append("Moderate Turn")
            else:
                interpretations.append("Strong Turn")
        
        # Individual angle interpretations
        angle_descriptions = []
        
        # Yaw interpretation (left/right turn from straight)
        if yaw < 15:
            angle_descriptions.append("Forward")
        elif yaw < 45:
            angle_descriptions.append("Turn")
        else:
            angle_descriptions.append("Profile")
        
        # Pitch interpretation (up/down from straight)
        if pitch > 30:
            angle_descriptions.append("Up/Down")
        
        # Roll interpretation (tilt from straight) 
        if roll > 30:
            angle_descriptions.append("Tilted")
        
        if angle_descriptions:
            interpretations.extend(angle_descriptions)
        
        return ", ".join(interpretations) if interpretations else "Neutral"
    
    def _draw_pose_axes(self, frame: np.ndarray, face: Dict, image_width: int, 
                       image_height: int, face_size: int):
        """Draw 3D pose axes on the face."""
        pose = face['pose']
        landmarks = face['landmarks']
        
        if not pose['valid'] or 'rotation_vector' not in pose or 'translation_vector' not in pose:
            return
        
        try:
            # Set camera matrix
            self.pose_estimator.set_camera_matrix(image_width, image_height)
            
            # Adaptive axis length based on face size
            axis_length = max(30, min(80, face_size * 0.8))
            
            # 3D axes points (in mm) - scaled based on face size
            axes_points = np.array([
                (0, 0, 0),                    # Origin
                (axis_length, 0, 0),          # X-axis (red)
                (0, axis_length, 0),          # Y-axis (green)  
                (0, 0, -axis_length)          # Z-axis (blue)
            ], dtype=np.float64)
            
            # Project 3D axes points to 2D
            rotation_vector = np.array(pose['rotation_vector'], dtype=np.float64)
            translation_vector = np.array(pose['translation_vector'], dtype=np.float64)
            
            projected_points, _ = cv2.projectPoints(
                axes_points,
                rotation_vector,
                translation_vector,
                self.pose_estimator.camera_matrix,
                self.pose_estimator.dist_coeffs
            )
            
            # Convert to integer coordinates
            projected_points = projected_points.reshape(-1, 2).astype(int)
            
            # Get nose tip as origin
            nose_tip = landmarks[2] if len(landmarks) > 2 else projected_points[0]
            
            # Adaptive line thickness based on face size
            line_thickness = max(2, min(5, face_size // 30))
            
            # Draw axes with labels
            self._draw_axis_line(frame, nose_tip, projected_points[1], (0, 0, 255), line_thickness, 'X')
            self._draw_axis_line(frame, nose_tip, projected_points[2], (0, 255, 0), line_thickness, 'Y')
            self._draw_axis_line(frame, nose_tip, projected_points[3], (255, 0, 0), line_thickness, 'Z')
            
        except Exception as e:
            logger.debug(f"Failed to draw pose axes: {str(e)}")
    
    def _draw_axis_line(self, frame: np.ndarray, start_point: Tuple[int, int], 
                       end_point: Tuple[int, int], color: Tuple[int, int, int], 
                       thickness: int, label: str):
        """Draw a single axis line with label."""
        cv2.arrowedLine(frame, tuple(start_point), tuple(end_point), color, thickness)
        cv2.putText(frame, label, tuple(end_point + [5, 5]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class VideoVisualizationProcessor:
    """
    Processor for creating visualization videos from detection results.
    """
    
    def __init__(self):
        """Initialize video visualization processor."""
        self.renderer = FaceVisualizationRenderer()
    
    def create_visualization_video(self, video_path: str, results: Dict[str, Any], 
                                 output_video_path: str = None, 
                                 max_frames_to_save: int = 1000) -> str:
        """
        Create visualization video with face detection overlays.
        
        Args:
            video_path: Original video path
            results: Detection results
            output_video_path: Path to save visualization video
            max_frames_to_save: Maximum frames to include in output video
            
        Returns:
            Path to the created visualization video
        """
        if output_video_path is None:
            output_video_path = str(Path(video_path).with_suffix('_detections.mp4'))
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        try:
            frame_count = 0
            for frame_result in results['frame_results']:
                if frame_count >= max_frames_to_save:
                    break
                
                frame_idx = frame_result['frame_index']
                timestamp = frame_result.get('timestamp', 0)
                
                # Read frame from video
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Render detections on frame
                rendered_frame = self.renderer.render_face_detections(
                    frame, frame_result, frame_idx, timestamp
                )
                
                out.write(rendered_frame)
                frame_count += 1
            
            logger.info(f"Visualization saved to: {output_video_path}")
            
        finally:
            cap.release()
            out.release()
        
        return output_video_path


class ResultsAnalyzer:
    """
    Analyzer for generating comprehensive detection result summaries.
    """
    
    def analyze_detection_results(self, frame_results: List[Dict], video_path: str) -> Dict[str, Any]:
        """
        Analyze face detection results across all frames.
        
        Args:
            frame_results: List of per-frame detection results
            video_path: Path to the video file
            
        Returns:
            Dictionary containing analysis summary and detailed face statistics
        """
        if not frame_results:
            return self._create_empty_analysis()
        
        # Calculate basic statistics
        basic_stats = self._calculate_basic_statistics(frame_results)
        
        # Determine classification
        classification_data = self._classify_detection_results(basic_stats)
        
        # Calculate confidence
        confidence = self._calculate_confidence(basic_stats, frame_results)
        
        return {
            **basic_stats,
            **classification_data,
            'confidence': confidence,
            'transaction_id': Path(video_path).stem,
        }
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create analysis for videos with no face detections."""
        return {
            'summary': 'no_faces_detected',
            'total_faces_detected': 0,
            'avg_faces_per_frame': 0,
            'max_faces_in_frame': 0,
            'frames_with_faces': 0,
            'face_detection_rate': 0.0,
            'classification': 'no_person',
            'binary_label': -1,
            'confidence': 0.0
        }
    
    def _calculate_basic_statistics(self, frame_results: List[Dict]) -> Dict[str, Any]:
        """Calculate basic detection statistics."""
        face_counts = [result['num_faces'] for result in frame_results]
        total_faces = sum(face_counts)
        frames_with_faces = sum(1 for count in face_counts if count > 0)
        avg_faces = total_faces / len(frame_results) if frame_results else 0
        max_faces = max(face_counts) if face_counts else 0
        detection_rate = frames_with_faces / len(frame_results) if frame_results else 0
        
        return {
            'total_faces_detected': total_faces,
            'avg_faces_per_frame': avg_faces,
            'max_faces_in_frame': max_faces,
            'frames_with_faces': frames_with_faces,
            'total_frames_processed': len(frame_results),
            'face_detection_rate': detection_rate,
        }
    
    def _classify_detection_results(self, basic_stats: Dict) -> Dict[str, Any]:
        """Classify detection results based on face patterns."""
        total_faces = basic_stats['total_faces_detected']
        avg_faces = basic_stats['avg_faces_per_frame']
        max_faces = basic_stats['max_faces_in_frame']
        
        if total_faces == 0:
            summary = 'no_faces_detected'
            classification = 'no_person'
            binary_label = -1
        elif avg_faces <= 1.2 and max_faces <= 2:
            summary = 'single_person_detected'
            classification = 'one_person'
            binary_label = 0
        else:
            summary = 'multiple_persons_detected'
            classification = 'more_than_one_person'
            binary_label = 1
        
        return {
            'summary': summary,
            'classification': classification,
            'binary_label': binary_label
        }
    
    def _calculate_confidence(self, basic_stats: Dict, frame_results: List[Dict]) -> float:
        """Calculate confidence based on detection consistency."""
        detection_rate = basic_stats['face_detection_rate']
        frames_with_faces = basic_stats['frames_with_faces']
        total_frames = len(frame_results)
        
        # Base confidence on detection consistency
        confidence = min(0.95, detection_rate * 0.8 + (frames_with_faces / total_frames) * 0.2)
        return confidence