#!/usr/bin/env python3
"""
3D Head Pose Estimation Module

This module provides pose estimation functionality using facial landmarks
and PnP (Perspective-n-Point) algorithm with OpenCV.
"""

import cv2
import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    3D head pose estimation from facial landmarks.
    Calculates roll, pitch, and yaw angles using 2D facial landmarks.
    """
    
    def __init__(self):
        # 3D model points of a generic face (in mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (will be updated based on image size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
    def set_camera_matrix(self, image_width: int, image_height: int):
        """Set camera matrix based on image dimensions."""
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def landmarks_to_pose_points(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convert 5-point landmarks to 6 points used for pose estimation.
        
        Args:
            landmarks: 5x2 array of facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
            
        Returns:
            6x2 array of 2D points corresponding to the 3D model points
        """
        if landmarks.shape[0] != 5:
            raise ValueError("Expected 5 facial landmarks")
        
        # Map the 5 landmarks to 6 pose estimation points
        left_eye = landmarks[0]
        right_eye = landmarks[1] 
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Estimate chin position (below the mouth points)
        mouth_center = (left_mouth + right_mouth) / 2
        chin = mouth_center + np.array([0, 20])  # Approximate chin position
        
        # Create the 6 points for pose estimation
        pose_points = np.array([
            nose,        # Nose tip
            chin,        # Chin
            left_eye,    # Left eye (mapped to left eye corner)
            right_eye,   # Right eye (mapped to right eye corner)
            left_mouth,  # Left mouth corner
            right_mouth  # Right mouth corner
        ], dtype=np.float64)
        
        return pose_points
    
    def estimate_pose(self, landmarks: np.ndarray, image_width: int, image_height: int) -> Dict[str, float]:
        """
        Estimate head pose from facial landmarks.
        
        Args:
            landmarks: 5x2 array of facial landmarks
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Dictionary containing roll, pitch, yaw angles in absolute degrees (0-180)
        """
        if self.camera_matrix is None or self.camera_matrix[0, 2] != image_width / 2:
            self.set_camera_matrix(image_width, image_height)
        
        try:
            # Convert landmarks to pose estimation points
            image_points = self.landmarks_to_pose_points(landmarks)
            
            # Solve PnP to get rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return {'roll': 90.0, 'pitch': 90.0, 'yaw': 90.0, 'valid': False}
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles from rotation matrix
            euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Convert to absolute values (0-180 degrees)
            roll_abs, pitch_abs, yaw_abs = self._convert_to_absolute_angles(euler_angles)
            
            # Calculate overall pose deviation (0 = straight at camera, 180 = maximum deviation)
            pose_deviation = self._calculate_pose_deviation(roll_abs, pitch_abs, yaw_abs)
            
            return {
                'roll': roll_abs,
                'pitch': pitch_abs, 
                'yaw': yaw_abs,
                'pose_deviation': pose_deviation,
                'valid': True,
                'rotation_vector': rotation_vector.flatten().tolist(),
                'translation_vector': translation_vector.flatten().tolist(),
                # Keep original signed values for reference
                'roll_signed': euler_angles[0],
                'pitch_signed': euler_angles[1],
                'yaw_signed': euler_angles[2]
            }
            
        except Exception as e:
            logger.warning(f"Failed to estimate pose: {str(e)}")
            return {'roll': 90.0, 'pitch': 90.0, 'yaw': 90.0, 'pose_deviation': 90.0, 'valid': False}
    
    def _rotation_matrix_to_euler_angles(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles using ZYX convention (yaw-pitch-roll).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        # Using ZYX convention (yaw-pitch-roll)
        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                      rotation_matrix[1, 0] * rotation_matrix[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Roll
            y = math.atan2(-rotation_matrix[2, 0], sy)                    # Pitch
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Yaw
        else:
            x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])  # Roll
            y = math.atan2(-rotation_matrix[2, 0], sy)                     # Pitch
            z = 0                                                          # Yaw
        
        # Convert from radians to degrees
        roll_deg = math.degrees(x)
        pitch_deg = math.degrees(y)
        yaw_deg = math.degrees(z)
        
        return roll_deg, pitch_deg, yaw_deg
    
    def _convert_to_absolute_angles(self, euler_angles: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert signed Euler angles to absolute values (0-180 degrees).
        
        Args:
            euler_angles: Tuple of (roll, pitch, yaw) in degrees
            
        Returns:
            Tuple of absolute angles where 0 = looking straight at camera, 180 = maximum deviation
        """
        roll_deg, pitch_deg, yaw_deg = euler_angles
        
        # Convert to absolute values (0-180 degrees)
        roll_abs = 180 - abs(roll_deg)
        pitch_abs = abs(pitch_deg)
        yaw_abs = 180 - abs(yaw_deg)
        
        return roll_abs, pitch_abs, yaw_abs
    
    def _calculate_pose_deviation(self, roll_abs: float, pitch_abs: float, yaw_abs: float) -> float:
        """
        Calculate overall pose deviation from straight-on view.
        
        Args:
            roll_abs: Absolute roll angle (0-180)
            pitch_abs: Absolute pitch angle (0-180)
            yaw_abs: Absolute yaw angle (0-180)
            
        Returns:
            Overall pose deviation score (0-180)
        """
        # Using weighted average where yaw (left/right) is most important for "looking at camera"
        pose_deviation = (yaw_abs * 0.5 + pitch_abs * 0.3 + roll_abs * 0.2)
        return min(180.0, pose_deviation)