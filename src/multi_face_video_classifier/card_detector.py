#!/usr/bin/env python3
"""
Card Detector using Grounding DINO.

Zero-shot ID card detection using IDEA-Research/grounding-dino-tiny.
Provides a clean interface for detecting ID cards in images with various input formats.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Configure module logger
logger = logging.getLogger(__name__)


class CardDetector:
    """
    ID Card detector using Grounding DINO zero-shot object detection.
    
    This class provides a simple interface for detecting ID cards in images using
    the Grounding DINO model with a fixed text query of 'id card'.
    
    Attributes:
        DEFAULT_MODEL_ID: Default Hugging Face model identifier.
        DEFAULT_BOX_THRESHOLD: Default box confidence threshold.
        DEFAULT_TEXT_THRESHOLD: Default text confidence threshold.
        CARD_QUERY: Fixed text query for ID card detection.
    """
    
    DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    DEFAULT_BOX_THRESHOLD = 0.4
    DEFAULT_TEXT_THRESHOLD = 0.2
    CARD_QUERY_DICT = {"others": ["identity card."],
                      "rc5": ["identity card.", "card."],
                      "rc6": ["identity card.", "card."],}
    #CARD_QUERY = "id card not window"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        mode: str = "rc6",
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    ) -> None:
        """Initialize the Card Detector.
        
        Args:
            model_id: Hugging Face model ID for Grounding DINO.
            device: Device to use ('cuda', 'cpu', or 'auto').
            box_threshold: Box confidence threshold.
            text_threshold: Text confidence threshold.
            
        Raises:
            RuntimeError: If model initialization fails.
        """
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.mode = 'others' if mode not in self.CARD_QUERY_DICT else mode
        # Set device
        print('video self.mode', self.mode)
        self.device = self._determine_device(device)
        
        # Initialize model components
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForZeroShotObjectDetection] = None
        
        # Initialize the model
        self._initialize_model()
        
        logger.info(
            "CardDetector initialized on %s (box_threshold=%.2f, text_threshold=%.2f)",
            self.device,
            self.box_threshold,
            self.text_threshold,
        )
    
    @staticmethod
    def _determine_device(requested_device: str) -> str:
        """Determine the appropriate device for inference.
        
        Args:
            requested_device: Requested device ('cuda', 'cpu', or 'auto').
            
        Returns:
            The actual device to use.
        """
        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested_device
    
    def _initialize_model(self) -> None:
        """Initialize the Grounding DINO model and processor.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            logger.info("Loading Grounding DINO model: %s", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as exc:
            error_msg = f"Failed to load model: {exc}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various input formats.
        
        Args:
            image_input: Can be file path, URL, numpy array, or PIL Image.
            
        Returns:
            PIL Image in RGB format.
            
        Raises:
            ValueError: If image input type is not supported.
            requests.RequestException: If URL loading fails.
            IOError: If file loading fails.
        """
        if isinstance(image_input, str):
            return self._load_image_from_path_or_url(image_input)
        elif isinstance(image_input, np.ndarray):
            return self._load_image_from_numpy(image_input)
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _load_image_from_path_or_url(self, path_or_url: str) -> Image.Image:
        """Load image from file path or URL.
        
        Args:
            path_or_url: File path or URL to the image.
            
        Returns:
            PIL Image in RGB format.
        """
        if path_or_url.startswith(("http://", "https://")):
            response = requests.get(path_or_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            image = Image.open(path_or_url)
        return image.convert("RGB")
    
    def _load_image_from_numpy(self, image_array: np.ndarray) -> Image.Image:
        """Load image from numpy array.
        
        Args:
            image_array: Numpy array representing the image.
            
        Returns:
            PIL Image in RGB format.
        """
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Assume BGR format from OpenCV and convert to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        else:
            return Image.fromarray(image_array)
    
    def detect(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Union[List[float], float, str]]]:
        """Detect ID cards in the given image.
        
        Args:
            image_input: Image to process (file path, URL, numpy array, or PIL Image).
            
        Returns:
            List of detection results, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - label: Always 'id card'
                - query_index: Index of the query that detected this box
        """
        try:
            # Load and prepare image
            image = self._load_image(image_input)
            
            all_detections = []
            
            # Run inference for each query
            for query_idx, query in enumerate(self.CARD_QUERY_DICT[self.mode]):
                inputs = self.processor(
                    images=image, text=query, return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process results
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[image.size[::-1]],
                )[0]
                
                # Format results for this query
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    bbox = [round(coord, 2) for coord in box.tolist()]
                    detection = {
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "confidence": float(score),
                        "label": "card",
                        "query_index": query_idx,
                        "query_text": query
                    }
                    all_detections.append(detection)
            
            # Combine overlapping detections
            combined_detections = self._combine_overlapping_detections(all_detections, iou_threshold=0.7)
            # Return only bounding boxes as lists
            bboxes = [detection["bbox"] for detection in combined_detections]
            return bboxes
            
        except Exception as exc:
            logger.error("Detection failed: %s", exc)
            return []
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_intersect = max(x1_1, x1_2)
        y1_intersect = max(y1_1, y1_2)
        x2_intersect = min(x2_1, x2_2)
        y2_intersect = min(y2_1, y2_2)
        
        if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
            return 0.0
        
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _combine_overlapping_detections(self, detections: List[Dict], iou_threshold: float = 0.8) -> List[Dict]:
        """Combine overlapping detections based on IoU threshold.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for combining detections
            
        Returns:
            List of combined detections
        """
        if not detections:
            return []
        
        # Sort detections by confidence (highest first)
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        combined = []
        used_indices = set()
        
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
            
            # Start a new group with this detection
            group = [detection]
            used_indices.add(i)
            
            # Find all detections that overlap with this one
            for j, other_detection in enumerate(detections[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                iou = self._calculate_iou(detection["bbox"], other_detection["bbox"])
                if iou >= iou_threshold:
                    group.append(other_detection)
                    used_indices.add(j)
            
            # Combine the group into a single detection
            combined_detection = self._merge_detection_group(group)
            combined.append(combined_detection)
        
        return combined
    
    def _merge_detection_group(self, group: List[Dict]) -> Dict:
        """Merge a group of overlapping detections into a single detection.
        
        Args:
            group: List of detection dictionaries to merge
            
        Returns:
            Merged detection dictionary
        """
        if len(group) == 1:
            return group[0]
        
        # Calculate weighted average of bounding boxes based on confidence
        total_confidence = sum(det["confidence"] for det in group)
        weights = [det["confidence"] / total_confidence for det in group]
        
        # Weighted average of bounding box coordinates
        avg_bbox = [0.0, 0.0, 0.0, 0.0]
        for det, weight in zip(group, weights):
            for i in range(4):
                avg_bbox[i] += det["bbox"][i] * weight
        
        # Use highest confidence and collect query information
        best_detection = max(group, key=lambda x: x["confidence"])
        query_indices = list(set(det["query_index"] for det in group))
        query_texts = list(set(det["query_text"] for det in group))
        
        merged_detection = {
            "bbox": [round(coord, 2) for coord in avg_bbox],
            "confidence": best_detection["confidence"],
            "label": "card",
            "query_index": best_detection["query_index"],
            "query_text": best_detection["query_text"],
            "merged_from": len(group),
            "all_query_indices": query_indices,
            "all_query_texts": query_texts
        }
        
        return merged_detection
    
    def detect_with_visualization(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        output_path: Optional[str] = None,
    ) -> Tuple[List[List[float]], np.ndarray]:
        """Detect ID cards and return results with visualization.
        
        Args:
            image_input: Image to process.
            output_path: Optional path to save visualization.
            
        Returns:
            Tuple of (bbox_list, visualization_image).
        """
        # Get detections (returns list of bboxes)
        bboxes = self.detect(image_input)
        
        # Load image for visualization
        image_pil = self._load_image(image_input)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"ID Card {i+1}"
            cv2.putText(
                image_cv,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image_cv)
            logger.info("Visualization saved to: %s", output_path)
        
        return bboxes, image_cv
    
    def get_best_detection(
        self, image_input: Union[str, np.ndarray, Image.Image]
    ) -> List[float]:
        """Get the best (highest confidence) ID card detection.
        
        Args:
            image_input: Image to process.
            
        Returns:
            Best detection bbox [x1, y1, x2, y2] or empty list if no detections.
        """
        bboxes = self.detect(image_input)
        
        if not bboxes:
            return []
        
        # Since we can't get confidence from the returned bboxes,
        # we'll need to run detection with full details
        detections = self._detect_with_full_details(image_input)
        
        if not detections:
            return []
        
        # Return bbox of detection with highest confidence
        best_detection = max(detections, key=lambda x: x["confidence"])
        return best_detection["bbox"]
    
    def _detect_with_full_details(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[Dict]:
        """Internal method to get full detection details.
        
        Args:
            image_input: Image to process.
            
        Returns:
            List of detection dictionaries with full details.
        """
        try:
            # Load and prepare image
            image = self._load_image(image_input)
            
            all_detections = []
            
            # Run inference for each query
            for query_idx, query in enumerate(self.CARD_QUERY):
                inputs = self.processor(
                    images=image, text=query, return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process results
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[image.size[::-1]],
                )[0]
                
                # Format results for this query
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    bbox = [round(coord, 2) for coord in box.tolist()]
                    detection = {
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "confidence": float(score),
                        "label": "card",
                        "query_index": query_idx,
                        "query_text": query
                    }
                    all_detections.append(detection)
            
            # Combine overlapping detections
            combined_detections = self._combine_overlapping_detections(all_detections, iou_threshold=0.8)
            return combined_detections
            
        except Exception as exc:
            logger.error("Detection failed: %s", exc)
            return []
    
   
    def get_detailed_detections(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[Dict]:
        """Get detailed detection information including query metadata.
        
        Args:
            image_input: Image to process.
            
        Returns:
            List of detailed detection dictionaries with confidence, query info, etc.
        """
        return self._detect_with_full_details(image_input)

