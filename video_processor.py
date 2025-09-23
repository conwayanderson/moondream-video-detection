"""
Video processing module for extracting frames and reconstructing videos with overlays.
"""

import cv2
import os
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


class VideoProcessor:
    """Handles video frame extraction and reconstruction with overlays."""
    
    def __init__(self, video_path: str):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the input video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
    
    def extract_frames(self, output_dir: str = "frames", max_frames: Optional[int] = None, 
                      frames_per_second: Optional[float] = None, duration: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """
        Extract frames from the video.
        
        Args:
            output_dir: Directory to save extracted frames
            max_frames: Maximum number of frames to extract (None for all)
            frames_per_second: How many frames per second to extract
            duration: How many seconds of video to process
        
        Returns:
            Tuple of (frame file paths, frame indices in original video)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        frame_paths = []
        frame_indices = []
        
        # Calculate frame sampling based on parameters
        if frames_per_second is not None:
            # Calculate frame interval based on desired frames per second
            frame_interval = max(1, int(self.fps / frames_per_second))
            
            # Calculate total frames to process
            if duration is not None:
                max_video_frames = int(duration * self.fps)
                total_frames_to_check = min(max_video_frames, self.total_frames)
            else:
                total_frames_to_check = self.total_frames
            
            frames_to_extract = total_frames_to_check // frame_interval
            
            print(f"Sampling every {frame_interval} frames ({frames_per_second} FPS) for {duration or 'entire'} video")
            print(f"Will extract approximately {frames_to_extract} frames")
            
        else:
            # Use max_frames logic (backward compatibility)
            frame_interval = 1
            frames_to_extract = min(max_frames or self.total_frames, self.total_frames)
            total_frames_to_check = frames_to_extract
            print(f"Extracting {frames_to_extract} frames...")
        
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=frames_to_extract, desc="Extracting frames") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Check if we've processed enough frames based on duration
                if duration is not None and frame_count >= duration * self.fps:
                    break
                
                # Check if we should extract this frame based on interval
                if frame_count % frame_interval == 0:
                    # Save frame
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    frame_indices.append(frame_count)  # Store original frame index
                    extracted_count += 1
                    pbar.update(1)
                    
                    # Check max_frames limit (backward compatibility)
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        print(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths, frame_indices
    
    def create_video_with_overlays(self, frame_paths: List[str], detections_list: List[dict], 
                                 output_path: str = "output_with_detections.mp4",
                                 frame_indices: Optional[List[int]] = None,
                                 object_label: str = None,
                                 persistence_frames: int = 0) -> str:
        """
        Create a new video with bounding box overlays, preserving original timing.
        
        Args:
            frame_paths: List of extracted frame file paths
            detections_list: List of detection results for each extracted frame
            output_path: Path for the output video
            frame_indices: List of original frame indices for the extracted frames
            object_label: User-specified object label to display on bounding boxes
            persistence_frames: Number of frames to keep boxes visible after detection stops
        
        Returns:
            Path to the created video file
        """
        if len(frame_paths) != len(detections_list):
            raise ValueError("Number of frames and detections must match")
        
        # Reset video capture to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        print(f"Creating video with overlays: {output_path}")
        
        # Create a mapping of frame indices to detections
        if frame_indices is not None:
            detection_map = {idx: detections for idx, detections in zip(frame_indices, detections_list)}
        else:
            # Fallback: assume sequential frames (backward compatibility)
            detection_map = {i: detections for i, detections in enumerate(detections_list)}
        
        # Build a comprehensive detection timeline with persistence
        detection_timeline = {}
        last_successful_detection = None
        
        # Process frames in order to handle failed detections properly
        sorted_frame_indices = sorted(detection_map.keys())
        
        for frame_idx in sorted_frame_indices:
            detections = detection_map[frame_idx]
            
            if detections.get("objects", []):
                # Successful detection - use it and update last successful
                detection_timeline[frame_idx] = detections
                last_successful_detection = detections
                
            elif last_successful_detection is not None:
                # Failed detection (likely 429 error) - use last successful detection
                detection_timeline[frame_idx] = last_successful_detection
        
        # Now apply additional persistence - extend detections forward in time
        if persistence_frames > 0:
            timeline_copy = detection_timeline.copy()
            for frame_idx, detections in timeline_copy.items():
                # Extend this detection forward for persistence_frames
                for offset in range(1, persistence_frames + 1):
                    persist_frame = frame_idx + offset
                    if persist_frame < self.total_frames:
                        # Only persist if no detection exists at this frame
                        if persist_frame not in detection_timeline:
                            detection_timeline[persist_frame] = detections
        
        print(f"Detection timeline: {len(detection_map)} analyzed frames, {len(detection_timeline)} frames with overlays (including persistence)")
        
        current_frame = 0
        with tqdm(total=self.total_frames, desc="Creating video") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Check if this frame should have overlays (either detection or persistence)
                if current_frame in detection_timeline:
                    frame_with_overlay = self._draw_bounding_boxes(frame, detection_timeline[current_frame], object_label)
                else:
                    frame_with_overlay = frame
                
                # Write frame to output video
                out.write(frame_with_overlay)
                current_frame += 1
                pbar.update(1)
        
        # Release everything
        out.release()
        
        print(f"Video created successfully: {output_path}")
        return output_path
    
    def _draw_bounding_boxes(self, frame: np.ndarray, detections: dict, object_label: str = None) -> np.ndarray:
        """
        Draw bounding boxes on a frame.
        
        Args:
            frame: OpenCV frame (BGR format)
            detections: Detection results from Moondream
            object_label: User-specified object label to display
        
        Returns:
            Frame with bounding boxes drawn
        """
        frame_copy = frame.copy()
        
        if not detections or "objects" not in detections:
            return frame_copy
        
        objects = detections["objects"]
        
        for obj in objects:
            # Convert normalized coordinates to pixel values
            x_min = int(obj["x_min"] * self.width)
            y_min = int(obj["y_min"] * self.height)
            x_max = int(obj["x_max"] * self.width)
            y_max = int(obj["y_max"] * self.height)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # Add label - use user-specified label if provided, otherwise use API response
            if object_label:
                label = object_label
            else:
                label = obj.get("label", "Object")
            
            confidence = obj.get("confidence", 0.0)
            text = f"{label}: {confidence:.2f}" if confidence > 0 else label
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw text background
            cv2.rectangle(
                frame_copy, 
                (x_min, y_min - text_height - baseline - 5),
                (x_min + text_width, y_min),
                (0, 0, 255), 
                -1
            )
            
            # Draw text
            cv2.putText(
                frame_copy, text, (x_min, y_min - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return frame_copy
    
    def cleanup_frames(self, frame_paths: List[str]):
        """
        Clean up extracted frame files.
        
        Args:
            frame_paths: List of frame file paths to delete
        """
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except OSError:
                pass
        
        # Try to remove the frames directory if it's empty
        try:
            frame_dir = os.path.dirname(frame_paths[0]) if frame_paths else "frames"
            os.rmdir(frame_dir)
        except OSError:
            pass
    
    def __del__(self):
        """Release video capture resources."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()