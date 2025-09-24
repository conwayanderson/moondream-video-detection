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
        frame_count = 0
        extracted_count = 0
        
        print(f"Extracting {frames_to_extract} frames...")
        
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
                
                # Print progress every 20 frames
                if extracted_count % 20 == 0 or extracted_count == frames_to_extract:
                    percent = (extracted_count / frames_to_extract) * 100
                    print(f"Extraction: {extracted_count}/{frames_to_extract} frames ({percent:.1f}%)")
                
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
                                 persistence_frames: int = 0,
                                 overlay_color: str = "#8A2BE2",
                                 label_size: float = 2.0) -> str:
        """
        Create a new video with bounding box overlays, preserving original timing.
        
        Args:
            frame_paths: List of extracted frame file paths
            detections_list: List of detection results for each extracted frame
            output_path: Path for the output video
            frame_indices: List of original frame indices for the extracted frames
            object_label: User-specified object label to display on bounding boxes
            persistence_frames: Number of frames to keep boxes visible after detection stops
            overlay_color: Color for bounding boxes and text (hex format)
            label_size: Size multiplier for text labels
        
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
        print(f"Creating video with {len(detection_timeline)} overlay frames...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Check if this frame should have overlays (either detection or persistence)
            if current_frame in detection_timeline:
                frame_with_overlay = self._draw_bounding_boxes(frame, detection_timeline[current_frame], object_label, overlay_color, label_size)
            else:
                frame_with_overlay = frame
            
            # Write frame to output video
            out.write(frame_with_overlay)
            current_frame += 1
            
            # # Print progress every 50 frames
            # if current_frame % 50 == 0 or current_frame == self.total_frames:
            #     percent = (current_frame / self.total_frames) * 100
            #     print(f"Video creation: {current_frame}/{self.total_frames} frames ({percent:.1f}%)")
        
        # Release everything
        out.release()
        
        print(f"Video created successfully: {output_path}")
        return output_path
    
    def _draw_bounding_boxes(self, frame: np.ndarray, detections: dict, object_label: str = None,
                           overlay_color: str = "#8A2BE2", label_size: float = 2.0) -> np.ndarray:
        """
        Draw high-resolution bounding boxes on a frame.
        
        Args:
            frame: OpenCV frame (BGR format)
            detections: Detection results from Moondream
            object_label: User-specified object label to display
            overlay_color: Color for bounding boxes and text (hex format)
            label_size: Size multiplier for text labels
        
        Returns:
            Frame with high-resolution bounding boxes drawn
        """
        # Render at 2x resolution for better quality
        scale_factor = 2.0
        original_height, original_width = frame.shape[:2]
        upscaled_width = int(original_width * scale_factor)
        upscaled_height = int(original_height * scale_factor)
        
        # Upscale frame using high-quality interpolation
        frame_upscaled = cv2.resize(frame, (upscaled_width, upscaled_height), interpolation=cv2.INTER_CUBIC)
        frame_copy = frame_upscaled.copy()
        
        if not detections or "objects" not in detections:
            # Scale back down to original size
            return cv2.resize(frame_copy, (original_width, original_height), interpolation=cv2.INTER_AREA)
        
        # Convert color to BGR with proper format handling
        try:
            if overlay_color.startswith('rgba('):
                # Parse RGBA format: rgba(r, g, b, a)
                rgba_str = overlay_color[5:-1]  # Remove 'rgba(' and ')'
                rgba_values = [float(x.strip()) for x in rgba_str.split(',')]
                r, g, b = int(rgba_values[0]), int(rgba_values[1]), int(rgba_values[2])
                bgr_color = (b, g, r)  # Convert RGB to BGR
            elif overlay_color.startswith('#'):
                # Parse hex format: #RRGGBB
                hex_color = overlay_color.lstrip('#')
                if len(hex_color) != 6:
                    raise ValueError(f"Invalid hex color length: {len(hex_color)}")
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert RGB to BGR
            else:
                raise ValueError(f"Unsupported color format: {overlay_color}")
        except (ValueError, IndexError) as e:
            print(f"Error parsing color '{overlay_color}': {e}. Using default purple.")
            bgr_color = (138, 43, 226)  # Default bright purple in BGR
        
        # Scale parameters for upscaled rendering
        box_thickness = max(int(4 * scale_factor), int(3 * label_size * scale_factor))
        font_scale = 0.6 * label_size * scale_factor  # Scale font for upscaled frame
        text_thickness = max(int(2 * scale_factor), int(2 * label_size * scale_factor))
        
        objects = detections["objects"]
        
        for obj in objects:
            # Scale coordinates to upscaled frame
            x_min = int(round(obj["x_min"] * self.width * scale_factor))
            y_min = int(round(obj["y_min"] * self.height * scale_factor))
            x_max = int(round(obj["x_max"] * self.width * scale_factor))
            y_max = int(round(obj["y_max"] * self.height * scale_factor))
            
            # Draw 10% opacity fill inside bounding box
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), bgr_color, -1)
            cv2.addWeighted(overlay, 0.1, frame_copy, 0.9, 0, frame_copy)
            
            # Draw bounding box with anti-aliasing for smoother lines
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), bgr_color, box_thickness, cv2.LINE_AA)
            
            # Add label - use user-specified label if provided, otherwise use API response
            if object_label:
                label = object_label
            else:
                label = obj.get("label", "Object")
            
            confidence = obj.get("confidence", 0.0)
            text = f"{label}: {confidence:.2f}" if confidence > 0 else label
            
            # Use standard font for consistency
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font_face, font_scale, text_thickness
            )
            
            # Scale padding for upscaled frame
            padding = int(8 * scale_factor)
            text_bg_width = text_width + (2 * padding)
            text_bg_height = text_height + baseline + (2 * padding)
            
            # Draw text background with smooth edges
            cv2.rectangle(
                frame_copy, 
                (x_min, y_min - text_bg_height),
                (x_min + text_bg_width, y_min),
                bgr_color, 
                -1,
                cv2.LINE_AA
            )
            
            # Draw text with anti-aliasing for better quality
            cv2.putText(
                frame_copy, text, 
                (x_min + padding, y_min - baseline - padding),
                font_face, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA
            )
        
        # Scale back down to original size for final output
        return cv2.resize(frame_copy, (original_width, original_height), interpolation=cv2.INTER_AREA)
    
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