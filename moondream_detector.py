"""
Moondream API integration for object detection on video frames.
"""

import moondream as md
from PIL import Image
import os
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import asyncio
import concurrent.futures
from threading import Lock


class MoondreamDetector:
    """Handles object detection using Moondream API."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Moondream detector.
        
        Args:
            api_key: Your Moondream API key
        """
        self.model = md.vl(api_key=api_key)
        self.progress_lock = Lock()
        print("Moondream detector initialized successfully")
    
    def detect_objects_in_frames(self, frame_paths: List[str], 
                               object_type: str = "person",
                               batch_delay: float = 0.1,
                               max_concurrent: int = 5) -> List[Dict]:
        """
        Detect objects in a list of frame images using concurrent processing.
        
        Args:
            frame_paths: List of paths to frame images
            object_type: Type of object to detect (e.g., "person", "car", "face")
            batch_delay: Delay between API calls to avoid rate limiting
            max_concurrent: Maximum number of concurrent API calls
        
        Returns:
            List of detection results for each frame (maintains order)
        """
        # Initialize results list with None values to maintain order
        detections_list = [None] * len(frame_paths)
        completed_count = 0
        
        print(f"Detecting '{object_type}' in {len(frame_paths)} frames using {max_concurrent} concurrent workers...")
        
        def process_single_frame(frame_index_and_path):
            """Process a single frame and return (index, result)"""
            frame_index, frame_path = frame_index_and_path
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Load image
                    image = Image.open(frame_path)
                    
                    # Add delay before API call to avoid rate limiting
                    if batch_delay > 0:
                        time.sleep(batch_delay)
                    
                    # Detect objects
                    result = self.model.detect(image, object_type)
                    
                    return frame_index, result
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        if attempt < max_retries - 1:
                            # Exponential backoff for rate limit errors
                            sleep_time = retry_delay * (2 ** attempt)
                            print(f"Rate limit hit for frame {frame_index}, retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                        else:
                            print(f"Max retries exceeded for frame {frame_path}: {error_msg}")
                    else:
                        print(f"Error processing frame {frame_path}: {error_msg}")
                    
                    return frame_index, {"objects": [], "request_id": None}
        
        # Create progress bar
        with tqdm(total=len(frame_paths), desc=f"Detecting {object_type}") as pbar:
            # Use ThreadPoolExecutor for concurrent API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # Submit all tasks
                frame_tasks = [(i, path) for i, path in enumerate(frame_paths)]
                future_to_index = {
                    executor.submit(process_single_frame, task): task[0] 
                    for task in frame_tasks
                }
                
                # Process completed tasks as they finish
                for future in concurrent.futures.as_completed(future_to_index):
                    frame_index, result = future.result()
                    detections_list[frame_index] = result
                    
                    # Update progress bar
                    with self.progress_lock:
                        completed_count += 1
                        num_objects = len(result.get("objects", []))
                        pbar.set_postfix({
                            "completed": completed_count,
                            "objects": num_objects
                        })
                        pbar.update(1)
        
        # Print summary
        total_objects = sum(len(d.get("objects", [])) for d in detections_list)
        frames_with_objects = sum(1 for d in detections_list if len(d.get("objects", [])) > 0)
        
        print(f"Detection complete!")
        print(f"Total objects detected: {total_objects}")
        print(f"Frames with objects: {frames_with_objects}/{len(frame_paths)}")
        
        return detections_list
    
    def detect_single_frame(self, frame_path: str, object_type: str = "person") -> Dict:
        """
        Detect objects in a single frame.
        
        Args:
            frame_path: Path to the frame image
            object_type: Type of object to detect
        
        Returns:
            Detection result for the frame
        """
        try:
            image = Image.open(frame_path)
            result = self.model.detect(image, object_type)
            return result
        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            return {"objects": [], "request_id": None}
    
    def save_detection_results(self, detections_list: List[Dict], output_path: str = "detections.txt"):
        """
        Save detection results to a file for debugging/analysis.
        
        Args:
            detections_list: List of detection results
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            f.write("Moondream Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, detections in enumerate(detections_list):
                f.write(f"Frame {i + 1}:\n")
                f.write(f"Request ID: {detections.get('request_id', 'N/A')}\n")
                
                objects = detections.get("objects", [])
                f.write(f"Objects found: {len(objects)}\n")
                
                for j, obj in enumerate(objects):
                    f.write(f"  Object {j + 1}:\n")
                    f.write(f"    Label: {obj.get('label', 'N/A')}\n")
                    f.write(f"    Confidence: {obj.get('confidence', 'N/A')}\n")
                    f.write(f"    Bounding box: ({obj.get('x_min', 0):.3f}, {obj.get('y_min', 0):.3f}, "
                           f"{obj.get('x_max', 0):.3f}, {obj.get('y_max', 0):.3f})\n")
                
                f.write("\n")
        
        print(f"Detection results saved to: {output_path}")
    
    @staticmethod
    def get_detection_summary(detections_list: List[Dict]) -> Dict:
        """
        Get a summary of detection results.
        
        Args:
            detections_list: List of detection results
        
        Returns:
            Summary statistics
        """
        total_frames = len(detections_list)
        total_objects = sum(len(d.get("objects", [])) for d in detections_list)
        frames_with_objects = sum(1 for d in detections_list if len(d.get("objects", [])) > 0)
        
        # Calculate average objects per frame
        avg_objects_per_frame = total_objects / total_frames if total_frames > 0 else 0
        
        # Calculate detection rate
        detection_rate = frames_with_objects / total_frames if total_frames > 0 else 0
        
        return {
            "total_frames": total_frames,
            "total_objects": total_objects,
            "frames_with_objects": frames_with_objects,
            "avg_objects_per_frame": avg_objects_per_frame,
            "detection_rate": detection_rate
        }