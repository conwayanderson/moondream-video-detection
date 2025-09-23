#!/usr/bin/env python3
"""
Example usage script demonstrating how to use the video detection modules.

This script shows how to use the VideoProcessor and MoondreamDetector classes
directly in your own code, rather than using the command-line interface.
"""

from video_processor import VideoProcessor
from moondream_detector import MoondreamDetector
import os


def example_basic_usage():
    """Basic example of processing a video with object detection."""
    
    # Configuration
    VIDEO_PATH = "path/to/your/video.mp4"  # Replace with your video path
    API_KEY = "your-moondream-api-key"     # Replace with your API key
    OBJECT_TYPE = "person"                 # Object to detect
    OUTPUT_VIDEO = "output_with_detections.mp4"
    
    print("=== Basic Video Object Detection Example ===")
    
    try:
        # Step 1: Initialize video processor
        print("1. Initializing video processor...")
        processor = VideoProcessor(VIDEO_PATH)
        
        # Step 2: Extract frames (limit to first 30 for this example)
        print("2. Extracting frames...")
        frame_paths = processor.extract_frames(max_frames=30)
        
        # Step 3: Initialize Moondream detector
        print("3. Initializing Moondream detector...")
        detector = MoondreamDetector(API_KEY)
        
        # Step 4: Run detection
        print("4. Running object detection...")
        detections = detector.detect_objects_in_frames(
            frame_paths, 
            object_type=OBJECT_TYPE,
            batch_delay=0.1
        )
        
        # Step 5: Create output video
        print("5. Creating output video...")
        output_path = processor.create_video_with_overlays(
            frame_paths, 
            detections, 
            OUTPUT_VIDEO
        )
        
        # Step 6: Print results
        summary = MoondreamDetector.get_detection_summary(detections)
        print(f"\n✅ Processing complete!")
        print(f"Output video: {output_path}")
        print(f"Objects detected: {summary['total_objects']}")
        print(f"Detection rate: {summary['detection_rate']:.1%}")
        
        # Step 7: Cleanup
        processor.cleanup_frames(frame_paths)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_custom_processing():
    """Example with custom processing and analysis."""
    
    VIDEO_PATH = "path/to/your/video.mp4"
    API_KEY = "your-moondream-api-key"
    
    print("=== Custom Processing Example ===")
    
    try:
        # Initialize components
        processor = VideoProcessor(VIDEO_PATH)
        detector = MoondreamDetector(API_KEY)
        
        # Extract every 10th frame for faster processing
        print("Extracting every 10th frame...")
        all_frame_paths = processor.extract_frames()
        selected_frames = all_frame_paths[::10]  # Every 10th frame
        
        print(f"Processing {len(selected_frames)} out of {len(all_frame_paths)} frames")
        
        # Detect multiple object types
        object_types = ["person", "car", "bicycle"]
        all_detections = {}
        
        for obj_type in object_types:
            print(f"Detecting {obj_type}...")
            detections = detector.detect_objects_in_frames(
                selected_frames, 
                object_type=obj_type,
                batch_delay=0.2
            )
            all_detections[obj_type] = detections
            
            # Save individual results
            detector.save_detection_results(
                detections, 
                f"detections_{obj_type}.txt"
            )
        
        # Analyze results
        for obj_type, detections in all_detections.items():
            summary = MoondreamDetector.get_detection_summary(detections)
            print(f"\n{obj_type.title()} Detection Summary:")
            print(f"  Total detected: {summary['total_objects']}")
            print(f"  Frames with {obj_type}: {summary['frames_with_objects']}")
            print(f"  Detection rate: {summary['detection_rate']:.1%}")
        
        # Create video with person detections (as primary focus)
        if "person" in all_detections:
            print("\nCreating video with person detections...")
            
            # Need to expand detections to match all frames
            expanded_detections = []
            frame_index = 0
            
            for i in range(len(all_frame_paths)):
                if i % 10 == 0 and frame_index < len(all_detections["person"]):
                    # Use detection result
                    expanded_detections.append(all_detections["person"][frame_index])
                    frame_index += 1
                else:
                    # No detection for this frame
                    expanded_detections.append({"objects": []})
            
            output_path = processor.create_video_with_overlays(
                all_frame_paths,
                expanded_detections,
                "output_custom_processing.mp4"
            )
            print(f"Custom processed video: {output_path}")
        
        # Cleanup
        processor.cleanup_frames(all_frame_paths)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_single_frame_detection():
    """Example of detecting objects in a single frame."""
    
    FRAME_PATH = "path/to/frame.jpg"  # Replace with your image path
    API_KEY = "your-moondream-api-key"
    
    print("=== Single Frame Detection Example ===")
    
    try:
        # Initialize detector
        detector = MoondreamDetector(API_KEY)
        
        # Detect objects in single frame
        result = detector.detect_single_frame(FRAME_PATH, "person")
        
        print(f"Objects detected: {len(result.get('objects', []))}")
        print(f"Request ID: {result.get('request_id')}")
        
        # Print detailed results
        for i, obj in enumerate(result.get("objects", [])):
            print(f"\nObject {i + 1}:")
            print(f"  Label: {obj.get('label', 'N/A')}")
            print(f"  Confidence: {obj.get('confidence', 'N/A')}")
            print(f"  Bounding box: ({obj.get('x_min', 0):.3f}, {obj.get('y_min', 0):.3f}, "
                  f"{obj.get('x_max', 0):.3f}, {obj.get('y_max', 0):.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("Moondream Video Detection - Example Usage")
    print("=" * 50)
    print()
    
    print("This script demonstrates how to use the video detection modules.")
    print("Before running, make sure to:")
    print("1. Replace 'path/to/your/video.mp4' with your actual video path")
    print("2. Replace 'your-moondream-api-key' with your actual API key")
    print("3. Install all requirements: pip install -r requirements.txt")
    print()
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_custom_processing()
    # example_single_frame_detection()
    
    print("To run an example, uncomment the function call at the bottom of this file.")


