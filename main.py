#!/usr/bin/env python3
"""
Main script for video object detection using Moondream API.

This script processes a video by:
1. Extracting frames from the video
2. Running object detection on each frame using Moondream
3. Overlaying bounding boxes on the frames
4. Reconstructing the video with the overlays

Usage:
    python main.py --video path/to/video.mp4 --api-key your-api-key --object person
"""

import argparse
import os
import sys
from typing import Optional

from video_processor import VideoProcessor
from moondream_detector import MoondreamDetector


def main():
    """Main function to orchestrate the video processing pipeline."""
    parser = argparse.ArgumentParser(description="Video object detection with Moondream API")
    
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        required=True,
        help="Moondream API key"
    )
    
    parser.add_argument(
        "--object", "-o",
        default="person",
        help="Object type to detect (default: person)"
    )
    
    parser.add_argument(
        "--output", "-out",
        default="output_with_detections.mp4",
        help="Output video file path (default: output_with_detections.mp4)"
    )
    
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)"
    )
    
    parser.add_argument(
        "--frames-dir", "-f",
        default="frames",
        help="Directory to store extracted frames (default: frames)"
    )
    
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frames after processing"
    )
    
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )
    
    parser.add_argument(
        "--save-detections",
        action="store_true",
        help="Save detection results to a text file"
    )
    
    args = parser.parse_args()
    
    # Validate input video file
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("=" * 60)
    print("Video Object Detection with Moondream")
    print("=" * 60)
    print(f"Input video: {args.video}")
    print(f"Object type: {args.object}")
    print(f"Output video: {args.output}")
    print(f"Max frames: {args.max_frames or 'All'}")
    print(f"Batch delay: {args.batch_delay}s")
    print()
    
    try:
        # Step 1: Initialize video processor
        print("Step 1: Initializing video processor...")
        video_processor = VideoProcessor(args.video)
        
        # Step 2: Extract frames
        print("Step 2: Extracting frames...")
        frame_paths, frame_indices = video_processor.extract_frames(
            output_dir=args.frames_dir,
            max_frames=args.max_frames
        )
        
        if not frame_paths:
            print("Error: No frames extracted from video")
            sys.exit(1)
        
        # Step 3: Initialize Moondream detector
        print("Step 3: Initializing Moondream detector...")
        detector = MoondreamDetector(args.api_key)
        
        # Step 4: Run object detection
        print("Step 4: Running object detection...")
        detections_list = detector.detect_objects_in_frames(
            frame_paths=frame_paths,
            object_type=args.object,
            batch_delay=args.batch_delay
        )
        
        # Step 5: Save detection results if requested
        if args.save_detections:
            print("Step 5a: Saving detection results...")
            detector.save_detection_results(detections_list, "detection_results.txt")
        
        # Step 6: Create video with overlays
        print("Step 5: Creating video with bounding box overlays...")
        output_video_path = video_processor.create_video_with_overlays(
            frame_paths=frame_paths,
            detections_list=detections_list,
            output_path=args.output,
            frame_indices=frame_indices,
            object_label=args.object,
            persistence_frames=0,  # Default no persistence for CLI
            overlay_color="#8A2BE2",  # Default bright purple
            label_size=2.0  # Default 2x size
        )
        
        # Step 7: Clean up frames if requested
        if not args.keep_frames:
            print("Step 6: Cleaning up temporary frames...")
            video_processor.cleanup_frames(frame_paths)
        else:
            print(f"Step 6: Keeping extracted frames in: {args.frames_dir}")
        
        # Step 8: Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        
        summary = MoondreamDetector.get_detection_summary(detections_list)
        print(f"📊 Detection Summary:")
        print(f"   • Total frames processed: {summary['total_frames']}")
        print(f"   • Total objects detected: {summary['total_objects']}")
        print(f"   • Frames with objects: {summary['frames_with_objects']}")
        print(f"   • Average objects per frame: {summary['avg_objects_per_frame']:.2f}")
        print(f"   • Detection rate: {summary['detection_rate']:.1%}")
        print()
        print(f"🎥 Output video: {output_video_path}")
        print(f"📁 File size: {get_file_size_mb(output_video_path):.1f} MB")
        
        if args.save_detections:
            print(f"📄 Detection results: detection_results.txt")
        
        print("\n✅ Success! Your video with object detection overlays is ready.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        sys.exit(1)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return 0.0


if __name__ == "__main__":
    main()
