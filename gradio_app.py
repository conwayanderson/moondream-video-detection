#!/usr/bin/env python3
"""
Gradio web interface for video object detection using Moondream API.

This provides a user-friendly web interface where users can:
1. Upload videos
2. Enter their Moondream API key
3. Select object types to detect
4. View processing progress
5. Download the result video with overlays
"""

import gradio as gr
import os
import tempfile
import shutil
from typing import Optional, Tuple, List
import time

from video_processor import VideoProcessor
from moondream_detector import MoondreamDetector


class GradioVideoProcessor:
    """Wrapper class for video processing with Gradio integration."""
    
    def __init__(self):
        self.temp_dir = None
        self.processor = None
        self.detector = None
    
    def get_video_duration(self, video_file):
        """Get video info and return empty duration (let user choose)."""
        if not video_file:
            return None
        
        try:
            import cv2
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            
            # Return None (empty) but print the video info for user reference
            print(f"Video uploaded: {duration:.3f} seconds total ({int(frame_count)} frames at {fps:.2f} FPS)")
            return None
        except:
            return None
    
    def process_video(self, 
                     video_file, 
                     api_key: str, 
                     object_type: str = "person",
                     frames_per_second: float = 1.0,
                     duration: Optional[int] = None,
                     batch_delay: float = 0.1,
                    persistence_frames: int = 10,
                    concurrent_workers: int = 5,
                    progress=gr.Progress()) -> str:
        """
        Process video with object detection and return results.
        
        Args:
            video_file: Uploaded video file from Gradio
            api_key: Moondream API key
            object_type: Object type to detect
            frames_per_second: How many frames per second to analyze
            duration: How many seconds of video to process
            batch_delay: Delay between API calls
            persistence_frames: Number of frames to keep boxes visible after detection stops
            concurrent_workers: Number of concurrent API workers
            progress: Gradio progress tracker
        
        Returns:
            Path to output video file
        """
        if not video_file:
            return None
        
        # Debug: Print what we received
        print(f"DEBUG: Received API key: '{api_key}' (length: {len(api_key) if api_key else 0})")
        
        if not api_key or not api_key.strip():
            return None
        
        # Handle duration - convert 0 or negative to None
        if duration is not None and duration <= 0:
            duration = None
        
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            
            # Copy uploaded file to temp directory
            video_path = os.path.join(self.temp_dir, "input_video.mp4")
            shutil.copy2(video_file, video_path)
            
            progress(0.1, desc="Initializing video processor...")
            
            # Initialize video processor
            self.processor = VideoProcessor(video_path)
            
            progress(0.15, desc="Extracting frames...")
            
            # Extract frames
            frames_dir = os.path.join(self.temp_dir, "frames")
            frame_paths, frame_indices = self.processor.extract_frames(
                output_dir=frames_dir,
                frames_per_second=frames_per_second,
                duration=duration
            )
            
            if not frame_paths:
                return None, "❌ No frames could be extracted from the video.", ""
            
            progress(0.3, desc="Initializing Moondream detector...")
            
            # Initialize detector
            self.detector = MoondreamDetector(api_key.strip())
            
            progress(0.4, desc=f"Detecting {object_type} in {len(frame_paths)} frames...")
            
            # Run detection with concurrent processing
            detections_list = self.detector.detect_objects_in_frames(
                frame_paths=frame_paths,
                object_type=object_type,
                batch_delay=batch_delay,
                max_concurrent=int(concurrent_workers)
            )
            
            progress(0.8, desc="Creating output video with overlays...")
            
            # Create output video
            output_path = os.path.join(self.temp_dir, "output_with_detections.mp4")
            final_output = self.processor.create_video_with_overlays(
                frame_paths=frame_paths,
                detections_list=detections_list,
                output_path=output_path,
                frame_indices=frame_indices,
                object_label=object_type,
                persistence_frames=persistence_frames
            )
            
            progress(0.95, desc="Generating summary...")
            
            # Generate summary
            summary = MoondreamDetector.get_detection_summary(detections_list)
            summary_text = self._format_summary(summary, object_type)
            
            # Generate detection log
            detection_log = self._format_detection_log(detections_list, frame_paths)
            
            progress(1.0, desc="Processing complete!")
            
            return final_output
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            return None
        
        finally:
            # Cleanup will be handled by Gradio's temporary file management
            pass
    
    
    def _format_summary(self, summary: dict, object_type: str) -> str:
        """Format detection summary for display."""
        return f"""
## Detection Summary

**Object Type:** {object_type.title()}

**Results:**
- Total frames processed: {summary['total_frames']}
- Total objects detected: {summary['total_objects']}
- Frames with objects: {summary['frames_with_objects']}
- Average objects per frame: {summary['avg_objects_per_frame']:.2f}
- Detection rate: {summary['detection_rate']:.1%}

**Status:** Processing completed successfully!
"""
    
    def _format_detection_log(self, detections_list: List[dict], frame_paths: List[str]) -> str:
        """Format detailed detection log."""
        log_lines = ["# Detailed Detection Log\n"]
        
        for i, (detections, frame_path) in enumerate(zip(detections_list, frame_paths)):
            frame_name = os.path.basename(frame_path)
            objects = detections.get("objects", [])
            
            log_lines.append(f"## Frame {i+1}: {frame_name}")
            log_lines.append(f"Objects found: {len(objects)}")
            
            if objects:
                for j, obj in enumerate(objects):
                    confidence = obj.get("confidence", 0.0)
                    log_lines.append(f"  - Object {j+1}: {obj.get('label', 'Unknown')}")
                    if confidence > 0:
                        log_lines.append(f"    Confidence: {confidence:.2f}")
                    log_lines.append(f"    Bounding box: ({obj.get('x_min', 0):.3f}, {obj.get('y_min', 0):.3f}, {obj.get('x_max', 0):.3f}, {obj.get('y_max', 0):.3f})")
            else:
                log_lines.append("  No objects detected")
            
            log_lines.append("")  # Empty line
        
        return "\n".join(log_lines)


def create_interface():
    """Create and configure the Gradio interface."""
    
    processor = GradioVideoProcessor()
    
    # Define the interface
    with gr.Blocks(title="Video Object Detection with Moondream", 
                   theme=gr.themes.Glass()) as interface:
        
        gr.Markdown("""
        # Video Object Detection with Moondream
        
        Upload a video, enter your Moondream API key, and get object detection overlays!
        
        **Features:**
        - Detect various objects (people, cars, animals, etc.)
        - Automatic bounding box overlays
        - Detailed detection statistics
        - Download processed video
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## Upload & Configure")
                
                video_input = gr.File(
                    label="Upload Video File",
                    file_types=["video"],
                    type="filepath"
                )
                
                api_key_input = gr.Textbox(
                    label="Moondream API Key",
                    info="Enter your Moondream API key here",
                    type="password",
                    elem_id="api_key_input"
                )
                
                object_type_input = gr.Textbox(
                    label="Object Type to Detect",
                    info="Enter any object type (e.g., person, car, dog, bicycle, etc.)",
                    value="person",
                    placeholder="person",
                    elem_id="object_type_input"
                )
                
                with gr.Row():
                    frames_per_second_input = gr.Number(
                        label="Frames per Second to Analyze",
                        info="How many frames per second to process (e.g., 1 = every 1 second, 0.5 = every 2 seconds)",
                        value=15.0,
                        minimum=0.1,
                        maximum=30.0,
                        step=0.1,
                        elem_id="frames_per_second_input"
                    )
                    
                    duration_input = gr.Number(
                        label="Duration to Process (seconds)",
                        info="How many seconds of video to process (leave empty for entire video)",
                        step=1,
                        elem_id="duration_input"
                    )
                
                with gr.Row():
                    batch_delay_input = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=0.5,
                        step=0.1,
                        label="Batch Delay (seconds)",
                        info="Delay between API calls to avoid rate limiting",
                        elem_id="batch_delay_input"
                    )
                    
                    persistence_frames_input = gr.Number(
                        label="Box Persistence (frames)",
                        info="Keep bounding boxes visible for this many frames after detection stops",
                        value=15,
                        minimum=0,
                        step=1,
                        elem_id="persistence_frames_input"
                    )
                
                concurrent_workers_input = gr.Number(
                    label="Concurrent API Workers",
                    info="Number of simultaneous API calls",
                    value=1,
                    minimum=1,
                    step=1,
                    elem_id="concurrent_workers_input"
                )
                
                process_btn = gr.Button(
                    "Process Video", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("## Results")
                
                output_video = gr.File(
                    label="Processed Video with Overlays",
                    interactive=False
                )
                
                summary_output = gr.Markdown(
                    value="Upload a video and click 'Process Video' to see results here."
                )
        
        # Detailed log section (collapsible)
        with gr.Accordion("Detailed Detection Log", open=False):
            detection_log = gr.Markdown(
                value="Detailed detection results will appear here after processing."
            )
        
        # Examples section
        with gr.Accordion("Usage Tips", open=False):
            gr.Markdown("""
            ### Getting Started
            1. **Get API Key**: Sign up at [Moondream.ai](https://moondream.ai) to get your API key
            2. **Upload Video**: Choose any video file (MP4, AVI, MOV, etc.)
            3. **Select Object**: Choose what type of objects to detect
            4. **Process**: Click the process button and wait for results
            
            ### Object Detection Tips
            - **Popular objects**: person, car, dog, cat, bicycle
            - **Animals**: horse, sheep, cow, elephant, bear, zebra, giraffe
            - **Vehicles**: truck, bus, motorcycle, airplane, train, boat
            - **Traffic**: traffic light, stop sign
            
            ### Performance Tips
            - Use **Duration** to limit processing for long videos
            - Increase **Batch Delay** if you hit API rate limits
            - Shorter videos (< 30 seconds) process faster
            
            ### Understanding Results
            - **Detection Rate**: Percentage of frames with detected objects
            - **Bounding Boxes**: Purple rectangles around detected objects
            - **Confidence Scores**: How certain the AI is about detections
            """)
        
        # Set up video upload handler to auto-set duration
        video_input.change(
            fn=processor.get_video_duration,
            inputs=[video_input],
            outputs=[duration_input]
        )
        
        # Set up the processing function
        process_btn.click(
            fn=processor.process_video,
            inputs=[
                video_input,
                api_key_input,
                object_type_input,
                frames_per_second_input,
                duration_input,
                batch_delay_input,
                persistence_frames_input,
                concurrent_workers_input
            ],
            outputs=[
                output_video
            ],
            show_progress=True
        )
        
        # Add JavaScript for local storage
        interface.load(
            fn=None,
            js="""
            function() {
                setTimeout(function() {
                    // Function to load and save values
                    function setupLocalStorage(id, key) {
                        const element = document.getElementById(id);
                        if (element) {
                            const input = element.querySelector('input');
                            if (input) {
                                // Load saved value
                                const savedValue = localStorage.getItem(key);
                                if (savedValue && savedValue !== 'null' && savedValue !== 'undefined') {
                                    input.value = savedValue;
                                    // Trigger change event to update Gradio's internal state
                                    input.dispatchEvent(new Event('input', { bubbles: true }));
                                    input.dispatchEvent(new Event('change', { bubbles: true }));
                                }
                                
                                // Set up auto-save
                                input.addEventListener('input', function() {
                                    localStorage.setItem(key, this.value);
                                });
                                input.addEventListener('change', function() {
                                    localStorage.setItem(key, this.value);
                                });
                            }
                        }
                    }
                    
                    // Set up local storage for all fields
                    setupLocalStorage('api_key_input', 'moondream_api_key');
                    setupLocalStorage('object_type_input', 'moondream_object_type');
                    setupLocalStorage('frames_per_second_input', 'moondream_frames_per_second');
                    setupLocalStorage('duration_input', 'moondream_duration');
                    setupLocalStorage('batch_delay_input', 'moondream_batch_delay');
                    setupLocalStorage('persistence_frames_input', 'moondream_persistence_frames');
                    setupLocalStorage('concurrent_workers_input', 'moondream_concurrent_workers');
                }, 2000);
                
                return [];
            }
            """
        )
    
    return interface


def main():
    """Launch the Gradio interface."""
    interface = create_interface()
    
    print("Starting Video Object Detection Web Interface...")
    print("Make sure you have your Moondream API key ready!")
    print("The interface will open in your browser automatically.")
    
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        show_error=True,        # Show detailed errors
        quiet=False             # Show startup logs
    )


if __name__ == "__main__":
    main()
