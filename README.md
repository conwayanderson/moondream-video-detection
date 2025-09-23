# Video Object Detection with Moondream

This Python project processes videos by extracting frames, running object detection using the Moondream API, and overlaying bounding boxes on the video playback.

## Features

- 🎥 Extract frames from any video format supported by OpenCV
- 🔍 Object detection using Moondream's vision API
- 📦 Bounding box overlay on detected objects
- 🎬 Video reconstruction with overlays
- 📊 Detection statistics and summaries
- ⚡ Batch processing with rate limiting
- 🧹 Automatic cleanup of temporary files

## Installation

1. Clone or download this project
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get your Moondream API key from [Moondream's website](https://moondream.ai/)

## Usage

### Web Interface (Recommended)

Launch the Gradio web interface for an easy-to-use GUI:

```bash
python gradio_app.py
```

Then open your browser to `http://localhost:7860` and:
1. Upload your video file
2. Enter your Moondream API key
3. Select the object type to detect
4. Click "Process Video"
5. Download the result!

### Command Line Interface

```bash
python main.py --video path/to/your/video.mp4 --api-key your-moondream-api-key
```

### Advanced Usage

```bash
python main.py \
    --video input_video.mp4 \
    --api-key your-api-key \
    --object "car" \
    --output "cars_detected.mp4" \
    --max-frames 100 \
    --batch-delay 0.2 \
    --save-detections \
    --keep-frames
```

### Command Line Arguments

- `--video, -v`: Path to input video file (required)
- `--api-key, -k`: Your Moondream API key (required)
- `--object, -o`: Object type to detect (default: "person")
  - Examples: "person", "car", "face", "dog", "bicycle", etc.
- `--output, -out`: Output video file path (default: "output_with_detections.mp4")
- `--max-frames, -m`: Maximum number of frames to process (default: all frames)
- `--frames-dir, -f`: Directory to store extracted frames (default: "frames")
- `--keep-frames`: Keep extracted frames after processing
- `--batch-delay`: Delay between API calls in seconds (default: 0.1)
- `--save-detections`: Save detection results to a text file

## Examples

### Detect People in a Video

```bash
python main.py --video family_video.mp4 --api-key your-key --object "person"
```

### Detect Cars with Custom Output

```bash
python main.py \
    --video traffic.mp4 \
    --api-key your-key \
    --object "car" \
    --output "traffic_with_cars.mp4"
```

### Process First 50 Frames Only

```bash
python main.py \
    --video long_video.mp4 \
    --api-key your-key \
    --max-frames 50 \
    --save-detections
```

## Project Structure

```
moondream-video-detection/
├── gradio_app.py           # Web interface (recommended)
├── main.py                 # Command-line interface
├── video_processor.py      # Video frame extraction and reconstruction
├── moondream_detector.py   # Moondream API integration
├── example_usage.py        # Code examples for programmatic usage
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Output

The script will create:

1. **Output video**: Video with bounding boxes overlaid on detected objects
2. **Detection results** (optional): Text file with detailed detection information
3. **Frame files** (optional): Individual frame images if `--keep-frames` is used

## Supported Object Types

The Moondream API supports detection of various objects including:
- person
- car
- bicycle
- dog
- cat
- bird
- horse
- sheep
- cow
- elephant
- bear
- zebra
- giraffe
- And many more...

## Rate Limiting

The script includes built-in rate limiting to avoid overwhelming the Moondream API:
- Default delay of 0.1 seconds between API calls
- Adjustable via `--batch-delay` parameter
- Progress tracking with real-time statistics

## Error Handling

- Graceful handling of API errors
- Automatic retry logic for failed frames
- Comprehensive error messages
- Safe cleanup of temporary files

## Performance Tips

1. **For long videos**: Use `--max-frames` to limit processing
2. **For better API performance**: Adjust `--batch-delay` based on your API limits
3. **For debugging**: Use `--save-detections` and `--keep-frames`
4. **For storage**: Remove `--keep-frames` to auto-cleanup temporary files

## Requirements

- Python 3.7+
- OpenCV (cv2)
- Pillow (PIL)
- Moondream API library
- NumPy
- Matplotlib
- tqdm

## License

This project is provided as-is for educational and development purposes.

## Troubleshooting

### Common Issues

1. **"Cannot open video file"**: Ensure the video file exists and is in a supported format
2. **API key errors**: Verify your Moondream API key is correct and active
3. **Out of memory**: Use `--max-frames` to process videos in smaller chunks
4. **Slow processing**: Increase `--batch-delay` if hitting API rate limits

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify your video file is readable
3. Ensure your API key is valid
4. Check the console output for specific error messages
