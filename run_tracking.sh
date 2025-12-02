#!/bin/bash
# BoT-SORT Video Tracking Script

cd /Users/shivv/Desktop/BoT-SORT
source venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/Users/shivv/Desktop/BoT-SORT"

# Check if video path is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_tracking.sh <path_to_video>"
    echo "Example: ./run_tracking.sh /Users/shivv/Desktop/my_video.mp4"
    exit 1
fi

VIDEO_PATH="$1"

echo "Running BoT-SORT tracking on: $VIDEO_PATH"
echo "Results will be saved to: runs/track/"
echo ""

python3 tools/mc_demo_yolov7.py \
  --weights pretrained/yolov7.pt \
  --source "$VIDEO_PATH" \
  --fuse-score \
  --agnostic-nms \
  --save-txt \
  --save-vid

echo ""
echo "Tracking complete! Check runs/track/ for results."
