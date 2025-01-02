#!/bin/bash

# Source the configuration file
source /home/gomer/oline/PoseTools/src/modules/data/config.cfg

# Override config values with command-line arguments if provided
VIDEO_NAME=${1:-$VIDEO_NAME}
NUM_GPUS=${2:-$NUM_GPUS}
PRETRAINED_MODEL_SIZE=${3:-$PRETRAINED_MODEL_SIZE}

PRETRAINED_MODEL="smpler_x_${PRETRAINED_MODEL_SIZE}32"
echo "${PRETRAINED_MODEL}"

# Automatically generate EXP_NAME and OUTPUT_FOLDER based on VIDEO_NAME
EXP_NAME="output/demo_inference_${VIDEO_NAME}" # Experiment name
OUTPUT_FOLDER="/home/gomer/oline/smplx/SMPLer-X/demo/results/${VIDEO_NAME}" # Output folder for results

# Define paths for video and image directory
VIDEO_PATH="/home/gomer/oline/data/video_files/${VIDEO_NAME}.mp4" # Input video path
IMAGE_DIR="/home/gomer/oline/smplx/SMPLer-X/demo/images/${VIDEO_NAME}"                   # Directory to save images

# Ensure the image directory exists
mkdir -p "$IMAGE_DIR" # Create directory if it doesn't exist

# Run ffmpeg to extract frames from the video
echo "Extracting frames from ${VIDEO_PATH}..."
ffmpeg -i "$VIDEO_PATH" -r 30 -f image2 "${IMAGE_DIR}/%06d.jpg"
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract frames from ${VIDEO_PATH}."
    exit 1
fi
echo "Frames extracted successfully to ${IMAGE_DIR}/"
# Calculate START and END based on the number of image files generated
START=1
END=$(ls -1q "${IMAGE_DIR}"/*.jpg | wc -l) # Count the number of .jpg files
if [ "$END" -eq 0 ]; then
    echo "Error: No images were extracted. Check your ffmpeg command."
    exit 1
fi
echo "Frame range: START=$START, END=$END"

# Navigate to the inference directory
cd /home/gomer/oline/smplx/SMPLer-X/main || { echo "Failed to navigate to inference directory"; exit 1; }
IMG_PATH="../demo/images/${VIDEO_NAME}"

# Run the inference command
python inference.py \
    --num_gpus "$NUM_GPUS" \
    --exp_name "$EXP_NAME" \
    --pretrained_model "$PRETRAINED_MODEL" \
    --agora_benchmark agora_model \
    --img_path "$IMG_PATH" \
    --start "$START" \
    --end "$END" \
    --output_folder "$OUTPUT_FOLDER" \
    --show_verts \
    --show_bbox \
    --save_mesh


# Move and rename the resulting file
BODY_POSES_FILE="/home/gomer/oline/smplx/SMPLer-X/demo/results/arrs/body_poses_list.npy"
DEST_DIR="/home/gomer/oline/data/smplx"


APPENDIX="b"
#"${PRETRAINED_MODEL_SIZE}"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Rename and move the file
if [ -f "$BODY_POSES_FILE" ]; then
    mv "$BODY_POSES_FILE" "$DEST_DIR/${VIDEO_NAME}_${APPENDIX}.npy"
    echo "File moved to $DEST_DIR with new name: ${VIDEO_NAME}_${APPENDIX}.npy"
else
    echo "Error: Expected output file $BODY_POSES_FILE not found."
    exit 1
fi

#################################
# HAMER Extraction 

# Step 1: Start the server in the background and redirect its output to a temporary log file
#echo "Starting the server..."
#LOG_FILE="/tmp/hamer_server.log"
#cd /home/gomer/hamer || { echo "Failed to navigate to Hamer directory"; exit 1; }
#source .hamer/bin/activate
#python run_server.py > "$LOG_FILE" 2>&1 &
#SERVER_PID=$!

# Step 2: Wait for the server to be ready
#echo "Waiting for the server to be ready..."
#READY_INDICATOR="Starting Uvicorn server"
#TIMEOUT=300  # Set a timeout in seconds
#INTERVAL=5   # Interval between checks
#
#for ((i=0; i<TIMEOUT; i+=INTERVAL)); do
#    if grep -q "$READY_INDICATOR" "$LOG_FILE"; then
#        echo "Server is ready."
#        break
#    fi
#    echo "Waiting for server... ($i seconds elapsed)"
#    sleep $INTERVAL
#done
#
## If the server is still not ready after the timeout, exit with an error
#if ! grep -q "$READY_INDICATOR" "$LOG_FILE"; then
#    echo "Error: Server did not start within $TIMEOUT seconds."
#    kill "$SERVER_PID"
#    exit 1
#fi
# Step 4: Run the Hamer Pose Estimator
echo "Running Hamer Pose Estimator..."
cd /home/gomer/oline/hamer || { echo "Failed to navigate to Hamer Pose Estimator directory"; exit 1; }
HAMER_FOLDER="/home/gomer/oline/data/hamer/${VIDEO_NAME}" 

python vid_to_hamer.py "$VIDEO_PATH" "$HAMER_FOLDER"

## Step 4: Stop the server
#echo "Stopping the server..."
#kill "$SERVER_PID"
#wait "$SERVER_PID" 2>/dev/null
#echo "Server stopped."