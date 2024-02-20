import os
import glob
import shutil
import argparse
import cv2
import numpy as np

def concat_frames(frames):
    # Concatenate two frames side by side
    return np.concatenate(frames, axis=1)

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='A simple script to demonstrate argument parsing.')

    # Add arguments
    parser.add_argument('--in_dir', help='Input dir path', required=True)
    parser.add_argument('--out_dir', help='Output dir path', required=True)

    # Parse the arguments
    args = parser.parse_args()

    input_path = args.in_dir
    output_path = args.out_dir

    # Get the shape of the frames
    frame_height = 1000
    frame_width = 2000

    # Define the output video codec and create VideoWriter object
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(os.path.join(output_path, 'compare.mp4'), fourcc, 1, (frame_width, frame_height))

    if not os.path.exists(input_path):
        print(f"The directory '{input_path}' does not exist.")
        return

    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            left = cv2.imread(os.path.join(input_path, dir, 'before_all_calib.png'))
            cv2.putText(left, 'BEFORE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            right = cv2.imread(os.path.join(input_path, dir, 'after_all_calib.png'))
            cv2.putText(right, 'AFTER', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            concat_frame = concat_frames([left, right])
            out.write(concat_frame)

    out.release()

if __name__ == '__main__':
    main()
