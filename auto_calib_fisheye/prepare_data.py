import os
import glob
import shutil
import argparse
import numpy as np

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='A simple script to demonstrate argument parsing.')

    # Add arguments
    parser.add_argument('--in_dir', help='Input dir path', required=True)
    parser.add_argument('--out_dir', help='Output dir path', required=True)
    parser.add_argument('--threshold', help='Threshold for syncing set of images', required=True)
    parser.add_argument('--gap', help='Gap between two consecutive sets of image', required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_path = args.in_dir
    output_path = args.out_dir
    DIFF_THRESHOLD = int(args.threshold)
    gap = int(args.gap)
    last_stamp = 0
    count = 0

    if not os.path.exists(input_path):
        print(f'Directory "{input_path}" does not exist!')
        return
    image_0_list = glob.glob(os.path.join(input_path, 'image_0_*'))
    image_1_list = glob.glob(os.path.join(input_path, 'image_1_*'))
    image_2_list = glob.glob(os.path.join(input_path, 'image_2_*'))
    image_3_list = glob.glob(os.path.join(input_path, 'image_3_*'))

    image_0_stamps = np.array([int(filename.split('.')[0].split('_')[-1]) for filename in image_0_list])
    image_1_stamps = np.array([int(filename.split('.')[0].split('_')[-1]) for filename in image_1_list])
    image_2_stamps = np.array([int(filename.split('.')[0].split('_')[-1]) for filename in image_2_list])
    image_3_stamps = np.array([int(filename.split('.')[0].split('_')[-1]) for filename in image_3_list])

    image_0_stamps = sorted(image_0_stamps)
    print(f'Maximum set possible: {len(image_0_stamps)}')

    for stamp in image_0_stamps:
        # file_realpath = os.path.join(input_path, file)
        print(f'Stamp: {stamp}')
        if abs(stamp - last_stamp) < gap or stamp < 1692294961895:
            continue
        image_1_diff = abs(image_1_stamps - stamp);
        image_2_diff = abs(image_2_stamps - stamp);
        image_3_diff = abs(image_3_stamps - stamp);
        print(f'List of diff {np.min(image_1_diff)} - {np.min(image_2_diff)} - {np.min(image_3_diff)}')
        if (np.min(image_1_diff) > DIFF_THRESHOLD or
            np.min(image_2_diff) > DIFF_THRESHOLD or
            np.min(image_3_diff) > DIFF_THRESHOLD):
            print(f'Failed to find synced set')
            continue

        if not os.path.exists(os.path.join(output_path, f'{stamp}')):
            os.makedirs(os.path.join(output_path, f'{stamp}'))
        shutil.copy(os.path.join(input_path, f'image_0_{stamp}.jpeg'), os.path.join(output_path, f'{stamp}/b.png'))
        shutil.copy(os.path.join(input_path, f'image_1_{image_1_stamps[np.argmin(image_1_diff)]}.jpeg'), os.path.join(output_path, f'{stamp}/f.png'))
        shutil.copy(os.path.join(input_path, f'image_2_{image_2_stamps[np.argmin(image_2_diff)]}.jpeg'), os.path.join(output_path, f'{stamp}/l.png'))
        shutil.copy(os.path.join(input_path, f'image_3_{image_3_stamps[np.argmin(image_3_diff)]}.jpeg'), os.path.join(output_path, f'{stamp}/r.png'))
        last_stamp = stamp

        count += 1
    print(f'Actual number of set: {count}')    

if __name__ == '__main__':
    main()

