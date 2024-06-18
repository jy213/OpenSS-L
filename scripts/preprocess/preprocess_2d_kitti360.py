import os
import math
import multiprocessing as mp
import numpy as np
import imageio
import cv2
import random

cam_locs = ['image_00'] # Update this to match KITTI 360 camera locations
# img_size = (1408, 376) # Update this to match KITTI 360 image size
img_size = (704, 188) # Update this to match KITTI 360 image size

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(
        image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def process_one_sequence(scene):
    '''process one sequence.'''

    out_dir_color = os.path.join(out_dir, scene, 'color')
    out_dir_pose = os.path.join(out_dir, scene, 'pose')
    out_dir_K = os.path.join(out_dir, scene, 'K')
    os.makedirs(out_dir_color, exist_ok=True)
    os.makedirs(out_dir_pose, exist_ok=True)
    os.makedirs(out_dir_K, exist_ok=True)

    sequence_name = '_'.join(scene.split('_')[1:7])
    timestamp_start = scene.split('_')[-2] # take only the last timestamp
    timestamp_end = scene.split('_')[-1]

    # Convert timestamps to integers
    start = int(timestamp_start)
    end = int(timestamp_end)

    # Generate 10 random timestamps between start and end
    with open(os.path.join(data_path, sequence_name, 'cam0_to_world.txt'), 'r') as f:
        lines = f.readlines()
        all_timestamps = [int(line.split(" ")[0]) for line in lines]

    timestamps_in_range = [ts for ts in all_timestamps if start <= ts <= end]

    # random_timestamps = [str(ts).zfill(len(timestamp_start)) for ts in random.sample(timestamps_in_range, 8)]
    last_timestamps = timestamps_in_range[-8:]  # Get the last 8 timestamps
    last_timestamps = [str(ts).zfill(len(timestamp_start)) for ts in last_timestamps]  # Convert to strings with leading zeros

    for cam in cam_locs:
        counter = 0
        for i in range(0, len(last_timestamps), 4):
            for timestamp in last_timestamps[i:i+4]:
                img_name = os.path.join(data_path, sequence_name, cam, 'data_rect', timestamp + '.png')
                img = imageio.v3.imread(img_name)
                img = cv2.resize(img, img_size)
                imageio.imwrite(os.path.join(out_dir_color, cam + '_' + str(counter) + '.png'), img)

                # camera_to_world
                pose_dir = os.path.join(data_path, sequence_name, 'cam0_to_world.txt')
                lines = [(int(x.split(" ")[0]), [np.float64(i) for i in x.split(" ")[1:] if i != '']) for x in open(pose_dir).read().splitlines()]
                matching_line = next((line for index, line in lines if index == int(timestamp)), None)
                pose = np.asarray(matching_line)
                pose = pose.reshape((4, 4))  # reshape the 1D array into a 4x4 matrix
                np.save(os.path.join(out_dir_pose, cam + '_' + str(counter) + '.npy'), pose)

                # Intrinsic matrix
                K_dir = os.path.join(data_path, 'calibration', 'perspective.txt')
                K = None
                with open(K_dir, 'r') as f:
                    for line in f:
                        if (cam == 'image_00' and line.startswith('P_rect_00:')) or (cam == 'image_01' and line.startswith('P_rect_01:')):
                            # Remove the identifier, split the remaining string into a list of strings, convert each string to a float
                            values = list(map(float, line.split(':')[1].split()))
                            # Create a 3x3 matrix by ignoring the last value of each row
                            K = np.asarray([values[0:3], values[4:7], values[8:11]])
                            break
                K = adjust_intrinsic(K, intrinsic_image_dim=(1408, 376), image_dim=img_size)
                np.save(os.path.join(out_dir_K, cam + '_' + str(counter) + '.npy'), K)

                counter += 1
    print(scene, ' done')

# Update the paths in the main part of the script to point to the KITTI 360 dataset
split = 'val' # 'train' | 'val'
out_dir = '/rds/general/user/jj1220/ephemeral/openscene-main/data/kitti360/preprocessed/kitti360_2d/{}'.format(split)
data_path = '/rds/general/user/jj1220/ephemeral/openscene-main/data/kitti360/KITTI-360/data_2d_raw/' # downloaded original KITTI 360 data
timestamp_dir = '/rds/general/user/jj1220/ephemeral/openscene-main/data/kitti360/preprocessed/kitti360_3d/{}'.format(split) # used for extracting the last timestamp
scene_list = sorted([os.path.splitext(filename)[0] for filename in os.listdir(timestamp_dir)])

# process_one_sequence(scene_list[0])
os.makedirs(out_dir, exist_ok=True)

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_sequence, scene_list)
p.close()
p.join()