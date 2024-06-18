import os
# import multiprocessing as mp
import numpy as np
import plyfile
# import torch

# Map classes to {0,1,...,18}, and ignored classes to 255
remapper = np.ones(46) * 255
for i, x in enumerate([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]):
    remapper[x] = i
remapper[34] = 2
remapper[35] = 4
remapper[37] = 5

def process_kitti360(fn):
    ''' Process a single KITTI360 point cloud file ''' 
    parts = fn.split('/')
    scene_name = '_'.join(parts[-4:-1] + [os.path.splitext(parts[-1])[0]])
    output_file = os.path.join(out_dir, scene_name + '.pth')

    # Check if the output file already exists
    if os.path.exists(output_file):
        print("File already processed: ", fn)
        return

    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    # colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1
    category_id = np.ascontiguousarray(v[:, 6]).astype(int)  

    remapped_labels = remapper[category_id]
    torch.save((coords, 0, remapped_labels), output_file)
    
    print("Successfully processed: ", fn)

split = 'train' # 'train' | 'val'
raw_data_dir = '/rds/general/ephemeral/user/jj1220/ephemeral/openscene-main/data/kitti360/raw/data_3d_semantic/{}'.format(split) # downloaded original kitti360 data
out_dir = '/rds/general/user/jj1220/ephemeral/openscene-main/data/kitti360/preprocessed/{}'.format(split)
os.makedirs(out_dir, exist_ok=True)

scene_list = [scene for scene in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, scene))]

print(scene_list)
# files = []
# for scene in scene_list:
#     file_path = os.path.join(raw_data_dir, scene, 'static')
#     for ply in os.listdir(file_path):
#         files.append(os.path.join(file_path, ply))
# print(files)
# p = mp.Pool(processes=mp.cpu_count())
# p.map(process_kitti360, files)
# p.close()
# p.join()