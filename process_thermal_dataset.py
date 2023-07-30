import os
import glob
import h5py
import random
import shutil
import cv2
import os.path as op
from PIL import Image
from pathlib2 import Path
from tqdm import tqdm

data_root = '/tsukimi/reid/Re_ID'
data_files = glob.glob(os.path.join(data_root, '*.p'))
print(len(data_files))

out_root = '/workspace/deep-person-reid/reid-data/thermal_reid/data'
processed_file_paths = []
# Build a person name to person id mapping
total_idx = 0
name_mapping = {}
for uid, path in tqdm(enumerate(data_files)):
    name = Path(path).stem.split('_')[-1]
    name_mapping[name] = uid

    data = h5py.File(path, 'r', libver="latest")
    thermal=(data["thermal"][:])
    t, h, w=thermal.shape

    for idx in range(t):
        frame = thermal[idx]
        # frame = cv2.resize(frame, (128, 128))
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = Image.fromarray(frame)
        out_path = op.join(out_root, f'{total_idx:05d}_{idx:05d}_{uid}.jpg')
        frame.save(out_path)
        processed_file_paths.append(out_path)
        total_idx += 1

# write processed file paths to a file
with open('thermal_file_paths.txt', 'w') as f:
    for path in processed_file_paths:
        f.write(f'{path}\n')

# write name mapping to a file
with open('thermal_name_mapping.txt', 'w') as f:
    for name, uid in name_mapping.items():
        f.write(f'{name} {uid}\n')


# Split the dataset into train, query and gallery
train_ratio, gallery_ratio, query_ratio = 0.35, 0.55, 0.1
processed_file_names = os.listdir(out_root)
random.shuffle(processed_file_names)
train_file_names = processed_file_names[:int(len(processed_file_names)*0.35)]
gallery_file_names = processed_file_names[int(len(processed_file_names)*0.35):int(len(processed_file_names)*0.9)]
query_file_names = processed_file_names[int(len(processed_file_names)*0.9):]

# move train, query, gallery to corresponding folders
thermal_data_root = '/workspace/deep-person-reid/reid-data/thermal_reid'
train_data_dir = op.join(thermal_data_root, 'train')
query_data_dir = op.join(thermal_data_root, 'query')
gallery_data_dir = op.join(thermal_data_root, 'gallery')
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(query_data_dir, exist_ok=True)
os.makedirs(gallery_data_dir, exist_ok=True)

# move train data
for name in tqdm(train_file_names):
    old_path = op.join(out_root, name)
    new_path = op.join(train_data_dir, name)
    shutil.move(old_path, new_path)

# move query data
for name in tqdm(query_file_names):
    old_path = op.join(out_root, name)
    new_path = op.join(query_data_dir, name)
    shutil.move(old_path, new_path)

# move gallery data
for name in tqdm(gallery_file_names):
    old_path = op.join(out_root, name)
    new_path = op.join(gallery_data_dir, name)
    shutil.move(old_path, new_path)