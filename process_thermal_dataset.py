import os
import glob
import h5py
import cv2
import os.path as op
from PIL import Image
from pathlib2 import Path

data_root = '/tsukimi/reid/Re_ID'
data_files = glob.glob(os.path.join(data_root, '*.p'))
print(len(data_files))

out_root = '/tsukimi/reid/thermal_reid/data'
processed_file_paths = []
# Build a person name to person id mapping
name_mapping = {}
for uid, path in enumerate(data_files):
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
        out_path = op.join(out_root, f'{idx:04d}_c0t{uid}.jpg')
        frame.save(out_path)
        processed_file_paths.append(out_path)

# write processed file paths to a file
with open('thermal_file_paths.txt', 'w') as f:
    for path in processed_file_paths:
        f.write(f'{path}\n')

# write name mapping to a file
with open('thermal_name_mapping.txt', 'w') as f:
    for name, uid in name_mapping.items():
        f.write(f'{name} {uid}\n')
