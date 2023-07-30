import sys
import os
import os.path as osp
import numpy as np
from pathlib2 import Path

import torchreid
from torchreid.data import ImageDataset


class ThermalDataset(ImageDataset):
    dataset_dir = 'thermal_reid'

    def __init__(self, root='/workspace/deep-person-reid/reid-data', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        train_dir = osp.join(self.dataset_dir, 'train')
        query_dir = osp.join(self.dataset_dir, 'query')
        gallery_dir = osp.join(self.dataset_dir, 'gallery')

        train_paths = [osp.join(train_dir, name) for name in os.listdir(train_dir)]
        query_paths = [osp.join(query_dir, name) for name in os.listdir(query_dir)]
        gallery_paths = [osp.join(gallery_dir, name) for name in os.listdir(gallery_dir)]

        #! Notice, since we only have one camera, but in order to make a validation happen in this code, we need to have at least two cameras
        #! Therefore, we will assign the int(Path(path).stem.split('_')[0])%2 to be the camera id, which will be 0 or 1 
        train = [(path, int(Path(path).stem.split('_')[-1]), int(Path(path).stem.split('_')[0])%2) for path in train_paths]
        query = [(path, int(Path(path).stem.split('_')[-1]), int(Path(path).stem.split('_')[0])%2) for path in query_paths]
        gallery = [(path, int(Path(path).stem.split('_')[-1]), int(Path(path).stem.split('_')[0])%2) for path in gallery_paths]

        super(ThermalDataset, self).__init__(train, query, gallery, **kwargs)

torchreid.data.register_image_dataset('thermal_dataset', ThermalDataset)
# use your own dataset only
datamanager = torchreid.data.ImageDataManager(
    root='/workspace/deep-person-reid/reid-data',
    sources='thermal_dataset',
    height=240,
    width=320,
    batch_size_train=32,
    batch_size_test=100,
    transforms=["random_flip", "random_crop"]
)

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="softmax",
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)
engine.run(
    save_dir="log/osnet_x1_0",
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)