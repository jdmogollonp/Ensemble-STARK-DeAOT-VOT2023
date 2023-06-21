class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/pixstor/data/grzc7/StarkSAM'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/mnt/pixstor/data/grzc7/StarkSAM/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/mnt/pixstor/data/grzc7/StarkSAM/pretrained_networks'
        self.lasot_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/lasot'
        self.got10k_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/got10k'
        self.lasot_lmdb_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/got10k_lmdb'
        self.trackingnet_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/trackingnet'
        self.trackingnet_lmdb_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/trackingnet_lmdb'
        self.coco_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/coco'
        self.coco_lmdb_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/vid'
        self.imagenet_lmdb_dir = '/mnt/pixstor/data/grzc7/StarkSAM/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
