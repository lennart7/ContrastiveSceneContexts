import itertools
import glob
import os.path as osp
import logging
from pathlib import Path

from torch.utils.data import Dataset
from torch import load as torch_load
import numpy as np
from datasets.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type

class Atlas(VoxelizationDataset):
    NUM_LABELS = 12
    NUM_IN_CHANNEL = 3
    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter')
    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    IS_FULL_POINTCLOUD_EVAL = True
    # def __init__(
    #     self,
    #     data_root,
    #     dataset,
    #     filename_suffix,
    #     delta_t,
    #     merge=False,
    #     load_all=False,
    #     matching_type='opt_nn'
    # ):
    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 phase=DatasetPhase.Train):

        self.data_root = "/data/otoc/data/train_split_one/"
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        filename_suffix = "pth"
        datas = []
        for data_root in self.data_root:
            datas.append(glob.glob(
                osp.join(data_root, self.dataset, "*" + self.filename_suffix)
            ))
        data_paths = sorted(
            itertools.chain.from_iterable(datas)
        )
        data_paths = [data_path + '.pth' for data_path in data_paths]
        logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            data_root=self.data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=-100,
            return_transformation=config.data.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config)

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
             pointcloud[:, 6:]))
        return pointcloud

    def load_data(self, index):
        mesh_file = self.filenames[index]
        filename = mesh_file[: -len(self.filename_suffix)]

        # coords: 3d pointcloud coordinates
        # colors: rgb colors for each point
        # label: label for each point
        # label2superpoint_mapping (dict, Key=label_id, Items=superpoint_ids): per label which superpoint id belongs to it
        # superpoint_ids: for each point, the id to which superpoint it belongs to

        # (
        #     coords_and_colors,
        #     label,
        #     label2superpoint_mapping,
        #     superpoint_ids,
        #     matching_opt_nn,
        #     matching_opt,
        #     matching_nn,
        #     mean_superpoints_t,
        #     mean_superpoints_dt
        # )
        d = torch_load(mesh_file)

        coords_and_colors = d[0]
        # TODO: need to split coords into coords and colors
        coords = coords_and_colors[:, :3].astype(np.float32)
        feats = coords_and_colors[:, 3:].astype(np.float32)
        label = d[1].astype(np.int32)
        label2superpoint_mapping = d[2]
        superpoint_ids = d[3]
        if self.sampled_inds:
            __import__('ipdb').set_trace()
            # scene_name = self.get_output_id(index)
            # mask = np.ones_like(labels).astype(np.bool)
            # sampled_inds = self.sampled_inds[scene_name]
            # mask[sampled_inds] = False
            # labels[mask] = 0

        # TODO: removed some codde regarding merge here...
        # need to check this

        return data

      def save_features(self, coords, upsampled_features, transformation, iteration, save_dir):
        inds_mapping, xyz = self.get_original_pointcloud(coords, transformation, iteration)
        ptc_feats = upsampled_features.cpu().numpy()[inds_mapping]
        room_id = self.get_output_id(iteration)
        torch.save(ptc_feats, f'{save_dir}/{room_id}')
      
      def get_original_pointcloud(self, coords, transformation, iteration):
        logging.info('===> Start testing on original pointcloud space.')
        data_path = self.data_paths[iteration]
        fullply_f = self.data_root / data_path
        query_xyz, _, query_label, _  = torch.load(fullply_f)

        coords = coords[:, 1:].numpy() + 0.5
        curr_transformation = transformation[0, :16].numpy().reshape(4, 4)
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
        coords = (np.linalg.inv(curr_transformation) @ coords.T).T

        # Run test for each room.
        from pykeops.numpy import LazyTensor
        from pykeops.numpy.utils import IsGpuAvailable
        
        query_xyz = np.array(query_xyz)
        x_i = LazyTensor( query_xyz[:,None,:] )  # x_i.shape = (1e6, 1, 3)
        y_j = LazyTensor( coords[:,:3][None,:,:] )  # y_j.shape = ( 1, 2e6,3)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
        indKNN = D_ij.argKmin(1, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
        inds = indKNN[:,0]
        return inds, query_xyz
