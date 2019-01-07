'''Track QC data processing'''
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import glob
from typing import Union, Callable
import tqdm
import warnings

from PIL import Image
import tifffile
from torchvision import transforms, utils
from scipy.misc import imresize
from sklearn.model_selection import StratifiedKFold

class TrackParser(object):
    def __init__(self,
                tracks_dir: str,
                all_tracks_glob: str,
                kept_tracks_glob: str=None,) -> None:
        '''Parse tracks to form arrays for model training and use.

        Parameters
        ----------
        track_dir : str
            path to tracks.
        all_tracks_glob : str
            pattern to match filenames of all tracks to be
            classified.
        kept_tracks_glob : str
            pattern to match filenames of tracks to be classified
            as "positive" (i.e. to be kept).

        Returns
        -------
        None.

        Notes
        -----
        Assumes that `all_tracks` and `kept_tracks` are paired when
        sorted lexographically.
        '''
        self.tracks_dir = tracks_dir
        self.all_tracks_glob  = all_tracks_glob
        self.kept_tracks_glob = kept_tracks_glob
        self.verbose = False

        self.all_tracks_fs = sorted(glob.glob(osp.join(tracks_dir, all_tracks_glob)))

        if self.kept_tracks_glob is not None:
            self.kept_tracks_fs = sorted(glob.glob(osp.join(tracks_dir, kept_tracks_glob)))
            # remove any kept tracks in all_tracks_fs
            self.all_tracks_fs = [x for x in self.all_tracks_fs if not x in self.kept_tracks_fs]

            assert len(self.all_tracks_fs) == len(self.kept_tracks_fs), \
                '#all tracks %d != #kept %d tracks' % (len(self.all_tracks_fs), len(self.kept_tracks_fs))

        self.all_tracksX = [x for x in self.all_tracks_fs if 'tracksX' in x]
        self.all_tracksY = [x for x in self.all_tracks_fs if 'tracksY' in x]
        assert len(self.all_tracksX) == len(self.all_tracksY)

        if self.kept_tracks_glob is not None:
            self.kept_tracksX = [x for x in self.kept_tracks_fs if 'tracksX' in x]
            self.kept_tracksY = [x for x in self.kept_tracks_fs if 'tracksY' in x]
            assert len(self.kept_tracksX) == len(self.kept_tracksY)

        self.load_join_tracks()

        if self.kept_tracks_glob is not None:
            self.labels = self.find_kept_tracks(self.all_tracks_NTxy,
                                                self.kept_tracks_NTxy,)
            assert self.labels.max() == 1
            assert self.labels.min() == 0
        else:
            self.labels = np.zeros(self.all_tracks_NTxy.shape[0])

        self.tracks = torch.from_numpy(self.all_tracks_NTxy).float()
        self.labels = torch.from_numpy(self.labels).long()
        return

    def load_join_tracks(self,) -> None:
        '''Load and join tracks to [N, T, 2] format'''
        self.all_tracks_NTxy = []
        self.all_tracks_origins = []
        min_t = 10000
        for i in range(len(self.all_tracksX)):
            x = np.loadtxt(self.all_tracksX[i], delimiter=',')
            y = np.loadtxt(self.all_tracksY[i], delimiter=',')
            if len(x) <=1:
                if self.verbose:
                    print('tracksX is empty')
                    print(x.shape, y.shape)
                    print('Skipping.')
                continue
            elif len(x.shape) < 2:
                if self.verbose:
                    print('tracksX and Y shapes are not 2D.')
                    print(x.shape, y.shape)
                    print('Reformatting to 2D.')
                x = x.reshape(1, x.shape[0])
                y = y.reshape(1, y.shape[0])

            if x.shape[1] < min_t:
                min_t = x.shape[1]

            xy = np.stack([x, y], axis=-1) # N, T, xy
            self.all_tracks_NTxy.append(xy)
            self.all_tracks_origins.append([i]*xy.shape[0])

        print('Concatenating tracks & clipping length to %d time steps.' % min_t)
        self.all_tracks_NTxy_clipped = []
        for xy in self.all_tracks_NTxy:
            self.all_tracks_NTxy_clipped.append(xy[:,:min_t,:])

        self.all_tracks_NTxy = np.concatenate(self.all_tracks_NTxy_clipped, axis=0)
        self.all_tracks_origins = np.concatenate(self.all_tracks_origins).astype(np.int32)

        if self.kept_tracks_glob is None:
            return
        else:
            pass

        self.kept_tracks_NTxy = []

        for i in range(len(self.kept_tracksX)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # hide empty file messages
                x = np.loadtxt(self.kept_tracksX[i], delimiter=',')
                y = np.loadtxt(self.kept_tracksY[i], delimiter=',')
            if len(x) <=1:
                if self.verbose:
                    print('tracksX is empty')
                    print(x.shape, y.shape)
                    print('Skipping.')
                continue
            elif len(x.shape) < 2:
                if self.verbose:
                    print('tracksX and Y shapes are not 2D.')
                    print(x.shape, y.shape)
                    print('Reformatting to 2D.')
                x = x.reshape(1, x.shape[0])
                y = y.reshape(1, y.shape[0])

            xy = np.stack([x, y], axis=-1) # N, T, xy
            self.kept_tracks_NTxy.append(xy[:,:min_t,:])
        self.kept_tracks_NTxy = np.concatenate(self.kept_tracks_NTxy, axis=0)
        return

    def find_kept_tracks(self,
        all_tracks: np.ndarray,
        kept_tracks: np.ndarray) -> np.ndarray:
        '''Find which tracks in `all_tracks` were kept in `kept_tracks`

        Parameters
        ----------
        all_tracks : np.ndarray
            [N_1, T, 2] tracks.
        kept_tracks : np.ndarray
            [N_2, T, 2] tracks.

        Returns
        -------
        labels : np.ndarray
            binary assignemnt if the track was kept (`1`) or discarded (`0`).
        '''
        labels = np.zeros(all_tracks.shape[0])
        for i in range(all_tracks.shape[0]):
            bidx = (all_tracks[i,:,0] == kept_tracks[:,:,0]).all(1)
            labels[i] = np.sum(bidx) > 0
        return labels

    def clean_track_files(self,
        new_labels: np.ndarray,
        save_suffix: str='_qc_model') -> None:
        '''Given a set of new labels, remove tracks marked `0` and save a new
        file for each original track file input

        Parameters
        ----------
        new_labels : np.ndarray
            [N,] binary labels for each track in `.tracks`.
        save_suffix : str
            suffix to append to new track file outputs.

        Returns
        -------
        None. Saves a new track file for each original source file.
        '''
        assert new_labels.shape[0] == self.tracks.size(0)
        idx = 0 # running counter for current position in `new_labels`
        for i in tqdm.tqdm(range(len(self.all_tracksX)),
            desc='Saving new tracks'):
            tx = np.loadtxt(self.all_tracksX[i], delimiter=',')
            ty = np.loadtxt(self.all_tracksY[i], delimiter=',')

            parent_dir_x, fbasename_x = osp.split(self.all_tracksX[i])
            parent_dir_y, fbasename_y = osp.split(self.all_tracksY[i])

            fbasename_x = osp.splitext(fbasename_x)[0]
            fbasename_y = osp.splitext(fbasename_y)[0]

            if len(tx.shape) == 1:
                tx = tx.reshape(1, tx.shape[0])
                ty = ty.reshape(1, ty.shape[0])
            elif len(tx) == 0:
                new_tx = []
                new_ty = []
            else:
                n = tx.shape[0]

                tf_labels = new_labels[idx:(idx+n)].astype(np.bool)

                new_tx = tx[tf_labels, :]
                new_ty = ty[tf_labels, :]

                # save
                np.savetxt(
                    osp.join(parent_dir_x, fbasename_x + save_suffix + '.csv'),
                    new_tx,
                    delimiter=',')
                np.savetxt(
                    osp.join(parent_dir_y, fbasename_y + save_suffix + '.csv'),
                    new_ty,
                    delimiter=',')
                idx += n
        return

def balance_classes(y: np.ndarray,
                    class_min: int=128) -> np.ndarray:
    '''
    Perform class balancing by undersampling majority classes
    and oversampling minority classes, down to a minimum value

    Parameters
    ----------
    y : np.ndarray
        class assignment indices.
    class_min : int
        minimum number of examples to use for a class.
        below this value, minority classes will be oversampled
        with replacement.

    Returns
    -------
    all_idx : np.ndarray
        indices for balanced classes. some indices may be repeated.
    '''
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))
    if min_count < class_min:
        min_count = class_min

    all_idx = [] # equal representation of each class
    for i, c in enumerate(classes):
        class_idx = np.where(y == c)[0].astype('int')
        rep = counts[i] < min_count # oversample minority classes
        if rep:
            print('Count for class %s is %d. Oversampling.' % (c, counts[i]))
        ridx = np.random.choice(class_idx, size=min_count, replace=rep)
        all_idx += [ridx]
    all_idx = np.concatenate(all_idx).astype('int')
    # shuffle classes
    all_ridx = np.random.choice(all_idx, size=len(all_idx), replace=False)
    return all_ridx

class TrackDataset(Dataset):

    def __init__(self,
                tracks: torch.FloatTensor,
                labels: torch.LongTensor,
                track_origins: np.ndarray=None,
                do_class_balancing: bool=False,
                center_tracks: bool=True,
                use_features: bool=True,
                transform: Callable=None) -> None:
        '''
        Dataset for classifying tracks for QC.

        Parameters
        ----------
        tracks : torch.FloatTensor
            [N, T, 2] track coordinates.
        labels : torch.LongTensor
            [N,] class labels.
        track_origins : np.ndarray
            [N,] int indices indicating the field-of-view file in which a track
            originated.
        do_class_balancing : bool
            balance classes by oversampling.
        center_tracks : bool
            transform tracks to all begin at the origin.
        transform : callable
            transform for samples.class_weights

        Returns
        -------
        None.
        '''
        super(TrackDataset, self).__init__()
        self.transform = transform
        self.tracks = tracks
        self.labels = labels
        self.track_origins = track_origins
        self.do_class_balancing = do_class_balancing
        self.center_tracks = center_tracks
        self.use_features = use_features

        print('Track dataset with %d tracks and labels.' % len(self.labels))

        if do_class_balancing:
            keep_idx = balance_classes(self.labels, class_min=64)
            self.tracks = tracks[keep_idx, ...]
            self.labels = labels[keep_idx]
            if track_origins is not None:
                self.track_origins = track_origins[keep_idx]
            else:
                self.track_origins = None
            print('%d samples after balancing.' % len(self.labels))
        assert self.tracks.size(0) == self.labels.size(0)

        self.orig_tracks = self.tracks
        if self.center_tracks:
            tracks = self.tracks.numpy()
            start_coords = np.tile(tracks[:,0:1,:], (1, tracks.shape[1], 1))
            self.tracks = torch.from_numpy(tracks - start_coords).float()

        return

    def calc_features(self, sample: dict,) -> torch.FloatTensor:
        '''Calculate a set of heuristic features

        Parameters
        ----------
        sample : dict
            keyed by 'input', 'start_coords'
        '''
        if not self.use_features:
            features = torch.zeros(6).float()
            return features
        track = sample['input']
        print('track', track.size())
        start_coords = sample['start_coords']
        total_dist = torch.sqrt(
            torch.sum(torch.pow(track[-1,:] - track[0,:], 2), dim=0))
        print('start_coords', start_coords.size())
        print('total_dist', total_dist.unsqueeze(0).size(), total_dist)
        disp_Txy = track[1:,:] - track[:-1,:]
        disp = torch.sqrt(torch.sum(torch.pow(disp_Txy, 2), dim=1))
        print('disp', disp.size())
        mean_disp = torch.mean(disp)
        var_disp = torch.std(disp)
        max_disp = torch.max(disp)
        min_disp = torch.min(disp)
        # N,
        features = torch.cat([start_coords,
                              total_dist.unsqueeze(0),
                              mean_disp.unsqueeze(0),
                              var_disp.unsqueeze(0),
                              max_disp.unsqueeze(0)]).float()
        return features

    def __len__(self,) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx):
        txy = self.tracks[idx,:,:]
        label = self.labels[idx]

        sample = {'input': txy,
                  'start_coords': self.orig_tracks[idx,0,:],
                  'output': label,}

        if self.transform is not None:
            sample = self.transform(sample)

        if self.track_origins is not None:
            sample['track_origin'] = self.track_origins[idx]

        features = self.calc_features(sample)
        sample['features'] = features
        return sample

class TrackImageDataset(Dataset):

    def __init__(self,
                track_ds: TrackDataset,
                img_dir: str,
                img_glob: str='*.tif',
                im_transform: Callable=None,
                bbox_sz: tuple=(150,150),) -> None:
        '''
        Dataset object for loading tracks and images concurrently
        '''
        super(TrackImageDataset, self).__init__()
        self.track_ds = track_ds
        self.img_files = sorted(glob.glob(osp.join(img_dir, img_glob)))
        self.im_transform = im_transform
        self.bbox_sz = bbox_sz

        self.__len__ = self.track_ds.__len__
        self.labels = self.track_ds.labels

        return

    def _imload(self, filename):
        ext = osp.splitext(filename)[-1]
        if 'tif' in ext:
            image = tifffile.TiffFile(filename).asarray()
        else:
            image = np.array(Image.open(filename))
        return image

    def __getitem__(self, idx: int) -> dict:
        sample = self.track_ds[idx]
        if sample.get('track_origin', None) is None:
            raise ValueError('sample must specify a track origin')
        else:
            img_idx = sample['track_origin']

        txy = self.track_ds.orig_tracks[idx,:, :] # [T, xy]
        print(txy[0,:])

        fov_image = self._imload(self.img_files[img_idx]) # [H, W, C]
        hp, wp = self.bbox_sz[0]//2, self.bbox_sz[1]//2
        fov_imagep = np.pad(fov_image, ((hp, hp), (wp, wp)), mode='reflect')

        ch, cw = txy.numpy()[0, :].astype(np.int32)
        chp = ch + hp
        cwp = cw + wp
        sample_roi = fov_imagep[chp-self.bbox_sz[0]//2 : chp+self.bbox_sz[0]//2,
                               cwp-self.bbox_sz[1]//2 : cwp+self.bbox_sz[1]//2,
                               ...]

        sample['image'] = sample_roi

        if self.im_transform is not None:
            sample = self.im_transform(sample)
        return sample

class RandomNoise(object):

    def __init__(self, sigma: float=3.) -> None:
        '''Inject white noise into tracks'''
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        xy = sample['input'] # [T, (x,y)]
        sc = sample['start_coords']
        sc = sc + torch.randn_like(sc)*self.sigma
        sample['start_coords'] = sc
        noise = torch.randn_like(xy)
        noise = noise*self.sigma
        xy_n = xy + noise
        # recenter track
        c = xy_n.numpy()[0,:]
        mask = np.tile(c.reshape(1, 2), (xy_n.size(0), 1))
        assert mask.shape == xy_n.numpy().shape, \
            '%s %s' % (str(mask.shape), str(xy_n.numpy().shape))
        xy_c = xy_n - torch.from_numpy(mask)
        sample['input'] = xy_c
        return sample


'''Image transforms'''
class ToRGB(object):
    '''Converts 1-channel grayscale images to RGB'''

    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        if len(image.shape) == 2:
            image = np.stack([image]*3, -1)
            image = np.squeeze(image)
        elif image.shape[2] == 1:
            image = np.stack([image]*3, -1)
            image = np.squeeze(image)
        elif image.shape[2] != 3 and image.shape[2] != 1:
            raise ValueError('image shape is unexpected %s' % str(image.shape))

        sample['image'] = image
        return sample

class ImageToTensor(object):
    '''Convert ndarrays in sample to Tensors'''

    def __init__(self, type: str='float', norm: Callable=None) -> None:
        self.type = type
        self.norm = norm
        return

    def __call__(self, sample: dict) -> dict:
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype('float64')
        if self.type == 'float':
            image = torch.from_numpy(image).float()

        if self.norm is not None:
            image = self.norm(image)

        sample['image'] = image
        return sample

class RandomFlip(object):
    '''Randomly flips image arrays'''

    def __init__(self, horz=True, vert=True, p=0.5):
        self.horz = horz
        self.vert = vert
        self.p = p

    def __call__(self, sample):
        image = sample['image']

        if self.horz and np.random.random() > self.p:
            image = image[:,::-1,...]

        if self.vert and np.random.random() > self.p:
            image = image[::-1,:,...]

        sample['image'] = image
        return sample

class Resize(object):
    '''Resizes images'''

    def __init__(self, size=(512, 512, 1)):
        self.sz = size

    def __call__(self, sample):
        image = sample['image']

        if len(image.shape) == 2:
            imageR = imresize(np.squeeze(image), self.sz)
        else:
            chans = []
            for c in range(image.shape[-1]):
                chanR = imresize(np.squeeze(image[...,c]), self.sz)
                chans.append(chanR)
            imageR = np.squeeze(np.stack(chans, axis=-1))

        if len(imageR.shape) < 3:
            imageR = np.expand_dims(imageR, -1)
        sample['image'] = imageR
        return sample

imgnet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

'''Transformer Zoo'''
imgnet_trans = transforms.Compose([Resize(size=(224,224,1)),
                                 RandomFlip(),
                                 ToRGB(),
                                 ImageToTensor(norm=imgnet_norm),]
                                 )
imgnet_trans_val = transforms.Compose([Resize(size=(224,224,1)),
                                 RandomFlip(),
                                 ToRGB(),
                                 ImageToTensor(norm=imgnet_norm),]
                                 )
