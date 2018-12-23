import os
import numpy as np
import glob
try:
    import cv2
except:
    print('cv2 not found, video not supported.')
from skimage.io import imread
from PIL import Image
from scipy.misc import imresize
import pandas as pd

class VidFeeder(object):
    '''
    Loads videos with OpenCV or skvideo.io depending on availability.
    Loads image sequence with skimage.io.
    Feeds preprocessed frames.

    Parameters
    ----------
    src_dir : string. path to input video or image sequence directory.
    crop : bounding box for frame cropping.
    resize : float. proportion of input image size to resize frame.
        i.e. 0.5 for half size.
    regex : string. filename pattern to match when detecting an image sequence.
        only used if `src_dir` is an image sequence directory.
    '''

    def __init__(self, src_dir, crop=None, resize=None, regex='img*.png'):
        self.src_dir = src_dir
        self.crop = crop
        self.resize = None
        self.regex = regex
        self.frame = 0
        if self.src_dir[-4:] in ['.mp4', '.avi', '.mov']:
            self.cap, self.x_res, self.y_res, self.frame_count = self._load_video()
            self.next = self._vid_next
            self.__next__ = self._vid_next
        else:
            self.img_files, self.x_res, self.y_res, self.frame_count = self._load_img_seq()
            self.next = self._imgseq_next
            self.__next__ = self._imgseq_next

        if self.crop:
            self.x_res = self.crop[2]
            self.y_res = self.crop[3]

    def __len__(self):
        return self.frame_count

    def _load_video(self):
        '''Loads video with `cv2` or `sklearn.io`'''
        try:
            import cv2
            cap = cv2.VideoCapture(self.src_dir)
            r, f = cap.read()
            assert r == True
            cap = cv2.VideoCapture(self.src_dir)
            x_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            import skvideo.io
            try:
                cap = skvideo.io.vread(self.src_dir)
            except:
                print('Cannot load video', src_dir)
                raise ValueError
            x_res, y_res = int(cap.shape[1]), int(cap.shape[2])
            frame_count = cap.shape[0]

        return cap, x_res, y_res, frame_count

    def _load_img_seq(self):
        '''Loads a sequence of images as a list of filenames'''
        img_files = glob.glob(os.path.join(self.src_dir, self.regex))
        img_files.sort()
        if len(np.array(Image.open((img_files[0]))).shape) > 2:
            x_res, y_res, channels = np.array(Image.open((img_files[0]))).shape
        else:
            x_res, y_res = np.array(Image.open((img_files[0]))).shape
        return img_files, x_res, y_res, len(img_files)

    def _crop_frame(self, frame, x_bb, y_bb, width_bb, height_bb):
        '''
        Parameters
        ----------
        frame : ndarray. (x, y, 3).
        x_bb : integer. bounding box upper left corner for cropping.
        y_bb : integer. bounding box upper right corner for cropping.
        width_bb : integer. width for bounding box.
        height_bb : integer. height for bounding box.

        Returns
        -------
        frame_c : ndarray. (height_bb, width_bb, 3).
        '''
        frame_c = frame[y_bb:y_bb+height_bb,x_bb:x_bb+width_bb, :]
        return frame_c

    def _resize_frame(self, frame, proportion):
        return imresize(frame, proportion)


    def _vid_next(self):
        '''Loads and processes next frame'''
        if type(self.cap) == np.ndarray:
            if self.frame > self.cap.shape[0]:
                print('Read final frame in VidFeerder', self.src_dir)
                return
            f = np.squeeze(self.cap[self.frame,...])
        else:
            r, f = self.cap.read()
            if not r:
                print('Read final frame in VidFeeder', self.src_dir)
                return

        if np.any(self.crop):
            fc = self._crop_frame(f, self.crop[0], self.crop[1],
                                    self.crop[2], self.crop[3])
        else:
            fc = f

        if self.resize:
            fr = self._resize_frame(fc, self.resize)
        else:
            fr = fc

        self.frame += 1
        return fr

    def _imgseq_next(self):
        f = np.array(Image.open((self.img_files[self.frame])))
        if np.any(self.crop):
            fc = self._crop_frame(f, self.crop[0], self.crop[1],
                                    self.crop[2], self.crop[3])
        else:
            fc = f

        if self.resize:
            fr = self._resize_frame(fc, self.resize)
        else:
            fr = fc

        self.frame += 1

        return fr

    def _vid_reset(self):
        self.cap, self.x_res, self.y_res, self.frame_count = self._load_video()
        self.frame = 0
        return

    def _imgseq_reset(self):
        self.frame = 0
        return

class VidFeederSplit(object):
    '''
    Loads image sequences that are split across an infrequently images sequence
    and a frequently imaged sequence.
    Assumes sequences start with an infrequent image.
    Feeds preprocessed frames.

    Parameters
    ----------
    freq_dir : string. path to frequently imaged sequence directory.
    infreq_dir : string. path to infrequently imaged sequence directory.
    crop : bounding box for frame cropping.
    resize : float. proportion of input image size to resize frame.
        i.e. 0.5 for half size.
    regex : string. filename pattern to match when detecting an image sequence.
        only used if `src_dir` is an image sequence directory.
    space_frames : integer. number of 'frequent' frames between
        'infrequent' frames.
    '''

    def __init__(self, freq_dir, infreq_dir,
                crop=None, resize=None,
                regex='*.tif', space_frames=2):
        self.freq_dir = freq_dir
        self.infreq_dir = infreq_dir
        self.crop = crop
        self.resize = None
        self.regex = regex
        self.frame = 0
        self.space_frames = space_frames
        self.freq_files, self.x_res, self.y_res, self.frame_count = self._load_img_seq(self.freq_dir)
        self.infreq_files, _, _, _ = self._load_img_seq(self.infreq_dir)

        self._merge_frequencies()

        self.next = self._imgseq_next
        self.__next__ = self._imgseq_next

        if self.crop:
            self.x_res = self.crop[2]
            self.y_res = self.crop[3]

        self.reset = self._imgseq_reset

    def __len__(self):
        return self.frame_count

    def _load_img_seq(self, src_dir):
        '''Loads a sequence of images as a list of filenames'''
        img_files = glob.glob(os.path.join(src_dir, self.regex))
        img_files.sort()
        if len(imread(img_files[0]).shape) > 2:
            x_res, y_res, channels = imread(img_files[0]).shape
        else:
            x_res, y_res = imread(img_files[0]).shape
        return img_files, x_res, y_res, len(img_files)

    def _merge_frequencies(self):

        df = pd.DataFrame(np.zeros(len(self.infreq_files) + len(self.freq_files)))
        window = self.space_frames + 1
        i = 0
        j = 0
        for t in range(df.shape[0]):
            if t % window == 0:
                df.loc[t] = self.infreq_files[i]
                i += 1
            else:
                df.loc[t] = self.freq_files[j]
                j += 1

        self.img_files = list(df.loc[:,0])
        return


    def _crop_frame(self, frame, x_bb, y_bb, width_bb, height_bb):
        '''
        Parameters
        ----------
        frame : ndarray. (x, y, 3).
        x_bb : integer. bounding box upper left corner for cropping.
        y_bb : integer. bounding box upper right corner for cropping.
        width_bb : integer. width for bounding box.
        height_bb : integer. height for bounding box.

        Returns
        -------
        frame_c : ndarray. (height_bb, width_bb, 3).
        '''
        frame_c = frame[y_bb:y_bb+height_bb,x_bb:x_bb+width_bb, :]
        return frame_c

    def _resize_frame(self, frame, proportion):
        return imresize(frame, proportion)

    def _imgseq_next(self):
        f = imread(self.img_files[self.frame])
        if np.any(self.crop):
            fc = self._crop_frame(f, self.crop[0], self.crop[1],
                                    self.crop[2], self.crop[3])
        else:
            fc = f

        if self.resize:
            fr = self._resize_frame(fc, self.resize)
        else:
            fr = fc

        self.frame += 1

        return fr

    def _imgseq_reset(self):
        self.frame = 0
        return

class DummyVidFeeder(object):
    '''Dummy vid feeder that feeds empty frames of a specified resolution'''

    def __init__(self, x_res, y_res):
        self.x_res = x_res
        self.y_res = y_res

    def next(self):
        return np.zeros((self.x_res, self.y_res)).astype('bool')
