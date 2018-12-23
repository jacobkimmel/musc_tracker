'''
Mask parser
'''

import os
import numpy as np
from PIL import Image
import glob
import pandas as pd
from skimage.measure import label, regionprops
import re

class MaskParser(object):

    def __init__(self, freq_dir, infreq_dir=None, regex='*',
                    space_frames=2, no_parsing=False,
                    time_regex='t[0-9][0-9][0-9]'):
        '''
        Parse masks to generate sequences of centroids and bounding boxes

        Only parses *1* field-of-view at a time.
        '''
        self.freq_dir = freq_dir
        self.infreq_dir = infreq_dir
        self.regex = regex
        self.space_frames = space_frames
        self.no_parsing = no_parsing
        self.time_regex = time_regex

        if not self.no_parsing:
            self._proc_masks()
        elif self.no_parsing and self.infreq_dir is not None:
            self._load_split_imgs()
        else:
            self._load_combined_imgs()

    def _load_split_imgs(self):
        '''Load masks split between infrequent and frequent sequences'''
        self.infreq_files = glob.glob(os.path.join(self.infreq_dir, self.regex))
        self.freq_files = glob.glob(os.path.join(self.freq_dir, self.regex))
        self.infreq_files.sort()
        self.freq_files.sort()

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

    def _load_combined_imgs(self):
        '''
        Load masks all combined in one directory
        Masks must use a single time index, defined in `self.time_regex`
        '''
        self.freq_files = glob.glob(os.path.join(self.freq_dir, self.regex))
        tre = re.compile(self.time_regex)

        time_idx = [tre.search(x)[0] for x in self.freq_files]
        self.freq_files = [x for _,x in sorted(zip(time_idx,self.freq_files))]

        self.img_files = self.freq_files
        return

    def _proc_masks(self):
        if self.freq_dir is not None and self.infreq_dir is not None:
            self._load_split_imgs()
        else:
            self._load_combined_imgs()

        self._collect_centroids_bboxes()
        return

    def _collect_centroids_bboxes(self):

        self.seq_centroids = []
        self.seq_bboxes = []

        for i in range(len(self.img_files)):

            M = np.array(Image.open(self.img_files[i]))
            P = regionprops(label(M))

            centroids = np.zeros((len(P), 2))
            bboxes = np.zeros((len(P), 4))
            for obj in range(len(P)):
                centroids[obj,:] = P[obj]['centroid']
                bboxes[obj,:] = P[obj]['bbox']

            self.seq_centroids.append(centroids)
            self.seq_bboxes.append(bboxes)

        return
