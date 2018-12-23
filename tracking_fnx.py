import os
import glob
import numpy as np
import itertools

from mask_parser import MaskParser
from bipartite_tracker import BipartiteTracker
from vidfeeder import DummyVidFeeder, VidFeeder
from labeled_movie_maker import TrackingMovie

def track_fov(mask_dir, img_dir=None,
                mask_regex='*csn.png', img_regex=None, output_dir=None,
                img_size=(2110, 2492), min_t=25, max_t=None,
                max_dist=100, gap_frames=15):
    '''Track a single field of view, imaged with a frequent and infrequent image
    series.

    Parameters
    ----------
    mask_dir : string. directory of masks from frequent imaging.
    img_dir : string. directory of images from frequent imaging.
    mask_regex : string. pattern to match a *single* field of view in these dirs.
    img_regex : string. pattern to match a *single* field of view in these dirs.
    output_dir : string. path for tracks export.
    space_frames : integer. number of frames between infrequent images.
    img_size : tuple. size of image in pixels.
    min_t : integer. first time point for tracking.
    max_t : integer. last time point for tracking.

    Returns
    -------
    tracksX, tracksY : ndarray, (N x T).
        arrays of X and Y coordinates for tracked objects.
    '''

    # Order masks temporally in mp.img_files, compile mp.seq_centroids, mp.seq_bboxes
    mp = MaskParser(mask_dir, regex=mask_regex)

    # Set up vidfeeder
    if img_dir is not None and img_regex is not None:
        vf = VidFeeder(img_dir, regex=img_regex)
    else:
        vf = DummyVidFeeder(x_res=img_size[0], y_res=img_size[1])

    if max_t is None:
        max_t = len(mp.seq_centroids)

    bt = BipartiteTracker(vf,
                        mp.seq_centroids[min_t:max_t],
                        mp.seq_bboxes[min_t:max_t],
                        max_dist=max_dist,
                        gap_frames=gap_frames,
                        acceleration=(1,0,0)) # turn off Kalman filter

    if img_dir is None:
        bt.app_off = True # turn off appearance model
    bt.cost_weights = np.array([[2,1],[1,1],[1,1]])
    x, y = bt.link_frames(mp.seq_centroids[min_t:max_t],
                        mp.seq_bboxes[min_t:max_t],
                        vf) # start tracking
    x_f, y_f = bt.fill_gaps(x, y, N = 8) # interpolate gaps in object paths
    # if output_dir is not None:
    #     # saves as `tracksX.csv`, `tracksY.csv`
    #     # bt.save_tracks(output_dir, x_f, y_f)
    return mp, x_f, y_f
