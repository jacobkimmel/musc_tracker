'''Track cells in a given well'''

import numpy as np
import argparse
import os
import glob
from mask_parser import MaskParser
from labeled_movie_maker import TrackingMovie
from bipartite_tracker import BipartiteTracker
from tracking_fnx import track_fov
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Track cells in a target well and generate tracking videos.')
parser.add_argument('well_dir', type=str, help='path to well directory')
parser.add_argument('--img_regex', type=str, default='*c1.tif',  help='pattern to match image files')
parser.add_argument('--mask_regex', type=str, default='*csn.png', help='pattern to match mask files')
parser.add_argument('--max_xy', type=int, default=30, help='maximum number of xy positions')
parser.add_argument('--time_regex', type=str, default='%03d', help='format string for time index')
parser.add_argument('--max_dist', type=int, default=100, help='maximum distance for frame linking')
parser.add_argument('--gap_frames', type=int, default=15, help='maximum tracking gap')
parser.add_argument('--app_model', action='store_true', help='use CNN appearance model')
parser.add_argument('--min_t', type=int, default=0, help='start time for tracking')
parser.add_argument('--max_t', type=int, default=None, help='end time for tracking')
parser.add_argument('--singlethread', action='store_true', help='run tracking on a single thread. can avoid issues with matplotlib IO bottlenecking.')
parser.add_argument('--novideo', action='store_true', help='do not export tracking videos (single threaded bottleneck)')
args = parser.parse_args()

xys = [str(x).zfill(int(np.log10(args.max_xy)+1)) for x in range(1,args.max_xy + 1)]


def track(i, xys, args):
    '''Track a single `xy` in `xys`'''
    xy = xys[i]

    min_t = args.min_t
    max_t = args.max_t

    freq_dir = args.well_dir
    out_dir = freq_dir
    mask_regex = '*xy' + xy + '*' + args.mask_regex
    img_regex  = '*xy' + xy + '*' + args.img_regex

    well = os.path.split(freq_dir)[-1]
    out_lab = well + '_xy' + xy + '_'

    print('Tracking initiate : ', out_lab)
    if args.app_model:
        mp, x, y = track_fov(freq_dir, img_dir=freq_dir,
            mask_regex=mask_regex, img_regex=img_regex,
            output_dir = freq_dir,
            min_t=min_t, max_t=max_t,
            max_dist=args.max_dist, gap_frames=args.gap_frames)
    else:
        mp, x, y = track_fov(freq_dir,
        mask_regex=mask_regex,
        output_dir = freq_dir,
        min_t=min_t, max_t=max_t,
        max_dist=args.max_dist, gap_frames=args.gap_frames)

    print('Tracking complete : ', out_lab)

    # remove incomplete tracks
    xc = x[~np.isnan(x[:,-1]),:]
    yc = y[~np.isnan(y[:,-1]),:]

    np.savetxt(os.path.join(out_dir, out_lab + 'tracksX.csv'), xc, delimiter=',')
    np.savetxt(os.path.join(out_dir, out_lab + 'tracksY.csv'), yc, delimiter=',')

    return freq_dir, mp, xc, yc, img_regex, out_lab, min_t, max_t

def make_movie(freq_dir, mp, xc, yc, img_regex, out_lab, min_t, max_t):
    out_dir = freq_dir
    # Make Movie
    imgp = MaskParser(freq_dir,
                regex=img_regex, no_parsing=True)

    movie_path = os.path.join(out_dir, out_lab + 'movie.avi')
    tm = TrackingMovie(imgp.img_files[min_t:max_t],
            xc, yc,
            masks = mp.img_files[min_t:max_t],
            out_file = movie_path,
            fps=5)
    tm.save_fig()
    plt.close()

    print('Saved ', movie_path)



from functools import partial
import multiprocessing

part_track = partial(track, xys=xys, args=args)

if args.app_model or args.singlethread:
    for xy in range(len(xys)):
        track(xy, xys, args)
else:
    p = multiprocessing.Pool()

    res = p.map(part_track, range(len(xys)))

    p.close()

if not args.novideo:
    for i in range(len(res)):
        params = res[i]
        make_movie(*params)
