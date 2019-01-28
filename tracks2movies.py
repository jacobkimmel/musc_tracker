import argparse
import numpy as np
import os
import os.path as osp
import glob
from mask_parser import MaskParser
from labeled_movie_maker import TrackingMovie, ExportTrackingMovie
from functools import partial
import multiprocessing

parser = argparse.ArgumentParser(
    description='Generates movies from a set of images and tracks')
parser.add_argument('img_dir', type=str,
    help='path to image directory')
parser.add_argument('img_glob', type=str,
    help='pattern to glob image files')
parser.add_argument('track_dir', type=str,
    help='path to track directory')
parser.add_argument('track_glob', type=str,
    help='pattern to glob track files (X and Y)')
parser.add_argument('--max_xy', type=int, default=30,
    help='number of xy positions in `img_dir`')
parser.add_argument('--start_idx', type=int, default=0,
    help='starting index for xy positions')
parser.add_argument('--min_t', type=int, default=0,
    help='start time for tracking')
parser.add_argument('--max_t', type=int, default=None,
    help='end time for tracking')
parser.add_argument('--out_dir', type=str, default=None,
    help='output path for movies. defaults to img_dir.')
parser.add_argument('--frame_suffix', type=str, default=None,
    help='suffix to append to movie frames')
args = parser.parse_args()

if args.out_dir is None:
    out_dir = args.img_dir
else:
    out_dir = args.out_dir

def movie_xy(xy, args):
    print('Exporting movie frames for XY %02d' % xy)
    print('XY: ', xy)
    tfiles = sorted(glob.glob(osp.join(args.track_dir, args.track_glob)))
    assert len(tfiles) > 0
    xfiles = [x for x in tfiles if 'tracksX' in x]
    yfiles = [x for x in tfiles if 'tracksY' in x]
    assert len(xfiles) == len(yfiles)
    assert len(xfiles) > 0 and len(yfiles) > 0
    xfs = np.array([int(osp.basename(x).split('xy')[1][:2]) for x in xfiles])
    yfs = np.array([int(osp.basename(x).split('xy')[1][:2]) for x in yfiles])
    xf = np.where(xfs==xy)[0].astype(np.int32)
    yf = np.where(yfs==xy)[0].astype(np.int32)
    assert np.all(xf == yf)
    if len(xf) == 0 and len(yf) == 0:
        print('No tracks found for XY %d' % xy)
        print('Skipping...\n')
        return

    tracksX = np.loadtxt(xfiles[xf[0]], delimiter=',')
    tracksY = np.loadtxt(yfiles[yf[0]], delimiter=',')
    assert tracksX.shape[0] == tracksY.shape[0]
    print('%d tracks in XY %02d' % (tracksX.shape[0], xy))

    xy_img_glob = '*xy' + str(xy).zfill(2) + '*' + args.img_glob

    imgp = MaskParser(freq_dir=args.img_dir,
                regex=xy_img_glob,
                no_parsing=True)
    assert len(imgp.img_files) > 0, 'no image files found'
    movie_path = os.path.join(out_dir,
        args.frame_suffix + '_xy' + str(xy).zfill(2) + '_movie.avi')

    if args.min_t is None:
        min_t = 0
    else:
        min_t = args.min_t

    if args.max_t is None:
        max_t = tracksX.shape[1]
    else:
        max_t = args.max_t

    etm = ExportTrackingMovie(imgp.img_files[min_t:max_t],
            tracksX,
            tracksY,
            masks = None,
            out_file = movie_path,
            fps=5,
            suffix=args.frame_suffix)
    etm.save_frames()
    print('Finished movie for XY %d \n' % xy)
    return

part_movie_xy = partial(movie_xy, args=args)

xys = np.arange(args.start_idx, args.max_xy)
P = multiprocessing.Pool()
res = P.map(part_movie_xy, xys)
P.close()
