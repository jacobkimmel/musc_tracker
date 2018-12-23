'''Interpolate masks in a directory'''
import argparse
import numpy as np
from mask_parser import MaskInterpolator
from multiprocessing import Pool
from functools import partial

parser = argparse.ArgumentParser(description='Interpolate masks.')
parser.add_argument('mask_dir', type=str, help='path to well directory')
parser.add_argument('--mask_glob', type=str, default='*densenet.png', help='pattern to match mask files')
parser.add_argument('--gap_size', type=int, default=5, help='gap sizes to interpolate')
parser.add_argument('--output_suffix', type=str, default='_temporal_smoothing', help='suffix for output files')
parser.add_argument('--max_xy', type=int, default=30, help='maximum number of xy positions')
parser.add_argument('--singlethread', action='store_true', help='perform single threaded operations')
args = parser.parse_args()


def mask_interp(xy, mask_dir, mask_glob, gap_size, output_suffix):
    
    print('INTERPOLATING XY ', str(xy).zfill(2))
    MI = MaskInterpolator(mask_dir=mask_dir, 
                          mask_glob='*xy' + str(xy).zfill(2) + mask_glob,
                          gap_size=gap_size,
                          output_suffix=output_suffix,
                         )
    MI.fill_gaps()
    MI.export_filled_masks()
    return

part_mask_interp = partial(mask_interp,
                          mask_dir=args.mask_dir,
                          mask_glob=args.mask_glob,
                          gap_size=args.gap_size,
                          output_suffix=args.output_suffix)
   
    

if args.singlethread:
    for xy in range(1, args.max_xy+1):
        print('INTERPOLATING XY ', str(xy).zfill(2))
        MI = MaskInterpolator(mask_dir=args.mask_dir, 
                              mask_glob='*xy' + str(xy).zfill(2) + args.mask_glob,
                              gap_size=args.gap_size,
                              output_suffix=args.output_suffix,
                             )

        MI.fill_gaps()
        MI.export_filled_masks()
        
else:
    p = Pool()
    res = p.map(part_mask_interp, range(1, args.max_xy + 1))
    p.close()
    
    
    