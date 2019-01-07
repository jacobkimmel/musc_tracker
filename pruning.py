'''
Manually screen tracking videos
'''
import numpy as np
import os
import os.path as osp
import glob

def get_occupied_fovs(track_dir: str, tracks_glob: str='*') -> (list, list):
    '''
    Returns a list of occupied XY positions

    Parameters
    ----------
    tracks_dir : str
        path to directory containing tracks.
    tracks_glob : str
        pattern to match track files.

    Returns
    -------
    pruning_fovs : list of integers.
        integers representing FOV position numbers to be viewed.
    fov_names : list
        names of FOVs.
    '''
    tXfiles = sorted(glob.glob(
        os.path.join(track_dir, tracks_glob + 'tracksX.csv')))
    tYfiles = sorted(glob.glob(
        os.path.join(track_dir, tracks_glob + 'tracksY.csv')))

    fov_names = [os.path.basename(x).split('tracks')[0] for x in tXfiles]
    pruning_fovs = []
    for fov in range(len(tXfiles)):
        f = open(tXfiles[fov], 'r')
        if len(f.read()) != 0:
            pruning_fovs.append(fov)
        else:
            pruning_fovs.append(None)
    return pruning_fovs, fov_names


def manual_screening(track_dir: str,
                    pruning_fovs: list,
                    fov_names: list,
                    movies_glob: str='*.avi',
                    save_in_progress: str=None,
                    noreview: bool=False,
                    start_idx: int=0) -> np.ndarray:
    '''
    Shows tracking movies and collects user input re:
    which tracks to eliminate.

    Parameters
    ----------
    track_dir : str
        path to directory containing tracking movies.
    pruning_fovs : list of integers.
        integers representing FOV position numbers to be viewed.
    fov_names : list
        names of FOVs.
    save_in_progess : str, optional.
        path to directory for in progress CSV exports.
        save pruned tracks to a separate file for each FOV
        as they are collected. each CSV is named by FOV in `fov_name` and
        contains a (1,30) row of tracks to keep padded by `-1`.
    noreview : bool
        skip viewing videos, just collect user inputs.
    start_idx : int
        FOV to start screening.

    Returns
    -------
    prune_fov : np.ndarray.
        N x 30 array, where N is the number of FOVs.
        FOVs positions are indexed by rows.
        * A value of 111 in a row indicates all tr        fov_names.append()
acks in that FOV are valid.
        * A -111 in a row indicates all tracks are invalid.
        * Values >0 indicate tracks **to keep** from each FOV.
        * Values <0 indicate tracks **to remove** from each FOV.
        * Values ==-999 indicate that index `0` should be removed.
    '''
    print('Reviewing ', track_dir)
    print('-'*20)
    nb_fovs = len(fov_names)
    prune_fov = np.ones([nb_fovs, 30])
    prune_fov *= -111

    movie_files = sorted(glob.glob(os.path.join(track_dir, movies_glob)))

    i = start_idx
    while i < len(fov_names):
        if i not in pruning_fovs:
            print('No cells to process in ', fov_names[i])
            print('Index %d' % i)
            if save_in_progress is not None:
                out = np.ones((1, 30))*-111
                np.savetxt(
                    os.path.join(save_in_progress, fov_names[i] + 'pruneFOV.csv'),
                    out,
                    delimiter=',')
                print('Saved FOV ',
                    os.path.join(save_in_progress, fov_names[i] + 'pruneFOV.csv'))
            i += 1
            continue
        print('Processing FOV %s' % fov_names[i])
        print('Index %d, Pruning FOV Index %d' % (i, pruning_fovs[i]))
        vfile = movie_files[i]
        print('Opening video file %s' % vfile)
        # 2> /dev/null pipes vlc library verbosity away
        if not noreview:
            os.popen('vlc ' + vfile + ' 2> /dev/null')
        print('Commands (separate multiple commands with spaces):')
        print('R\t:\tReload video. Useful to get around VLC bugs.')
        print('S\t:\tSkip to a specific field of view.')
        print('+111\t:\tKeep all tracks.\n\t\t(Overrides all other commands.)')
        print('-111\t:\tDiscard all tracks.\n\t\t(Overrides all following commands.)')
        print('+[0-110]\t:\tKeep the track designated by this int marker.\n\t\t(All others discarded by default.)')
        print('-[0-110]\t:\tDiscard the track designated by this int marker.\n\t\t(All others saved by default.)')
        print('-999\t:\tSpecial case. Remove index zero, since it cannot be made negative.')
        var = input('Enter commands\t>> ')
        if var == 'S':
            skip = input('Enter the FOV number to skip to\t>> ')
            i = int(skip)
            continue

        while var == '' or var == 'R' or var == 'r':
            os.popen('vlc ' + vfile + ' 2> /dev/null')
            var = input('Pruning class : >> ')
        var = str(var)
        n = [int(x) for x in var.split()]
        print('Saving %s to index %i in prune_fov' % (fov_names[i], i))
        prune_fov[i, 0:len(n)] = n

        if save_in_progress is not None:
            out = np.ones((1, 20))*-111
            out[0,0:len(n)] = n
            np.savetxt(
                os.path.join(save_in_progress, fov_names[i] + 'pruneFOV.csv'),
                out,
                delimiter=',')
            print('Saved FOV ',
                os.path.join(save_in_progress, fov_names[i] + 'pruneFOV.csv'))
        i += 1

    return prune_fov

def save_prune_fov(track_dir: str, prune_fov: np.ndarray) -> None:
    '''Save `prune_fov` as a CSV'''
    np.savetxt(os.path.join(track_dir, 'prune_fov.csv'),
        prune_fov,
        delimiter=',')
    return

def remove_pruned_tracks(track_dir: str,
                         tracks_glob: str,
                         prune_fovs: np.ndarray,
                         fov_names: list,
                         verbose: bool=True) -> None:
    '''
    Removes indicated tracks, as recorded in pruneXY

    Parameters
    ----------
    track_dir : string. path to directory with tracking movies.
    prune_fovs : np.ndarray.
        N x 30 array, where N is the number of FOVs.
        FOVs positions are indexed by rows.
        * A value of 111 in a row indicates all tracks in that FOV are valid.
        * A -111 in a row indicates all tracks are invalid.
        * Values >=0 indicate tracks **to keep** from each FOV.
        * Values <0 indicate tracks **to remove** from each FOV.
        * Values ==-999 indicate that index `0` should be removed.
    fov_names : list.
        list of FOV names.

    Returns
    -------
    None.
    '''
    tXfiles = sorted(glob.glob(
        osp.join(track_dir, tracks_glob + 'tracksX.csv')))
    tYfiles = sorted(glob.glob(
        osp.join(track_dir, tracks_glob + 'tracksY.csv')))
    if verbose:
        print('tXfiles: ', len(tXfiles))
        print('prune_fovs shape: ', prune_fovs.shape)

    for i in range(prune_fovs.shape[0]):
        tracksX = np.loadtxt(
            tXfiles[i],
            delimiter = ',')
        tracksY = np.loadtxt(
            tYfiles[i],
            delimiter = ',')
        if len(tracksX.shape) == 1:
            tracksX = tracksX.reshape(1, tracksX.shape[0])
            tracksY = tracksY.reshape(1, tracksY.shape[0])

        p = prune_fovs[i, :]

        if verbose:
            print('track files', tXfiles[i])
            print('prune_fov vector: ', p)

        if p[0] == 111:
            # keep all tracks
            pkeep = np.arange(tracksX.shape[0]).astype(np.int32)
        elif p[0] == -111:
            # discard all tracks
            tracksXpruned = []
            tracksYpruned = []
            pkeep = np.array([], dtype=np.int32)
        elif np.sum(p >= 0) > 0:
            # some positive values were directly specified
            # keep these tracks, remove others
            pkeep = p[p >= 0].astype(np.int32)
            # remove any incorrectly entered indices that
            # are out of the proper range
            out_of_bounds = pkeep > tracksX.shape[0]
        elif np.sum(np.logical_and(p < 0, p != -111)) > 0:
            # some negative values for track removal were specified
            # remove these values
            pdiscard = p[np.logical_and(p < 0, p != -111)]
            pdiscard[pdiscard==-999] = 0 # `-999` is an alias to remove index `0`
            pdiscard = np.abs(pdiscard) # flip values to act as int indices
            pkeep = np.setdiff1d(np.arange(tracksX.shape[0]),
                                pdiscard).astype(np.int32)
        else:
            raise ValueError('`prune_fovs` values not understood: %s' % str(p))

        print('pkeep', pkeep, 'tracksX shape', tracksX.shape)
        tracksXpruned = tracksX[pkeep.astype('int'), :]
        tracksYpruned = tracksY[pkeep.astype('int'), :]

        np.savetxt(
            os.path.join(track_dir, fov_names[i] + 'tracksX_pruned.csv'),
            tracksXpruned,
            delimiter=',')
        np.savetxt(
            os.path.join(track_dir, fov_names[i] + 'tracksY_pruned.csv'),
            tracksYpruned,
            delimiter=',')
        np.savetxt(
            os.path.join(track_dir, fov_names[i] + 'tracks_keep_idx.csv'),
            pkeep,
            delimiter=',')
    return

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Perform manual QC of tracking \
                                results and "prune" away poor quality tracks.')
    parser.add_argument('track_dir', type=str,
            help='path containing trcaking movies')
    parser.add_argument('--tracks_glob', type=str, default='*',
            help='pattern to match track filenames')
    parser.add_argument('--movies_glob', type=str, default='*.avi',
            help='pattern to match movie filenames')
    parser.add_argument('--clear_tracks', action='store_true',
            help='clear prunedXYs from tracks')
    parser.add_argument('--noreview', action='store_true',
            help='No video review, just prune tracks')
    parser.add_argument('--start_idx', type=int, default=0,
            help='FOV index to start reviewing. Default = 0.')
    args = parser.parse_args()

    track_dir = args.track_dir
    tracks_glob = args.tracks_glob
    movies_glob = args.movies_glob

    if args.clear_tracks and not args.noreview:
        remove_pruned_tracks(track_dir, tracks_glob, prune_fov, fov_names)
    elif args.clear_tracks and args.noreview:
        prune_fov = np.loadtxt(os.path.join(track_dir, 'prune_fov.csv'), delimiter = ',')
        pruning_fovs, fov_names = get_occupied_fovs(track_dir, tracks_glob)
        remove_pruned_tracks(track_dir, tracks_glob, prune_fov, fov_names)
    else:
        pruning_fovs, fov_names = get_occupied_fovs(track_dir, tracks_glob)
        prune_fov = manual_screening(track_dir,
                    pruning_fovs,
                    fov_names,
                    movies_glob=movies_glob,
                    save_in_progress=track_dir,
                    noreview=args.noreview,
                    start_idx=args.start_idx)
        save_prune_fov(track_dir, prune_fov)

if __name__ == '__main__':
    main()
