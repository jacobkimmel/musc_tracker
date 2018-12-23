'''
General bipartite graph matching tracker
'''

import numpy as np
from skimage.io import imread
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
import os, glob
import scipy
from multiprocessing import Pool
from functools import partial
import pandas as pd
from vidfeeder import VidFeeder

from scipy.spatial.distance import cosine as cosine_dist
from scipy.misc import imresize

# alias for legacy support
VidFeeder = VidFeeder

def df_to_seq_centroids(df):
    '''Outputs list of centroid arrays from dataframe'''

    seq_centroids = []
    seq_soft_centroids = []
    seq_bboxes = []

    T = df['t'].max()
    for t in range(int(T+1)):
        seq_centroids.append( np.array(df[df['t']==t].iloc[:,1:3]).astype('int32') )
        seq_soft_centroids.append( np.array(df[df['t']==t].iloc[:,3:5]).astype('int32') )
        seq_bboxes.append( np.array(df[df['t']==t].iloc[:,5:]).astype('int32') )

    return seq_centroids, seq_soft_centroids, seq_bboxes


class BipartiteTracker(object):
    '''
    Tracks objects from a given image directory using bipartite graph matching.

    Parameters
    ----------
    vid_feeder : VidFeeder object.
        object that feeds new video frames.
    seq_centroids : list of ndarrays.
        temporally ordered list of detection centroids.
    seq_bboxes : list of ndarrays.
        temporally ordered list of detection bounding boxes.
    gap_frames : integer.
        number of frames to allow for buffer-and-recover upon occlusion.
    max_dist : float.
        maximum Euclidean distance between linked detections in pixels.
    window_sz : integer.
        size of ROI windows for appearance model. Should be ~same size as a
        single object.
    acceleration : tuple.
        (dt, a, max_dt) where
        `dt` is an integer denoting the time to begin applying
        acceleration
        `a` is a float for the acceleration exponent in:
            dt = dt**a
        `max_dt` is the maximum number of time steps to project a location before
        holding constant.
    verbose : boolean. optional, default = `False`.
        print cost matrices, position read predictions, bboxes to `stdout`.

    Usage
    -----
    ```
    bt = BipartiteTracker(vid_feeder, seq_centroids, seq_bboxes) # init tracker
    x, y = bt.link_frames(seq_centroids, seq_bboxes, vid_feeder) # start tracking
    x, y = bt.fill_gaps(x, y) # interpolate gaps in object paths
    bt.save_tracks(output_dir, x, y) # saves as `tracksX.csv`, `tracksY.csv`
    ```

    Notes
    -----
    * Links detections between frames of a tracking video as a linear assignment
    problem.
    * Cost matrices are generated considering Euclidean distances, bounding box
    overlaps, a CNN based appearance model, and a motion model.
    * Solves for minimal cost using the Munkes/Hungarian algorithm.
    * Handles occlusion using a "buffer-and-recover" system, parametrized by
    `gap_frames`. See (Luo 2017, Multiple Object Tracking: Literature Review).
    * Kalman filtering is performed to estimate the future location of objects
    based on a linear motion model. Kalman predictions are used as the buffer for
    "buffer-and-recover".
    '''

    def __init__(self, vid_feeder, seq_centroids, seq_bboxes,
                    gap_frames=15, max_dist = 10., window_sz = 20,
                    acceleration = (7, 0.9, 4), verbose=False):

        self.epsilon = 1e-5
        self.vid_feeder = vid_feeder
        self.seq_centroids = seq_centroids
        self.seq_bboxes = seq_bboxes
        self.gap_frames = gap_frames
        self.max_dist = max_dist
        self.window_sz = window_sz
        self.acceleration = acceleration
        self.verbose = verbose
        self.verbosity = 1000 # print linking every N frames
        self.app_off = False # calculate appearance model
        self.object_permanence = False # don't assume permanence
        # bbox guess size for assumed positions
        self.bbsz = 20 # 1/2 length of square side, i.e. 20 --> 40x40 window
        # cost matrix weights (coeff, exp) for (distance, bbox, appearance)
        self.cost_weights = np.array([[1,1],[1,1],[1,1]])
        if self.app_off:
            self.cost_weights[2,0] = 0 # 0 out appearance weights
        self.save_costs = True
        if self.save_costs:
            self.dms = []
            self.acms = []
            self.bbs = []
        # intialize kalman filter parameters
        self.P = np.eye(4)*1.e-1 # init covariance estimates for each obj
        self.Q = np.eye(4)*1e-2 # process variance estimate
        self.R = np.eye(2)*1e-2 # measurement variance estimate

    def _kalman_update(self, x_k0, P_k0, z=None, dt=1, bounds_check=True):
        '''
        Uses a Kalman filter to determine likeliest object state x_k1 given
        previous state `x_k0`, measurement `z`, time step `k`.
        If only the previous state `x_k0` is provided, returns the estimated
        next state `x_k1minus` and estimated covariance matrix `P_k1minus`.

        Parameters
        ----------
        x_k0 : ndarray.
            4 x 1 matrix of cell position (x,y) and velocity (dx, dy)
            [x, y, dx, dy]
        P_k0 : ndarray.
            model covariances at time `k0`.
        z : ndarray.
            [x, y, dx, dy] measurement at k1.
        dt : integer. time steps since last measurement.
        bounds_check : boolean.
            enforces boundaries, preventing model from predicting out-of-frame
            positions. uses `self.vid_feeder.crop`

        Returns
        -------
        x_k1hat : ndarray.
            4 x 1 matrix of cell position (x,y) and velocity (dx, dy)
            [x, y, dx, dy]
        '''
        # update matrix A performs linear model
        # position1 = position0 + velocity0 * time
        if dt > self.acceleration[0]:
            dt = dt**self.acceleration[1]
            dt = np.max([self.acceleration[2], dt])

        A = np.array([
                    [1,0,dt,0],
                    [0,1,0,dt],
                    [0,0,1,0],
                    [0,0,0,1]
                    ])
        H = np.array([[1,0,0,0], [0,1,0,0]])
        x_k0 = np.array(x_k0) # ensure x_k0 is an array (4,)
        assert x_k0.shape[0] == 4

        # (1) Project state ahead: x_k1minus = Ax_k0
        x_k1minus = np.dot(A, x_k0) # a priori estimate
        # (2) Project error covar ahead: P_k1minus = A(P_k0)A^T + Q
        P_k1minus = A.dot(P_k0).dot(A.T) + self.Q

        # if detection was not provided, return the estimated position / covar
        if not np.any(z):
            if bounds_check:
                x_k1minus[0] = np.clip(x_k1minus[0], 0, self.vid_feeder.x_res)
                x_k1minus[1] = np.clip(x_k1minus[1], 0, self.vid_feeder.y_res)

            return x_k1minus, P_k1minus

        # (3) Compute Kalman Gain: K_k1 = P_k1minus(H^T)(HP_k1minusH^T+R)^-1
        # H = 1 since measurements are direct
        # K_k1 = P_k1minus*(P_k1minus + R)^-1
        K_k1 = P_k1minus.dot(H.T).dot(np.linalg.inv((H.dot(P_k1minus).dot(H.T) + self.R)))
        # (4) Update estimate with measurement z
        # x_k1hat = x_k1minus + K_k1(z - Hx_k1minus)
        x_k1hat = x_k1minus + K_k1.dot(z - H.dot(x_k1minus))
        # (5) Update error covariance
        # P_k1 = (I-K_k1*H)*P_k1minus
        P_k1 = (np.eye(K_k1.shape[0]) - K_k1.dot(H)).dot(P_k1minus)

        if bounds_check:
            x_k1hat[0] = np.clip(x_k1hat[0], 0, self.vid_feeder.x_res)
            x_k1hat[1] = np.clip(x_k1hat[1], 0, self.vid_feeder.y_res)

        return x_k1hat, P_k1

    def _kalman_guess(self, pos0, t):
        '''
        Estimates position of object `obj_i` at `t` based on the past state of
        the particle.

        Parameters
        ----------
        t : integer.
            time step.

        Returns
        -------
        x_k1_guesses : ndarray, N x 4.
            [x, y, dx, dy] best guess for object at time `t`.
        gaps_filled : ndarray, N x 2.
            [i, dt] for all objects.
            dt > 1 indicates a filled gap.
        '''
        x_k1_guesses = np.zeros((pos0.shape[0], 4))

        gaps_filled = np.zeros((pos0.shape[0], 2))

        for i in range(pos0.shape[0]):
            # if track didn't link in previous frame and stayed at `-1`
            # it may have missed detection
            if pos0[i,0] < 0:
                # find indices where tracksX has real non-negative values i.e. last detection
                idx = np.where(self.tracksX[i,:] > 0)
                dt = t - idx[0].max() + 1
                # if the detection is less than `gap_frames`, project with guess

                tl = (t+1) - dt
                x_k0 = np.squeeze(self.x_k[tl][i,:])
                P_k0 = np.squeeze(self.P[tl][i,:,:])
                x_k1minus, P_k1minus = self._kalman_update(x_k0, P_k0, z=None, dt=dt)

                x_k1_guesses[i,:] = x_k1minus
                gaps_filled[i,:] = [i, dt] # keep track of which indices we're gap filling
            else:
                x_k0 = np.squeeze(self.x_k[t][i,:])
                P_k0 = np.squeeze(self.P[t][i,:,:])
                x_k1minus, P_k1minus = self._kalman_update(x_k0, P_k0, z=None, dt=1)
                x_k1_guesses[i,:] = x_k1minus # only (x,y)
                gaps_filled[i,:] = [i, 1]

        return x_k1_guesses, gaps_filled

    def _kalman_smooth(self, pos1, t, gaps_filled):
        '''
        Given a set of new observations in `pos1`, calculates the smoothed Kalman
        filter state `x_k1` for time `t+1`.

        Parameters
        ----------
        pos1 : ndarray, N x 2.
            ordered list of new detection locations at time `t+1`.
            non-linked rows are marked with `-1`.
        t : integer.
            time point.
        gaps_filled : ndarray, N x 2.
            [i, dt] for all objects.
            dt > 1 indicates a filled gap.

        Returns
        -------
        x_k1_states : ndarray, N x 4.
            [x, y, dx, dy] of updated object states at time `t+1`.
        P_k1_states : ndarray, N x 4 x 4.
            covariance matrices for objects in N at time `t+1`.
        '''

        x_k1_states = np.zeros((pos1.shape[0], 4))
        P_k1_states = np.zeros((pos1.shape[0], 4, 4))

        for i in range(pos1.shape[0]):

            tl = np.int32((t+1) - gaps_filled[i,1]) # dt between t+1 and last x_k
            x_k0 = np.squeeze(self.x_k[tl][i,:])
            P_k0 = np.squeeze(self.P[tl][i,:,:])

            if pos1[i,0] > 0:
                x_k1, P_k1 = self._kalman_update(x_k0, P_k0, z=pos1[i,:], dt=gaps_filled[i,1])
            else:
                x_k1 = np.ones(4)*-1
                P_k1 = np.ones((4,4))*-1

            x_k1_states[i, :] = x_k1
            P_k1_states[i,:,:] = P_k1



        return x_k1_states, P_k1_states

    def _bbox_overlap(self, bbox0, bbox1):
        '''
        Calculates overlap of bounding boxes as intersection / union
        '''
        # bboxes = (row_min, col_min, row_max, col_max)

        # height = min(row_max0, row_max1) - max(row_min0, row_min1)
        h = np.min([bbox0[2], bbox1[2]]) - np.max([bbox0[0], bbox1[0]])
        # width = min(col_max0, col_max1) - max(col_min0, col_min1)
        w = np.min([bbox0[3], bbox1[3]]) - np.max([bbox0[1], bbox1[1]])

        h = np.max([0, h])
        w = np.max([0, h])

        intersection = np.max([self.epsilon, h*w]) # use an epsilon val to avoid / by 0 after inversion

        # size bbox : row_max - row_min * col_max - col_min
        union = (bbox0[2] - bbox0[0])*(bbox0[3]-bbox0[1]) + (bbox1[2] - bbox1[0])*(bbox1[3]-bbox1[1])

        iou = intersection/union # intersection over union
        return iou

    def _bbox_pad_simple(self, bbox0, pos0):
        '''
        Pads `bbox0` using simple windows around missing points in `pos0`

        Parameters
        ----------
        bbox0 : ndarray. M x 4.
            (row_min, col_min, row_max, col_max), where M < N.
        pos0 : ndarray. N x 2.
            (x, y).
        '''
        from scipy.spatial import distance_matrix

        pos0p = pos0[:,:2]
        if np.any(bbox0):
            bbox0_centers = np.vstack([(bbox0[:,2] + bbox0[:,0])//2, (bbox0[:,3] + bbox0[:,1])//2]).T
            dm = distance_matrix(bbox0_centers, pos0p)

            # Find nearest partners based on center distances
            nearest_points_to_bbox = np.argmin(dm, 1)
        else:
            nearest_points_to_bbox = []

        new_bbox0 = np.zeros((len(pos0), 4))
        j = 0
        for i in nearest_points_to_bbox:
            new_bbox0[i,:] = bbox0[j,:]
            j += 1

        empty_idx = np.where( np.sum(new_bbox0 == 0, 1) == 4 )[0]
        for i in empty_idx:
            new_point = [np.max([pos0p[i,0] - self.bbsz, 0]),
                        np.min([pos0p[i,0] + self.bbsz, self.vid_feeder.x_res]),
                        np.max([pos0p[i,1] - self.bbsz, 0]),
                        np.min([pos0p[i,1] + self.bbsz, self.vid_feeder.y_res])]

            new_bbox0[i,:] = new_point

        return new_bbox0

    def _bbox_pad(self, bbox0, pos0, gaps_filled):
        '''
        Pads `bbox0` using information from the past in the event that `pos0`
        was padded.

        Find bboxes in `bbox0` with highest overlap to past, uses those. Pads from
        the past for boxes with least overlap.

        Parameters
        ----------
        bbox0 : ndarray. M x 4.
            (row_min, col_min, row_max, col_max), where M < N.
        pos0 : ndarray. N x 2.
            (x, y).
        gaps_filled : ndarray. N x 2.
            (idx, dt), where dt is frames since last linkage.
            1 : lined last frame, 0 : not linking anymore.

        Returns
        -------
        bbox0 : ndarray. N x 4.
            (row_min, col_min, row_max, col_max).
        '''
        from sklearn.metrics import pairwise_distances

        fill_idx = np.where(gaps_filled[:,1] > 1)[0] # dt > 1
        last_ts = gaps_filled[fill_idx, 1].astype('int32')

        past_bboxes = np.zeros((len(fill_idx), 4))
        # if an object is disappearing and reappearing,indices in the past may
        # not match. in this case, just use a simple SxS box around the
        # predicted position
        for i in range(len(fill_idx)):
            try:
                past_bboxes[i,:] = self.seq_bboxes[last_ts[i]][fill_idx[i],:]
            except:
                past_bboxes[i,:] = [pos0[fill_idx[i],0]-self.bbsz,
                                    pos0[fill_idx[i],1]-self.bbsz,
                                    pos0[fill_idx[i],0]+self.bbsz,
                                    pos0[fill_idx[i],1]+self.bbsz]

        # IoU, higher is more overlap
        bb = pairwise_distances(past_bboxes, bbox0, metric=self._bbox_overlap)

        # find cells with least overlap
        # find indices corresponding to each
        best_overlaps = bb.max(1)
        least_overlap = best_overlaps
        least_overlap.sort()

        new_bbox0 = np.zeros((pos0.shape[0], 4))
        sz_diff = pos0.shape[0] - bbox0.shape[0]

        least_idx = []
        for i in range(sz_diff):
            idx = int(np.where(best_overlaps == least_overlap[i])[0][0])
            if idx not in least_idx:
                least_idx.append(idx)
            else:
                idx = int(np.where(best_overlaps == least_overlap[i])[0][1])
                least_idx.append(idx)

        # Pad in bboxes from the past that aren't well represented by current
        # measurements, suggest a detection was missed
        for i in range(pos0.shape[0]):
            if i in least_idx:
                new_bbox0[i, :] = past_bboxes[i, :]

        # fill remaining bboxes that had high overlap from the current measurement
        empty_new_idx = np.sum(new_bbox0 == 0, 1) > 1
        new_bbox0[empty_new_idx,:] = bbox0

        return new_bbox0

    def _vgg16_appearance_model(self, pooling='max'):
        '''
        Loads a VGG16 appearance model pretrained on ImageNet

        Parameters
        ----------
        pooling : string {'max', 'average', None}. pooling for final output.

        Returns
        -------
        model : Keras model object.
        '''
        import keras.backend as K
        from keras.applications import VGG16
        model = VGG16(include_top=False, pooling=pooling)
        return model

    def _appearance_comparison(self, bbox0, bbox1, im0, im1, window_sz, model):
        '''
        Computes appearance characteristics using `model` between objects in
        `bbox0` and `bbox1` in `im0` and `im1` respectively.

        Returns distances as cosine similarity.

        Parameters
        ----------

        bbox0 : ndarray. M x 4 of bounding box dimensions.
            (min_row, min_col, max_row, max_col).
        bbox1 : ndarray. N x 4 of bounding box dimensions.
            (min_row, min_col, max_row, max_col).
        im0, im1 : ndarray.
            images containing `bbox0`, `bbox1` respectively.
        window_sz : integer.
            size of windows for feature extraction.
        model : object with a `.predict(im)` method that extracts appearance
            features.

        Returns
        -------
        cd : float. [0,1]. cosine distance.
        #ed : float. [0, inf). Euclidean distance.
        '''
        from scipy.spatial.distance import cosine as cosine_dist

        # pad channels dimension as "RGB" for 2D intensity images
        if len(im0.shape) < 3:
            im0 = np.stack([im0]*3, -1)
        if len(im1.shape) < 3:
            im1 = np.stack([im1]*3, -1)

        # pad images to ensure bboxes never run off the edge
        im0p = np.pad(im0, ((window_sz, window_sz), (window_sz, window_sz), (0,0)), mode='reflect')
        im1p = np.pad(im1, ((window_sz, window_sz), (window_sz, window_sz), (0,0)), mode='reflect')

        # move bboxes to fit new dimensional indexing of the image
        new_bbox0 = bbox0.copy() + window_sz
        new_bbox1 = bbox1.copy() + window_sz

        new_bbox0 = new_bbox0.astype('int32')
        new_bbox1 = new_bbox1.astype('int32')


        im0_roi = im0p[new_bbox0[0]:new_bbox0[2], new_bbox0[1]:new_bbox0[3], :]
        im1_roi = im1p[new_bbox1[0]:new_bbox1[2], new_bbox1[1]:new_bbox1[3], :]

        import keras.backend as K
        if K.backend() == 'tensorflow':
            # tensorflow: (batch, dim00, dim01, channels)
            im0_classif = np.expand_dims(im0_roi, 0)
            im1_classif = np.expand_dims(im1_roi, 0)
        else:
            # theano: (batch, channels, dim00, dim01)
            im0_classif = np.expand_dims(np.rollaxis(im0_roi, -1), 0)
            im1_classif = np.expand_dims(np.rollaxis(im1_roi, -1), 0)

        f0 = model.predict(im0_classif)
        f1 = model.predict(im1_classif)

        #if self.verbose:
            #print('Appearance features')
            #print(f0, ' | ', f1)
        assert np.isnan(f0).sum() == 0 and np.isnan(f1).sum() == 0, 'appearance feature was NaN'
        assert np.isinf(f0).sum() == 0 and np.isinf(f1).sum() == 0, 'appearance feature was inf'

        cd = cosine_dist(f0, f1)
        ed = np.sqrt( np.sum( (f0 - f1)**2 ) )

        return cd

    def _appearance_dist_matrix(self, model, positions0, positions1, im0, im1, window_sz=15):
        '''
        Generates distance matrices using the appearance model in `model`
        '''
        from sklearn.metrics import pairwise_distances

        if self.app_off:
            return np.zeros((positions0.shape[0], positions1.shape[0]))

        # Ensure centroids are integers
        positions0 = positions0.astype('int32')
        positions1 = positions1.astype('int32')
        # eliminate any rows where objects weren't detected

        bbox0 = np.zeros((positions0.shape[0], 4))
        for i in range(positions0.shape[0]):
            bbox0[i,:] = np.concatenate([positions0[i,:]-window_sz,
                                        positions0[i,:]+window_sz])

        bbox1 = np.zeros((positions1.shape[0], 4))
        for i in range(positions1.shape[0]):
            bbox1[i,:] = np.concatenate([positions1[i,:]-window_sz,
                                        positions1[i,:]+window_sz])

        app_dist = partial(self._appearance_comparison,
                            model=model, im0=im0, im1=im1, window_sz=window_sz)

        cos_dm = pairwise_distances(bbox0, bbox1, app_dist)

        return cos_dm


    def _make_dist_cost(self, positions0, positions1, augment=False):
        '''
        Generates a cost matrix for linking positions in two conditions.
        Cost based on Euclidean distance.
        If matrix is not square (uneven number of elements in positions0, 1),
        augments the matrix to allow for births and deaths prior to Munkres.
        Parameters
        ----------
        positions0 : ndarray. M x 2 of X,Y positions.
        positions1 : ndarray. N x 2 of X,Y positions.
        augment : boolean. Augment matrix to make square with birth/death matrices.
        Returns
        -------
        costs : matrix of linkage costs, either M x N or O x O if augmented, where
                O = max(M, N)
        '''
        from scipy.spatial import distance_matrix
        from scipy.sparse import spdiags

        d = distance_matrix(positions0, positions1)

        # if not square, augment matrix
        no_link_cost = np.max(d) + 1
        if (d.shape[0] != d.shape[1]) and augment:
            cost_lr = d.T
            top_half = np.hstack([d,
                                  spdiags(no_link_cost * np.ones((1, d.shape[0])), 0, d.shape[0], d.shape[0]).toarray()])
            bottom_half = np.hstack([spdiags(no_link_cost * np.ones((1, d.shape[1])), 0, d.shape[1], d.shape[1]).toarray(),
                                     cost_lr])
            costs = np.vstack([top_half, bottom_half])
        else:
            costs = d

        return costs

    def _scale_matrix(self, M, feature_range=(0,1)):
        '''Scales `M` to `feature_range`'''
        X = M.copy()
        # if invariant, return 0's
        if X.max() - X.min() == 0:
            return np.zeros(X.shape)
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        return X_scaled

    def _make_cost_matrix(self, positions0, positions1, bbox0, bbox1,
                            im0, im1, model, max_dist=20, augment=False):
        '''
        Generates a cost matrix for linking positions in two conditions.
        Cost based on Euclidean distance and bounding box overlap.
        If matrix is not square (uneven number of elements in positions0, 1),
        augments the matrix to allow for births and deaths prior to Munkres.

        Parameters
        ----------
        positions0 : ndarray.
            M x 2 of X,Y positions.
        positions1 : ndarray.
            N x 2 of X,Y positions.
        bbox0 : ndarray. M x 4 of bounding box dimensions.
            (min_row, min_col, max_row, max_col).
        bbox1 : ndarray. N x 4 of bounding box dimensions.
            (min_row, min_col, max_row, max_col).
        augment : boolean.
            Augment matrix to make square with birth/death matrices.

        im0, im1 : ndarray.
            images for `position0`, `position1`.
        max_dist : float.
            maximum linking distance in space.

        Returns
        -------
        costs : matrix of linkage costs, either M x N or O x O if augmented, where
                O = max(M, N)
        '''
        from scipy.spatial import distance_matrix
        from scipy.sparse import spdiags
        from sklearn.metrics import pairwise_distances

        # compute spacial distances
        dm = distance_matrix(positions0, positions1)
        non_links_dist = dm > max_dist

        # compute bounding box overlap distances
        if self.verbose:
            print('POSITIONS')
            print(positions0, '|', positions1)
            print('BBOXES')
            print(bbox0, '|', bbox1)
        bb = pairwise_distances(bbox0, bbox1, metric=self._bbox_overlap)
        non_links_bb = (bb == self.epsilon)

        # computer appearance model distance
        a_cm = self._appearance_dist_matrix(model, positions0, positions1,
                                            im0, im1, window_sz=self.window_sz)

        # compute cost of linking based on motion model
        # todo


        # normalize distance matrices for joining by scaling [0,1] with std var
        dms = self._scale_matrix(dm)
        # invert bb to get union / intersecton
        bbs = self._scale_matrix(bb**-1)
        a_cms = self._scale_matrix(a_cm)

        # specify weighting parameters
        a0, b0 = self.cost_weights[0,:] # distance coefficient, exp
        a1, b1 = self.cost_weights[1,:] # bounding box overlap coefficient, exp
        a2, b2 = self.cost_weights[2,:] # appearance, exp

        d = a0*dms**b0 + a1*bbs**b1 + a2*a_cms**b2

        if self.save_costs:
            self.dms.append(dms)
            self.bbs.append(bbs)
            self.acms.append(a_cms)
        # if not square, augment matrix
        no_link_cost = np.max(d) + 1
        if (d.shape[0] != d.shape[1]) and augment:
            cost_lr = d.T
            top_half = np.hstack([d,
                                  spdiags(no_link_cost * np.ones((1, d.shape[0])), 0, d.shape[0], d.shape[0]).toarray()])
            bottom_half = np.hstack([spdiags(no_link_cost * np.ones((1, d.shape[1])), 0, d.shape[1], d.shape[1]).toarray(),
                                     cost_lr])
            costs = np.vstack([top_half, bottom_half])
        else:
            costs = d

        # add nonlinks
        costs[non_links_dist] += 1000000
        #costs[non_links_bb] += 1000000
        if self.verbose:
            print('CM shapes', dms.shape, bbs.shape, a_cms.shape)
            # join costs from models into global cost matix
            print('DMS : ', dms)
            print('BBS : ', bbs)
            print('ACM : ', a_cms)
            print('COSTS : ', costs)

        return costs

    def _detect_merge_split(self, positions0, positions1, bbox0, bbox1,
                            last_merge=False, last_merge_idx=[]):
        '''
        Determines whether a merge/split event has occured based on the size of
        bounding boxes from neighboring frames

        Parameters
        ----------
        positions0 : ndarray.
            N x 2 of X,Y positions from frame at t=i.
        positions1 : ndarray.
            N x 2 of X,Y positions from frame at t=i+1.
        bbox0 : ndarray.
            M x 4 of bounding box dimensions.
            (row_min, col_min, row_max, col_max). t=i.
        bbox1 : ndarray.
            N x 4 of bounding box dimensions.
            (row_min, col_min, row_max, col_max). t=i+1.
        last_merge : boolean.
            if the last frame was merged.
        last_merge_idx : ndarray.
            indices of the last elements that were duplicated for a merge.

        Returns
        -------
        idx : ndarray.
            integer indexes of merging or splitting object.
        merge_split : {0, 1, 2}.
            0 : None, 1 : merge, 2 : split.

        Notes
        -----

        * Checks the sizes of bbox0, bbox1 to determine if an object has appeared
        or disappeared. If neither has occured, a merge/split event is assumed to
        be absent.
        For many objects this may not be true (merge & split in same frame).
        * If a disappearance has occured, the merging object is determined
        based on a size increase above the previous max. If no size increase is
        detected, no idx is assumed.
        * If an appearance has occured, splitting object is determine based on a
        size decrease below a threshold. If no size decrease is detected,
        no splitting object idx is assumed.
        * If a merge occured in the last frame, we want to keep padding the t+1
        position to allow both objects to share the centroid.
        Checks `last_merge` to see if the last frame was a merge, and

        '''
        from scipy.spatial import distance_matrix

        merge_split = 0
        idx = np.array([])

        # check if object number has changed based on bounding boxes
        disappearance = bbox0.shape[0] > bbox1.shape[0]
        appearance = bbox0.shape[0] < bbox1.shape[0]
        if not (disappearance or appearance):
            return np.array([]), merge_split

        # Calculate bounding box sizes
        a0 = (bbox0[:, 2] - bbox0[:,0]) * (bbox0[:, 3] - bbox0[:,1])
        a1 = (bbox1[:, 2] - bbox1[:,0]) * (bbox1[:, 3] - bbox1[:,1])


        if disappearance:
            if a1.max() > 1.2*a0.max():
                idx = a1 > a0.max()*1.2
                merge_split = 1
        elif appearance:
            if a1.min() < 0.80*a0.min():
                idx = a1 < 0.80*a0.min()
                merge_split = 2

        if last_merge and not (disappearance or appearance):
            last_merge_pos = positions0[last_merge_idx,:]
            dm = distance_matrix(last_merge_pos, positions1)

            new_idx = np.argmin(dm, axis = 1)
            idx = np.zeros(positions1.shape[0])
            idx[new_idx] = 1
            idx = idx.astype('bool')
            merge_split = 1

        return idx, merge_split


    def _link_frame2frame(self, positions0, positions1, bbox0, bbox1,
                            im0, im1, model, max_dist=20):
        '''
        Links objects between frames.
        Utilizes the Munkres (Hunagrian) Algorithm to perform optimal
        global nearest neighbor assignment.

        Parameters
        ----------
        positions0 : ndarray.
            N x 2 of X,Y positions from frame at t=i.
        positions1 : ndarray.
            N x 2 of X,Y positions from frame at t=i+1.
        bbox0 : ndarray.
            M x 4 of bounding box dimensions. (x, y, height, width). t=i.
        bbox1 : ndarray.
            N x 4 of bounding box dimensions. (x, y, height, width). t=i+1.
        im0, im1 : ndarray.
            images for `position0`, `position1`.
        max_dist : float.
            maximum distance to allow frame :: frame linking [in pixels].

        Returns
        -------
        pairs : ndarray.
            N x 4 array of linked X,Y frame positions.
            where each row represents a pair, in the format:
                [position0X, position0Y, position1X, position1Y]
        true_links : ndarray.
            N x 1 boolean array of valid linkages in pairs (rowwise).
        l1t0 : ndarray.
            boolean array of indices for positions in t_i+1 that linked to a
            position in t0
        '''
        from scipy.optimize import linear_sum_assignment

        # Return empty arrays if one set of positions are missed detections
        if not (np.any(positions0) and np.any(positions1)):
            return np.array([]), np.array([]), np.array([]), np.array([])

        costs = self._make_cost_matrix(positions0, positions1,
                                        bbox0, bbox1, im0, im1, model, self.max_dist)
        links = linear_sum_assignment(costs)
        pairs = np.hstack([positions0[links[0], :],
                           positions1[links[1], :]])
        distances = np.sqrt((pairs[:, 0] - pairs[:, 2])**2 + (pairs[:, 1] - pairs[:, 3])**2)
        if self.verbose:
            print('True Link Distances:')
            print(distances)
        true_links = distances < max_dist

        # which indices don't have links to the other time point?
        d0t1 = np.setdiff1d(np.arange(positions0.shape[0]), links[0])
        d1t0 = np.setdiff1d(np.arange(positions1.shape[0]), links[1])

        l1t0 = np.ones(positions1.shape[0]).astype('bool')
        l1t0[d1t0] = 0
        if self.verbose:
            print(d0t1)
            print(d1t0)
            print(l1t0)

        return pairs, true_links, l1t0, links

    def _gap_filling(self, pos0, t):
        '''Find gaps in pos0 marked by -1'''
        gaps_filled = [] # initialze gap filling tracker
        for i in range(pos0.shape[0]):
            # if track didn't link in previous frame and stayed at `-1`
            # it may have missed detection
            if pos0[i,0] < 0:
                # find indices where tracksX has real non-negative values i.e. last detection
                idx = np.where(self.tracksX[i,:] > 0)
                # if the detection is less than `gap_frames`, allow linking to last known location
                if abs(t - idx[0].max()) < gap_frames:
                    pos0[i,:] = [self.tracksX[i,idx[0].max()], self.tracksY[i,idx[0].max()]]
                    gaps_filled += [i] # keep track of which indices we're gap filling
        return pos0, gaps_filled

    def link_frames(self, seq_centroids, seq_bboxes, vid_feeder):
        '''
        Link all frames in a segmented image sequence.

        Parameters
        ----------
        seq_centroids : list of ndarrays.
            list of centroid arrays, ordered temporally.
        gap_frames : integer.
            number of frames to allow tracking gaps.
        vid_feeder : VidFeeder object.
            feeds video of source frames cropped appropriately.

        Returns
        -------
        tracksX : ndarray.
            N x T array of X coordinates.
        tracksY : ndarray.
            N x T array of Y coordinates.
        '''

        # initial objects count
        N = seq_centroids[0].shape[0]
        T = len(seq_centroids)

        if N == 0: # don't track w/o t0 detections
            return np.zeros((0, T)), np.zeros((0, T))

        # Generate tracks arrays
        self.tracksX = np.ones([N, T]) * -1
        self.tracksY = np.ones([N, T]) * -1

        # intialize starting positions
        self.tracksX[:,0] = seq_centroids[0][:,0]
        self.tracksY[:,0] = seq_centroids[0][:,1]
        self.gaps = []
        bbox0 = seq_bboxes[0]

        # init Kalman filter variables
        x_kinit = np.zeros((N, 4)) # first level := time, second level arrays [N, 4]
        for i in range(N):
            x_kinit[i,:] = [self.tracksX[i,0], self.tracksY[i,0], 0, 0]
        self.x_k = [x_kinit]
        self.P = [np.stack([self.P]*N)]

        # load first image and appearance model
        im0 = vid_feeder.next()
        if not self.app_off:
            model = self._vgg16_appearance_model(pooling='max')
        else:
            model = None

        # Loop through time points, connecting frames
        for t in range(T-1):
            if t % self.verbosity == 0:
                print('Linking T%d to T%d' % (t, t+1))

            # Load `pos0` from self.tracksX, self.tracksY
            pos0raw = np.hstack([self.tracksX[:,t].reshape(self.tracksX.shape[0], 1),
                        self.tracksY[:,t].reshape(self.tracksY.shape[0], 1)])
            # Find gaps in frame sequences denoted by `-1` in `pos0`
            # pos0, gaps_filled = self._gap_filling(pos0raw)
            # self.gaps.append(gaps_filled, t) # store which gaps were filled

            # Guess positions based on last Kalman state
            x_k1_guesses, gaps_filled = self._kalman_guess(pos0raw, t)
            pos0 = x_k1_guesses

            # See if any projections exceed `gap_frames`
            # if so, force linkage to nearest neighbors on a cost basis
            # if `object_permanence` is assumed
            if self.object_permanence:
                if (gaps_filled[:,1] > self.gap_frames).sum() > 0 and pos0.shape[0] >= gaps_filled.shape[0]:
                    working_max_dist = 10e6
                else:
                    working_max_dist = self.max_dist
            # otherwise, set any positions exceeding gap frames to an
            # unlinkable location
            else:
                working_max_dist = self.max_dist
                unlinkable = gaps_filled[:,1] > self.gap_frames
                pos0[unlinkable,:2] = -1*working_max_dist



            # if pos0 was padded from the past and bbox0 needs to be, pad it from the past
            # if pos0 missed linkage previously, but was detected properly in the frame
            # padding may not be needed, so check `bbox0` size.
            if pos0.shape[0] > bbox0.shape[0]:
                bbox0 = self._bbox_pad_simple(bbox0, pos0)
                if self.verbose:
                    print('Padded bbox0')
                    print('*---*')
                    print(bbox0)
                    print('*---*')

            assert bbox0.shape[0] == pos0.shape[0], 't positions and bboxes are not the same size'
            # Load the next set of positions, bboxes, im from `t+1`
            pos1 = seq_centroids[t+1]
            bbox1 = seq_bboxes[t+1]
            assert bbox1.shape[0] == pos1.shape[0], 't+1 positions and bboxes are not the same size'
            im1 = vid_feeder.next()

            # if pos1 is smaller than pos0, pad it with -10000 to maintain dimensionality
            # if pos1.shape[0] < pos0.shape[0]:
            #     d = pos1.shape[0] - pos0.shape[0]
            #     pad = np.ones([d, 2])*-10000
            #     pos1 = np.vstack([pos1, pad])

            pairs, true_links, l1t0, links = self._link_frame2frame(pos0[:,:2], pos1, bbox0, bbox1,
                                                im0, im1, model, max_dist=working_max_dist)


            # if pos1[i,:] == -1, it didn't link
            ordered_pos1 = np.zeros((pos0.shape[0], 2)) - 1
            for i in range(pairs.shape[0]):
                if true_links[i] == True:
                    # find which indices in tracksX, Y linked
                    coors = pairs[i,:2]
                    bool_idx = ((pos0[:,:2] - coors == 0).sum(1) == pos0[:,:2].shape[1])
                    idx = np.where(bool_idx)

                    ordered_pos1[idx,0] = pairs[i, 2]
                    ordered_pos1[idx,1] = pairs[i, 3]
                    # self.tracksX[idx,t+1] = pairs[i, 2]
                    # self.tracksY[idx,t+1] = pairs[i, 3]

            # Apply Kalman filter smoothing to the linked tracks

            x_k1_states, P_k1_states = self._kalman_smooth(ordered_pos1, t, gaps_filled)
            self.x_k.append(x_k1_states)
            self.P.append(P_k1_states)

            self.tracksX[:,t+1] = x_k1_states[:,0]
            self.tracksY[:,t+1] = x_k1_states[:,1]

            # Save the last frame for next loop
            im0 = im1
            # don't push unlinked BBox onto the next frame linkage

            if self.verbose:
                print(gaps_filled)
            self.gaps.append(gaps_filled)
            bbox_past = bbox0
            bbox_idx = l1t0
            if np.any(bbox_idx):
                bbox0 = bbox1[bbox_idx,:]
            else:
                bbox0 = np.array([])

        self.tracksX[self.tracksX < 0] = None
        self.tracksY[self.tracksY < 0] = None

        return self.tracksX, self.tracksY

    def _interp_path(self, path):
        '''
        Fills gaps in a tracking sequence by interpolation.

        Parameters
        ----------
        path : ndarray.
            1D array length T, undetected timepoints marked with nan.

        Returns
        -------
        ip : ndarray.
            1D array length T, with relevant points interpolated.
        '''
        ip = path
        path_x = range(np.where(~np.isnan(path))[0].max())
        path_xp = np.where(~np.isnan(path))[0]
        path_fp = path[path_xp]

        tmp = scipy.interp(path_x, path_xp, path_fp)

        ip[:tmp.shape[0]] = tmp

        return ip

    def _fix_ends(self, N, path):
        '''Fixs `nan` at path ends by padding
        if less than 'N' frames are missed'''

        start = path[:N+1]
        end = path[-N+1:]

        if np.isnan(start).sum() <= N:
            last_val_idx = np.where(~np.isnan(start))[0]
            if np.any(last_val_idx):
                start[np.isnan(start)] = start[last_val_idx.min()]
        if np.isnan(end).sum() <= N:
            last_val_idx = np.where(~np.isnan(end))[0]
            if np.any(last_val_idx):
                end[np.isnan(end)] = end[last_val_idx.max()]
        return path

    def fill_gaps(self, tracksX, tracksY, N = 5):
        '''
        Fills gaps in track arrays by linear interpolation

        Parameters
        ----------
        tracksX : ndarray.
            N x T array of X coordinates.
        tracksY : ndarray.
            N x T array of Y coordinates.
        N : integer.
            number of missed detections to allow at track ends.

        Undetected positions marked with nan.

        Returns
        -------
        Parameters, with interpolatable points filled.
        '''

        for i in range(tracksX.shape[0]):
            tracksX[i,] = self._interp_path(tracksX[i,])
            tracksY[i,] = self._interp_path(tracksY[i,])
            tracksX[i,] = self._fix_ends(N, tracksX[i,])
            tracksY[i,] = self._fix_ends(N, tracksY[i,])

        return tracksX, tracksY

    def save_tracks(self, out_dir, tracksX, tracksY):
        '''
        Saves tracks to output directory as `tracksX.csv` & `tracksY.csv`
        '''
        np.savetxt(os.path.join(out_dir, 'tracksX.csv'), tracksX, delimiter=',')
        np.savetxt(os.path.join(out_dir, 'tracksY.csv'), tracksY, delimiter=',')
        return

    def make_gaps_array(self):
        '''Generates N x T boolean array `True` where a gap was filled by the
        Kalman filter interpolation'''

        gaps_array = np.zeros((self.tracksX.shape[0], self.tracksX.shape[1]))

        for t in range(len(self.gaps)):
            filled = self.gaps[t][:,1] > 1
            gaps_array[filled,t] += 1

        self.gaps_array = gaps_array
        return gaps_array


class LabeledMovieMaker(object):
    '''Generate movies of labeled binary objects to visualize tracking'''

    def __init__(self):
        pass

    def label_image_sequence(self, img_dir, out_path, x, y, regex='output*.png', min_t=0, max_t=None, verbose=False):
        '''
        Makes a labeled image sequence
        '''
        from skimage.io import imsave
        from PIL import Image, ImageFont, ImageDraw
        xp = x.astype('int32')
        yp = y.astype('int32')
        imgs = glob.glob(os.path.join(img_dir, regex))
        imgs.sort()

        if min_t != 0:
            for i in range(len(imgs)):
                if 't' + str(min_t).zfill(5) in imgs[i]:
                    start_idx = i

            imgs = imgs[start_idx:start_idx+(max_t-min_t)]

        colors = [(255,0,0), (0, 255, 0), (0, 0, 255), (125, 0, 125)]*30
        font = ImageFont.truetype("Ubuntu-B.ttf", 16)

        if not max_t:
            max_t = len(imgs)

        for t in range(max_t-min_t):
            if verbose:
                print('Labeling T%d' % t)
            I = Image.fromarray(imread(imgs[t]))
            draw = ImageDraw.Draw(I)
            for c in range(xp.shape[0]):
                if np.isnan(xp[c,t]):
                    continue
                else:
                    draw.text((yp[c,t], xp[c,t]), str(c).zfill(2), colors[c], font=font)
            draw.text((10,10), str(t).zfill(5), (0,0,0), font=font) # write timestamp
            imsave(os.path.join(out_path, 'tracking_t%05d.png' % t), np.array(I))

        return

class LabeledMovieMakerSplit(object):
    '''Generate movies of labeled binary objects to visualize tracking
    of split imaged sequences'''

    def __init__(self):
        pass

    def label_image_sequence(self, mp, out_path, x, y, min_t=0, max_t=None, verbose=False):
        '''
        Makes a labeled image sequence

        mp : MaskParser object.
        '''
        from skimage.io import imsave
        from PIL import Image, ImageFont, ImageDraw
        xp = x.astype('int32')
        yp = y.astype('int32')
        imgs = mp.img_files

        if min_t != 0:
            for i in range(len(imgs)):
                if 't' + str(min_t).zfill(3) in imgs[i]:
                    start_idx = i

            imgs = imgs[start_idx:start_idx+(max_t-min_t)]

        colors = [(255,0,0), (0, 255, 0), (0, 0, 255), (125, 0, 125)]*30
        font = ImageFont.truetype("Ubuntu-B.ttf", 60)

        if not max_t:
            max_t = len(imgs)

        for t in range(max_t-min_t):
            if verbose:
                print('Labeling T%d' % t)
            I = Image.open(imgs[t])
            if len(I.size) < 3:
                I = I.convert('RGB')
            draw = ImageDraw.Draw(I)
            for c in range(xp.shape[0]):
                if np.isnan(xp[c,t]):
                    continue
                else:
                    draw.text((yp[c,t], xp[c,t]), str(c).zfill(2), colors[c], font=font)
            draw.text((10,10), str(t).zfill(5), (0,0,0), font=font) # write timestamp
            imsave(os.path.join(out_path, 'tracking_t%03d.png' % t), np.array(I))

        return


def canny_edges(I):
    dx = np.array([-1, 0, 1]).reshape(1, 3)
    dy = dx.T

    Gx = ndi.filters.convolve(I, dx)
    Gy = ndi.filters.convolve(I, dy)

    return np.sqrt(Gx**2 + Gy**2)
