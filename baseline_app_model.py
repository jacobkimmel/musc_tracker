'''Implements a baseline appearance model using simple image heuristics'''

import numpy as np
import glob
import os

class HeuristicAppearance(object):
    '''Heuristic appearance model

    Attributes
    ----------
    metric : string
        distance metric used. {"euclidean"}
    transform : callable
        transform function
    '''

    def __init__(self, metric='euclidean'):

        self.metric = metric

        if transform is None:
            self.transform=np.flatten

    def euclidean(self, im0, im1):
        '''Euclidean distance between two images.

        Parameters
        ----------
        im0, im1 : np.ndarray
            images to compare.

        Returns
        -------
        d : float
            sum of squared
        '''
        d = np.sum( (im1 - im0)**2 )
        return d

    def predict(self, im):
        '''Returns a flattened image'''
        return self.transform(im)
