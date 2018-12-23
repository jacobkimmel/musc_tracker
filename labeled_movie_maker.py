'''
Generates labeled tracking movies
'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image, ImageFont, ImageDraw, ImageMath

from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte

class TrackingMovie(object):

    def __init__(self, imgs, tracksX, tracksY,
                    masks=None,
                    figsize=(8,8), out_file=None,
                    fps=10, min_t=0, max_t=None):
        '''
        Generates labeled tracking movies for images in `img_dir` with
        segmentation masks in `seg_dir` and tracks in `tracksX`, `tracksY`.

        Utilizes matplotlib animated figures and the `FFMpegWriter` backend.

        Parameters
        ----------
        imgs : array-like. paths to image files.
        tracksX : ndarray. N x T array of X coordinates.
        tracksY : ndarray. N x T array of Y coordinates.
        masks : array-like, optional.paths to segmentation masks.
        figsize : 2-tuple. (height, width) of output figure.
        out_file : string, optional. path for file output.
        fps : integer, optional. frames per second for output.

        Returns
        -------
        None.
        '''

        self.imgs = imgs
        self.masks = masks

        self.figsize = figsize
        self.out_file = out_file
        self.fps = fps

        self.x = tracksX.astype('int32')
        self.y = tracksY.astype('int32')

        self._generate_fig()

    def _scale_brightness(self, I):
        '''Scales brightness'''
        Iscale = ( (I - I.min()) / (I-I.min()).max() ) * 255
        return Iscale

    def _animate(self, i):
        '''
        Iteration fnx for a `matplotlib.animation.FuncAnimation`
        '''
        colors = [(255,0,0), (0, 255, 0), (0, 0, 255), (125, 0, 125)]*30
        font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf', 60)

        # Proper 16-bit to 8-bit scaling
        Iuint16 = np.array(Image.open(self.imgs[i]))
        #Iuint8 = img_as_ubyte(Iuint16)
        Iscale = self._scale_brightness(Iuint16)
        Ip = Image.fromarray(Iscale).convert('RGB')

        # Draw labels on PIL image
        draw = ImageDraw.Draw(Ip)
        for c in range(self.x.shape[0]):
            if np.isnan(self.x[c,i]):
                continue
            else:
                draw.text((self.y[c,i], self.x[c,i]), str(c).zfill(2), colors[c], font=font)

        # Add Timestamp
        draw.text((Iuint16.shape[0]-200, Iuint16.shape[1]-200), str(i).zfill(3), colors[0], font=font)

        # Convert PIL to numpy array
        I = np.array(Ip)

        # Add segmentation outlines
        if self.masks:
            M = np.array(Image.open(self.masks[i])).astype('bool')
        A = I.copy()
        if self.masks:
            A[binary_dilation(find_boundaries(M), disk(1)), :] = I.mean() + I.std()*4
        self.im.set_array(A)
        return (self.im,)

    def _generate_fig(self):
        '''
        Generates animated figure.
        '''
        fig = plt.figure(figsize = self.figsize)
        ax = fig.add_subplot(111)
        self.im = ax.imshow(np.array(Image.open(self.imgs[0])), cmap='gray')

        anim = animation.FuncAnimation(fig, self._animate, frames=len(self.imgs), )

        self.anim = anim
        return

    def save_fig(self):

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps = self.fps)

        self.anim.save(self.out_file, writer=writer)
        return
