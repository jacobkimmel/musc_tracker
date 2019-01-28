'''
Generates labeled tracking movies
'''
import os
import os.path as osp
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw, ImageMath
from skimage.io import imsave
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte
import warnings

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

class ExportTrackingMovie(object):

    def __init__(self, imgs,
                    tracksX,
                    tracksY,
                    masks=None,
                    figsize=(8,8),
                    out_file=None,
                    fps=10,
                    min_t=0,
                    max_t=None,
                    suffix=None):
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
        if len(self.x.shape)==1:
            self.x = np.expand_dims(self.x, 0)
        if len(self.y.shape)==1:
            self.y = np.expand_dims(self.y, 0)

        self._set_colors()

        im_file = self.imgs[0]
        self.prefix = osp.splitext(osp.basename(im_file))[0].split('t')[0]
        self.out_dir = osp.split(self.out_file)[0]
        if suffix is None:
            self.suffix = '_movieframe'
        else:
            self.suffix = suffix

    def _scale_brightness(self, I):
        '''Scales brightness'''
        Iscale = ( (I - I.min()) / (I-I.min()).max() ) * 255
        return Iscale


    def _set_colors(self, n_colors: int=10,) -> None:
        self.n_colors = n_colors
        # N, 3 array of RGB colors, [0,255]
        self.colors = (np.array(sns.husl_palette(n_colors, h=.5)) * 255).astype('int')
        # shuffle the colors so adjacent cells aren't similar colors
        self.colors = self.colors[np.random.choice(np.arange(
            self.colors.shape[0]), size=self.colors.shape[0], replace=False).astype('int'), :]
        return

    def _label_frame(self, i: int):
        '''
        Label a single frame of the movie.
        '''
        font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf', 60)

        # Proper 16-bit to 8-bit scaling
        Iuint16 = np.array(Image.open(self.imgs[i]))
        #Iuint8 = img_as_ubyte(Iuint16)
        Iscale = self._scale_brightness(Iuint16)
        Ip = Image.fromarray(Iscale).convert('RGB')

        # Draw labels on PIL image
        draw = ImageDraw.Draw(Ip)
        for c in range(self.x.shape[0]):
            chosen_color = tuple(self.colors[c % self.n_colors, :].tolist())

            if np.isnan(self.x[c,i]):
                continue
            else:
                draw.text((self.y[c,i], self.x[c,i]), str(c).zfill(2), chosen_color, font=font)

        # Add Timestamp
        draw.text((Iuint16.shape[0]-200, Iuint16.shape[1]-200), str(i).zfill(3), (255, 0, 0), font=font)

        # Convert PIL to numpy array
        I = np.array(Ip)

        # Add segmentation outlines
        if self.masks:
            M = np.array(Image.open(self.masks[i])).astype('bool')
        A = I.copy()
        if self.masks:
            A[binary_dilation(find_boundaries(M), disk(1)), :] = I.mean() + I.std()*4

        return A

    def save_frames(self,) -> None:
        print('SAVING FRAMES FOR %s' % self.prefix)

        for t in range(len(self.imgs)):

            I_label = self._label_frame(t)
            bname = self.prefix + self.suffix + '_t' + str(t).zfill(4) + '.png'

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(osp.join(self.out_dir, bname), I_label)

        return

    def call_ffmpeg(self,) -> None:

        cmd = ['ffmpeg', '-y',
               '-r', str(self.fps),
               '-f', 'image2',
               '-i', osp.join(self.out_dir, self.prefix+self.suffix+'_t%04d.png'),
                '-b:v 8000k',
                '-vcodec', 'libx264',
                '-crf', '25',
                '-pix_fmt', 'yuv420p',
                osp.join(self.out_dir, self.prefix + self.suffix + '_movie_ffmpeg.mp4')]
        print('FFMPEG CALL:')
        print(cmd)
        subprocess.run(" ".join(cmd), shell=True, cwd=self.out_dir)
