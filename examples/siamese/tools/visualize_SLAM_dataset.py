#!/usr/bin/python2
# Software License Agreement (BSD License)
#
# Copyright (c) 2014, Arnaud TANGUY, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import random
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
#%matplotlib inline

import caffe
import caffe.imagenet

import os

# Make sure that caffe is on the python path:
caffe_root = '../caffe-dev/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'distribute/python')

# directories: rgb, depth
slam_dataset = '/media/DATA/Datasets/SLAM/rgbd_dataset_freiburg1_room'

class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8], 
                               visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        self.axes[self._i].set_visible(False)
        if self._i + 1 < len(self.axes):
            self.axes[self._i+1].set_visible(True)
            self._i += 1
        else:
            self._i = 0
            self.axes[self._i].set_visible(True)



    def prev_plot(self):
        self.axes[self._i].set_visible(False)
        if self._i - 1 > 0:
            self.axes[self._i-1].set_visible(True)
            self._i -= 1
        else:
            self._i = len(self.axes)-1
            self.axes[self._i].set_visible(True)

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()

# our network takes BGR images, so we need to switch color channels
def showimage(im, axes, title=''):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    ax = axes.new()
    ax.imshow(im)
    ax.set_title(title)
    
# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, axes, title='', padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    showimage(data, axes, title)

def descriptor(desc_flat):
    desc = desc_flat[0:len(desc_flat)]
    return desc

def dist_descriptor(desc1, desc2):
    return ((desc1-desc2)**2).sum()
    #return np.absolute(LA.norm(desc1) - LA.norm(desc2))

axes = AxesSequence()

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.imagenet.ImageNetClassifier(caffe_root + 'examples/imagenet/imagenet_deploy.prototxt',
                                        caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')
net.caffenet.set_phase_test()
net.caffenet.set_mode_cpu()
#net.caffenet.set_mode_gpu()

print 'Blob items'
print [(k, v.data.shape) for k, v in net.caffenet.blobs.items()]
print 'Param items'
print [(k, v[0].data.shape) for k, v in net.caffenet.params.items()]
    

# Get image path from SLAM dataset
rgb_dataset = []
for line in open(slam_dataset + '/rgb.txt', 'r'):
    li=line.strip()
    if not li.startswith("#"):
        rgb_dataset.append(line.rstrip().split(" ")[1])

descriptors = []
for img in rgb_dataset[0:5]:
#for img in [rgb_dataset[0],rgb_dataset[0]]: 
#for img in random.sample(rgb_dataset, 10):
    rgb_img = slam_dataset + '/' + img 
    print 'Displaying ' + rgb_img
    # Run a classification pass (computes the network)
    ### XXX: why does the net have to be recreated everytime!!!! Very slow!!!!
    net.predict(rgb_img)
    
    # Load and display test image
    example_image = mpimg.imread(rgb_img)
    example_image_rgb = np.fliplr(example_image.reshape(-1,3)).reshape(example_image.shape)
    showimage(example_image_rgb, axes, 'Original Image: '+img)
    
    # index four is the center crop
    image = net.caffenet.blobs['data'].data[4].copy()
    image -= image.min()
    image /= image.max()
    showimage(image.transpose(1, 2, 0), axes)
    
    ##
    # Fist layer output
    ##
    # the parameters are a list of [weights, biases]
    filters = net.caffenet.params['conv1'][0].data.copy()
    vis_square(filters.transpose(0, 2, 3, 1), axes, 'Conv1')
    
    # The first layer output, conv1 (rectified responses of the filters above, first 36 only)
    feat = net.caffenet.blobs['conv1'].data[4, :36]
    vis_square(feat, axes, 'conv1 - rectified', padval=1)
    #
    ###
    ##Second layer output
    ###
    filters = net.caffenet.params['conv2'][0].data.copy()
    vis_square(filters[:48].reshape(48**2, 5, 5), axes, 'conv2')
    #
    ## The second layer output, conv2 (rectified, only the first 36 of 256 channels)
    feat = net.caffenet.blobs['conv2'].data[4, :36]
    vis_square(feat, axes, 'conv2 - rectified (first 36 only)', padval=1)
    
    ### The third layer output, conv3 (rectified, all 384 channels)
    feat = net.caffenet.blobs['conv3'].data[4]
    vis_square(feat, axes, 'conv3 - rectified (all 384 channels)', padval=0.5)
 
    ### The third layer output, conv3 (rectified, all 384 channels)
    feat = net.caffenet.blobs['conv3'].data[4, :36]
    vis_square(feat, axes, 'conv3 - rectified (first 36 only)', padval=0.5)
    #
    #
    ### The fourth layer output, conv4 (rectified, all 384 channels)
    feat = net.caffenet.blobs['conv4'].data[4]
    vis_square(feat, axes, 'conv4 - rectified (all 384 channels)', padval=0.5)
    
    feat = net.caffenet.blobs['conv4'].data[4, :36]
    vis_square(feat, axes, 'conv4 - rectified (first 36 only)', padval=0.5)
 
    ### The fifth layer output, conv5 (rectified, all 256 channels)
    feat = net.caffenet.blobs['conv5'].data[4]
    vis_square(feat, axes, 'conv5 - rectified (all 256 channels)', padval=0.5)
 
    feat = net.caffenet.blobs['conv5'].data[4, :36]
    vis_square(feat, axes, 'conv5 - rectified (first 36 only)', padval=0.5)
    #
    #
    ## The fifth layer after pooling, pool5
    feat = net.caffenet.blobs['pool5'].data[4]
    vis_square(feat, axes, '5th layer, after pooling', padval=0.5)
    
    feat = net.caffenet.blobs['fc6'].data[4]
    ax = axes.new()
    ax.plot(feat.flat)
    ax.set_title('fc6: First fully connected layer')
    
    feat = net.caffenet.blobs['fc7'].data[4]
    ax = axes.new()
    ax.plot(feat.flat)
    ax.set_title('fc7: Second fully connected layer')
    
    feat = net.caffenet.blobs['fc8'].data[4]
    descriptors.append(descriptor(feat.flat))
    ax = axes.new()
    ax.plot(feat.flat)
    ax.set_title('fc8: Third fully connected layer')
    
    feat = net.caffenet.blobs['prob'].data[4]
    ax = axes.new()
    ax.plot(feat.flat)
    ax.set_title("prob: Final Probability output (softmax classification)")
    
    
    
    # CLASSIFICATION
    imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        import os
        os.system('../data/ilsvrc12/get_ilsvrc_aux.sh')
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    
    top_k = net.caffenet.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
    print labels[top_k]
    
for i in range(0, len(descriptors)):
    print 'Distance 0-' + str(i) + ': ' + dist_descriptor(descriptors[0], descriptors[i]).astype('str')

axes.show()
