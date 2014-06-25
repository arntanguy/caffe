#!/usr/bin/python2
# -*- coding: utf-8 -*-

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

import argparse
########################################
## Define and parse arguments
########################################
parser = argparse.ArgumentParser(description='Display Images from the loop-closure dataset')
parser.add_argument('--loop_closure_dataset', type=str, default="loop_closures_dataset.txt")
parser.add_argument('--loop_closure_positive', type=str, default="loop_closures_positive.txt")
parser.add_argument('--max-loop-closures', '-m', type=int, 
                    dest='max_loop_closures', default=10, 
                    help='Maximum number of loop closure images to display')
args = parser.parse_args()

import numpy as np
import sys
import matplotlib.image as mpimg
from axes_sequence import *

LC_TIMESTAMP = 0
LC_RGB = 1
LC_DEPTH = 2
LC_GEOMETRIC_DIST = 3
LC_ROTATION = 6

rgb = []
depth = []
translation = []
rotation = []
for line in open(args.loop_closure_dataset, 'r'):
    if not line.startswith("#"):
        line = line.rstrip().split(" ")
        rgb.append( line[LC_RGB] )
        depth.append( line[LC_DEPTH] )
        translation.append(float(line[LC_GEOMETRIC_DIST]))
        rotation.append(float(line[LC_ROTATION]))


pair_indices = []
for line in open(args.loop_closure_positive):
    if not line.startswith("#"):
        line = line.rstrip().split(" ")
        pair_indices.append( (int(line[0]), int(line[1])) )

# Load and display test image
axes = AxesSequence()
for i in range(0, min(args.max_loop_closures, len(rgb))):
    rgb1 = mpimg.imread(rgb[pair_indices[i][0]])
    rgb2 = mpimg.imread(rgb[pair_indices[i][1]])
    depth1 = mpimg.imread(depth[pair_indices[i][0]])
    depth2 = mpimg.imread(depth[pair_indices[i][1]])
    concat_rgb = np.concatenate((rgb1, rgb2), axis=1)
    concat_depth = np.concatenate((depth1, depth2), axis=1)
    #concat_depth = np.fliplr(concat_depth.reshape(-1,3)).reshape(concat_depth.shape)
    #showimage(rgb1, axes, 'Image '+str(i)) 
    showimage(concat_rgb, axes, u'Loop-closure %i (rgb): translation=%.2fm, rotation=%.2f°' % (i, translation[i], rotation[i]))
    showimage(concat_depth, axes, 'Loop-closure '+str(i)+ ' (depth)', ccmap=pylab.gray()) 



axes.show()
