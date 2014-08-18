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
def range_type(s):
    try:
        x, y = map(int, s.split(','))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError("Range must be min,max")


parser = argparse.ArgumentParser(description='Display Images from the loop-closure dataset')
parser.add_argument('--loop_closure_dataset', type=str, default="loop_closures_dataset.txt")
parser.add_argument('--loop_closure_positive', type=str, default="loop_closures_positive.txt")
parser.add_argument('--range', dest="range", type=range_type, default="0,10")
parser.add_argument('--depth',dest='depth',action='store_true', default=False)
parser.add_argument('--rgb',dest='rgb',action='store_true', default=True)
args = parser.parse_args()

import numpy as np
import sys
import matplotlib.image as mpimg
from axes_sequence import *

LC_TIMESTAMP = 0
LC_RGB = 1
LC_DEPTH = 2
LC_TRANSLATION = 3
LC_ROTATION = 6

dic = {}
for line in open(args.loop_closure_dataset, 'r'):
    if not line.startswith("#"):
        line = line.rstrip().split(" ")
        trans = np.array(line[3:6], dtype=np.float)
        rot = np.array(line[6:10], dtype=np.float)
        print "Key %i" % (int(line[0]))
        dic[int(line[0])] = [line[LC_RGB], line[LC_DEPTH], trans, rot]


pair_indices = []
for line in open(args.loop_closure_positive):
    if not line.startswith("#"):
        line = line.rstrip().split(" ")
        pair_indices.append( (int(line[0]), int(line[1])) )

begin=args.range[0]
end=args.range[1]
if end > len(pair_indices) or begin < 0:
    print "Invalid range : provided range [%i,%i] is not within range [%i,%i]" % (begin, end, 0, len(pair_indices))
    exit(-1)

# Load and display test image
axes = AxesSequence()
i=0
for (id1,id2) in pair_indices[begin:end]:
    data1 = dic[id1]
    data2 = dic[id2]
    trans = np.linalg.norm(np.subtract(data1[2], data2[2]))
    rot = np.degrees(np.arccos(2*np.power(np.dot(data1[3], data2[3]),2) -1))
    if args.rgb:
        print "Img: " + data1[0] + data2[0]
        rgb1 = mpimg.imread(data1[0])
        rgb2 = mpimg.imread(data2[0])
        concat_rgb = np.concatenate((rgb1, rgb2), axis=1)
        showimage(concat_rgb, axes, u'Loop-closure (%i-%i) (rgb): translation=%.2fm, rotation=%.2f°' % (id1, id2, trans, rot))
    if args.depth:
        depth1 = mpimg.imread(data1[1])
        depth2 = mpimg.imread(data2[1])
        concat_depth = np.concatenate((depth1, depth2), axis=1)
        showimage(concat_depth, axes, 'Loop-closure '+str(i)+ ' (depth)', ccmap=pylab.gray()) 
    i += 1



axes.show()
