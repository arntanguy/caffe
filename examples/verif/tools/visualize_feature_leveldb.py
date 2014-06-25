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

import argparse
########################################
## Define and parse arguments
########################################
parser = argparse.ArgumentParser(description='Display feature distances from a leveldb database of keys => features')
parser.add_argument(dest='database', type=str, 
                    help='Specify the leveldb database')
parser.add_argument('--sequence-duration', '-d', type=float, 
                    dest='duration', default=1., 
                    help='Duration of the rgbd sequence (in seconds)')
parser.add_argument('--max-loop-closures', '-m', type=int, 
                    dest='max_loop_closures', default=10, 
                    help='Maximum number of loop closure images to display')
args = parser.parse_args()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
from StringIO import StringIO

import leveldb

import matplotlib.image as mpimg
from axes_sequence import *
from collections import OrderedDict


db = leveldb.LevelDB(args.database);

dic = OrderedDict()
#for key, value in db.RangeIter():
for key, value in db.RangeIter('00000001','00000050'):
    dic[key] = np.genfromtxt(StringIO(value), delimiter=" ")

for key, value in dic.items():
    print value
#x = []
#y = []
#t = []
#slam_dataset = ''
#for line in open(args.feature_distances, 'r'):
#    li=line.strip()
#    if not li.startswith("#"):
#        line = line.rstrip().split(" ")
#        x.append(float(line[0]))
#        y.append(float(line[1]))
#        t.append(line[2])
#    else:
#        line = line.rstrip().split(" ")
#        if line[0] == '#DATASET':
#            slam_dataset = line[1]
#            print 'Using slam dataset %s' % slam_dataset
#
#mu = 1./len(t)
#time =  [i*mu for i in range(0, len(t))]
#colours = [i for i in y]
## colours = time
#
#
#########################################
### Display n most likely loop closures #
#########################################
#
## Get the best n potential loop closures
#N=min(len(y), args.max_loop_closures)
#array = np.array(y)
#indices = array.argsort()[:N]
#loop_closures = [t[e] for e in indices]
## print "[%s]" % ', '.join(map(str, loop_closures ))
#
#
## Load and display test image
#axes = AxesSequence()
#for i in range(0, len(loop_closures)):
#    example_image = mpimg.imread(slam_dataset+'/rgb/'+loop_closures[i]+'.png')
#    example_image_rgb = np.fliplr(example_image.reshape(-1,3)).reshape(example_image.shape)
#    showimage(example_image_rgb, axes, 'Distance for '+str(i+1)+'th closest: '+str(y[indices[i]])+'\ntime: '+str(args.duration*time[indices[i]]))
#
#
#
#########################################
### Display feature distance plot
#########################################
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#p = ax.scatter(x, [i*args.duration for i in time], y, c=colours)
#fig.colorbar(p)
##ax.plot(x, y)
#ax.set_xlabel('Geometric Distance')
#ax.set_zlabel('Feature Distance (norm2)')
#ax.set_ylabel('Frame (time)')
#axes.show()
