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
parser = argparse.ArgumentParser(description='Generate a dataset of loop-closures from TUM SLAM Dataset.')
parser.add_argument('--slam_dataset', dest='slam_dataset', 
                    type=str, default='/media/DATA/Datasets/SLAM_LOOP', 
                    help='Path to one of the RGB-D SLAM Dataset and Benchmark (http://vision.in.tum.de/data/datasets/rgbd-dataset)')
parser.add_argument('--washington-dataset', dest="washington_dataset",
                    type=str, default='/media/DATA/Datasets/Washington-converted',
                    help='Path to washington\'s RGBD scene dataset (http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/)')
parser.add_argument('--translation', dest='translation', 
                    type=float, default=2.,
                    help='Maximal translation allowed between two images to be considered as a loop-closure (in meters). Default = 2m')
parser.add_argument('--rotation', dest='rotation',
                    type=float, default=30.,
                    help='Maximal roation allowed between two images to be considered as a loop-closure (in degrees). Default = 30')
parser.add_argument('--keyframe-step', dest='keyframe_step',
                    type=int, default=10,
                    help='Number of frames between two keyframes')
parser.add_argument('--keyframe-distance', dest="keyframe_distance",
                    type=int, default=10,
                    help='Minimal number of keyframes separating two images to be considered as a loop-closure. This is used to reduce the number of redundant loop-closures detected')
parser.add_argument('--save-dir', dest='save_dir_path',
                    type=str, default="/tmp",
                    help='Directory where the loop-closure dataset will be saved')
parser.add_argument('--save-positive-loops', dest="save_file_positive_name",
                    type=str, default='loop_closures_positive.txt',
                    help='Name of the destination file for the positive loop closures. Default: loop_closures_positive.txt')
parser.add_argument('--save-negative-loops', dest="save_file_negative_name",
                    type=str, default='loop_closures_negative.txt',
                    help='Name of the destination file for the negative loop closures. Default: loop_closures_negative.txt')
parser.add_argument('--save-dataset_info', dest="save_file_dataset_name",
                    type=str, default='loop_closures_dataset.txt',
                    help='Name of the destination file for the dataset information associated with the loop-closure pairs. Default: loop_closures_dataset.txt')
parser.add_argument('--no-tum', dest='use_tum', default=True, action='store_false', help='Do not run on TUM Dataset')
parser.add_argument('--no-washington', dest='use_washington', default=True, action='store_false', help='Do not run on TUM Dataset')
parser.add_argument('--save-valid-loops-only', dest='save_all_loops', default=True, action='store_false',
        help='Only save valid loop closures. Do not save the remaining pairs')
args = parser.parse_args()


import numpy as np;
import os
from collections import OrderedDict
from itertools import islice

GT_ID=0
GT_TRANSLATION=1
GT_ROTATION=2
GT_RGB=3
GT_DEPTH=4


LC_TIMESTAMP1=0
LC_RGB1=1
LC_DEPTH1=2
LC_TIMESTAMP2=3
LC_RGB2=4
LC_DEPTH2=5
LC_TRANSLATION=6
LC_ROTATION=7

def get_directories(slam_dataset):
    """Get all directories present in the SLAM dataset folder"""
# Retrieve only top-level directories
    dirs=[]
    for dirname in os.walk(slam_dataset).next()[1]:
        dirs.append(dirname)
    return dirs

def get_files(directory):
    """List all files in given directory"""
    print directory
    files = []
    for root, dir, file in os.walk(directory):
        files.append(file)
    return files[0]



            

class LoopClosureDataset:

    def __init__(self, save_dir_path, save_file_positive, save_file_negative, save_file_dataset):
        self.current_label = 0

        #   Init names
        self.save_file_positive_name = save_file_positive
        self.save_file_negative_name = save_file_negative
        self.save_file_dataset_name = save_file_dataset

        self.save_dir_path = save_dir_path
        print "The following files will be created in directory %s:\n\t%s\n\t%s\n\t%s" % (self.save_dir_path, self.save_file_positive_name, self.save_file_negative_name, self.save_file_dataset_name)

        self.save_file_positive_path = self.save_dir_path + "/" + self.save_file_positive_name
        self.save_file_negative_path = self.save_dir_path + "/" + self.save_file_negative_name
        self.save_file_dataset_path  = self.save_dir_path + "/" + self.save_file_dataset_name

        # Init file streams
        self.save_file_positive = open(self.save_file_positive_path, 'w')
        self.save_file_positive.write("#  Positive loop closures: id pairs of loop-closure images")

        self.save_file_negative = open(self.save_file_negative_path, 'w')
        self.save_file_negative.write("# Negative loop-closures id pairs of loop-closure images")

        self.save_file_dataset = open(self.save_file_dataset_path, 'w')
        self.save_file_dataset.write("# id rgb depth tx ty tz q1 q2 q3 q4")

    def read_file_tum(self, slam_directory, keyframe_step):
        """Reads the dataset.txt file, take a value for each keyframe (distance between keyframes is defined by keyframe_step), and return a parsed version of it as a list of lists, where each element of the list is 
        [id, [tx ty tz] , [qx qy qz qw], rgb_file, depth_file]"""
        filedata = []
        frame_count = 0
        for line in open(slam_directory+'/dataset.txt'):
            li = line.strip()
            if not li.startswith('#'):
                if frame_count % keyframe_step == 0:
                    line = line.rstrip().split(" ")
    # Map timestamp label to int id
                    key = str(self.current_label)
                    self.current_label += 1

                    rgb_file = line[1]
                    depth_file = line[2]
                    position = np.array(line[3:6], dtype=np.float)
                    rotation = np.array(line[6:10], dtype=np.float)

                    data = [key, position, rotation, rgb_file, depth_file]
                    filedata.append(data)
                    frame_count=0
                frame_count += 1
        return filedata

    #def read_file_washington(self, dataset_root, directory, keyframe_step):
    #    """Read Washington's university RGBD scene dataset (http://rgbd-dataset.cs.washington.edu/), and return a parsed version of it as a list of lists, where each element of the list is 
    #    [timestamp, [tx ty tz] , [qx qy qz qw]]"""
    #
    #    dir_num = directory[-2:]
    #    files = get_files(dataset_root+"/imgs/"+directory)
    #    files.sort()
    #
    ## RGBd depth file
    #    nb_files = len(files)/2
    ## Read Pose file 
    #    filedata = []    #filedata = []
    #    frame_count = 0
    #    frame_count=0
    #    for line in open(dataset_root+"/pc/"+dir_num+".pose", 'r'):
    #        if(frame_count % keyframe_step == 0):
    #            li = line.rstrip().split(" ")
    #            key = str(self.current_label)
    #            self.current_label += 1
    #            rotation = np.array(li[0:4], dtype=np.float) 
    #            position = np.array(li[4:7], dtype=np.float)
    #            rgb_file = dataset_root+"/imgs/scene_"+dir_num+"/"+files[frame_count*2]
    #            depth_file = dataset_root+"/imgs/scene_"+dir_num+"/"+files[frame_count*2+1]
    #            data = [key, position, rotation, rgb_file, depth_file]
    #            filedata.append(data)
    #        frame_count += 1
    #    return filedata

    def find_loop_closures(self, file, translation, rotation, keyframe_step, keyframe_distance):
        gt = file #read_file_tum(slam_directory, keyframe_step)
        valid_lc = []
        invalid_lc = []
        pairs = []
    # Check if pairs between current keyframe and all others (more than keyframe_distance) apart are a loop-closure 
        for i in range(0, len(gt)):
            current_translation = gt[i][GT_TRANSLATION]
            current_rotation = gt[i][GT_ROTATION]
    # All keyframes but the keyframe_distance closest
            for j in [x for x in range(0, len(gt)) if np.abs(x-i) >= keyframe_distance]:
                other_translation = gt[j][GT_TRANSLATION]
                other_rotation = gt[j][GT_ROTATION]
    # Check whether this is a loop closure
                geom_dist_t = np.linalg.norm(np.subtract(current_translation, other_translation))
                rotation_angle = np.degrees(np.arccos(2*np.power(np.dot(current_rotation, other_rotation),2) -1))
                pair1 = (gt[i][GT_ID], gt[j][GT_ID])
                pair2 = (gt[j][GT_ID], gt[i][GT_ID])
                if not ((pair1 in pairs) or (pair2 in pairs)):
                    pairs.append(pair1)
                    if geom_dist_t < translation and rotation_angle < rotation:
    # timestamp, rgb image, depth image, timestamp, rgb image, depth image, is_loop_closure, geometric distance (translation in meters), rotation angle (degrees)
    # avoid duplicates
                        valid_lc.append(pair1)
                    else:
                        invalid_lc.append(pair1)
        return (valid_lc, invalid_lc)

    def write_loop_closures(self, valid_lc, invalid_lc):
        """Save loop-closures to file in human readable format"""
        for l in valid_lc:
            self.save_file_positive.write(" ".join(l)+"\n")
        for l in invalid_lc:
            self.save_file_negative.write(" ".join(l)+"\n")
    
    def write_dataset(self, f):
        for line in f:
            self.save_file_dataset.write("%s %s %s %s %s\n" % (line[GT_ID], line[GT_RGB], line[GT_DEPTH], " ".join([str(x) for x in line[GT_TRANSLATION].tolist()]), " ".join([str(x) for x in line[GT_ROTATION].tolist()])))
        

    def create_from_tum_dataset(self, slam_dataset, max_translation, max_rotation, keyframe_step, keyframe_distance):
        print "Generating loop-closure dataset from %s" % (slam_dataset)
        print "The following directories have been found:"
        dirs = get_directories(slam_dataset) 
        for name in dirs:
            print "\t%s" % (name)

        #for name in dirs[0:1]:
        for name in dirs:
            print "Processing %s" % (name)
            f = self.read_file_tum(slam_dataset + "/" + name, keyframe_step)
            valid_lc, invalid_lc = self.find_loop_closures(f, max_translation, max_rotation, keyframe_step, keyframe_distance)
            print '\tFound %i loop-closures out of %i image pairs' % (len(valid_lc), len(valid_lc)+len(invalid_lc))
            self.write_loop_closures(valid_lc, invalid_lc)
            self.write_dataset(f)

#    def create_from_washington_dataset(self, washington_dataset_root, max_translation, max_rotation, keyframe_step, keyframe_distance):
#        print "Generating loop-closure dataset from %s" % args.washington_dataset
#        dirs=get_directories(args.washington_dataset+"/imgs")
#        dirs.sort()
#        print "The following directories have been found:"
#        for d in dirs:
#            print "\t%s" % d
#
#        for d in dirs:
#            print "Processing %s" % (d)
#            f = self.read_file_washington(args.washington_dataset, d, args.keyframe_step)
#            valid_lc, invalid_lc = self.find_loop_closures(f, max_translation, max_rotation, keyframe_step, keyframe_distance)
#            print '\tFound %i loop-closures out of %i image pairs' % (len(valid_lc), len(valid_lc)+len(invalid_lc))
#            self.write_loop_closures(valid_lc, invalid_lc)
#            self.write_dataset(f)
#


lc = LoopClosureDataset(args.save_dir_path, args.save_file_positive_name, args.save_file_negative_name, args.save_file_dataset_name)
if args.use_tum:
    lc.create_from_tum_dataset(args.slam_dataset, args.translation, args.rotation, args.keyframe_step, args.keyframe_distance)


if args.use_washington:
    lc.create_from_tum_dataset(args.washington_dataset, args.translation, args.rotation, args.keyframe_step, args.keyframe_distance)


