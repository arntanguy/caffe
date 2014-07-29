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
parser = argparse.ArgumentParser(description='Merge all the loop-closure datasets from sub-directories')
args = parser.parse_args()


import os
import random

class Merge:
    def __init__(self):
        self.current_label=0
        self.lc_dataset = open("loop_closures_dataset.txt", 'w')
        self.lc_positive = open("loop_closures_positive.txt", 'w')
        self.lc_negative = open("loop_closures_negative.txt", 'w')
        self.loop_closure_dataset = []

        for dir in self.get_directories("."):
            dataset = dir+"/dataset.txt"
            positive = dir+"/loop_closures_positive.txt"
            negative = dir+"/loop_closures_negative.txt"
            self.merge(dataset, positive, negative)
        # Shuffle dataset to mix images from all directories
        random.shuffle(self.loop_closure_dataset)
        self.lc_dataset.write("".join(self.loop_closure_dataset))
    
    def merge(self, dataset, positive, negative):
        offset=0
        for line in open(dataset):
            if not line.startswith("#"):
                l = line.split(" ")
                l[0] = str(int(l[0])+self.current_label)
                offset += 1
                self.loop_closure_dataset.append(" ".join(l))
                #self.lc_dataset.write(" ".join(l))
        for line in open(positive):
            if not line.startswith("#"):
                l = line.split(" ")
                l[0] = str(int(l[0])+self.current_label)
                l[1] = str(int(l[1])+self.current_label)
                self.lc_positive.write(" ".join(l)+"\n")
        for line in open(negative):
            if not line.startswith("#"):
                l = line.split(" ")
                l[0] = str(int(l[0])+self.current_label)
                l[1] = str(int(l[1])+self.current_label)
                self.lc_negative.write(" ".join(l)+"\n")
        self.current_label += offset

    def get_directories(self,slam_dataset):
        """Get all directories present in the SLAM dataset folder"""
    # Retrieve only top-level directories
        dirs=[]
        for dirname in os.walk(slam_dataset).next()[1]:
            dirs.append(dirname)
        return dirs
    
merge=Merge()
