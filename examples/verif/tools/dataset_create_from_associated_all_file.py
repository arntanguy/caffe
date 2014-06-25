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
parser = argparse.ArgumentParser(description='Generate a dataset file from associated_all file.')
parser.add_argument('associated_all', nargs="?",
                    type=str, default='associated_all.txt')
parser.add_argument('dataset', nargs="?",
                    type=str, default='dataset.txt',
                    help="Path where the dataset will be written")
args = parser.parse_args()

import os


def convert_associated_all_to_dataset(associated_all_path, dataset_path):
    label=0
    save_file = open(dataset_path, 'w')

    for line in open(associated_all_path):
        li = line.strip()
        if not li.startswith('#'):
                line = line.rstrip().split(" ")
# Map timestamp label to int id
                key = str(label)
                label += 1
                position = " ".join(line[1:4])
                rotation = " ".join(line[4:8])
                dataset_dir_path=os.path.abspath(".")
                rgb_file = dataset_dir_path+"/"+line[9]
                depth_file = dataset_dir_path+"/"+line[11]
                save_file.write("%s %s %s %s %s\n" % (key, rgb_file, depth_file, position, rotation))
            

print "Converting associated_all file %s to dataset file %s" % (args.associated_all, args.dataset)
convert_associated_all_to_dataset(args.associated_all, args.dataset)
