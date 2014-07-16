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
parser = argparse.ArgumentParser(description="""Generates a shuffle list containing a mixture of pairs of valid and invalid loop closures.

Run it without arguments, and it will use the following files in the current directory:
- loop_closures_positive.txt
- loop_closures_negative.txt
It will create the following file
- loop_closures_shuffle_list.txt"
""")
parser.add_argument('positive', type=str, 
                    help='Path to the positive loop-closures file. It should contains pairs of image ids for positive loop-closes',
                    nargs="?", default="loop_closures_positive.txt")
parser.add_argument('negative', type=str, 
                    help='Path to the positive loop-closures file. It should contains pairs of image ids for positive loop-closes',
                    nargs="?", default="loop_closures_negative.txt")
parser.add_argument('shuffle_list', type=str, 
                    help='Path to the suffle list file that will be written',
                    nargs="?", default="loop_closures_shuffle_list.txt")
args = parser.parse_args()


import random

def read_loop_closure_files(path, max_iter, positive):
    lc = []
    i=0
    for line in open(path):
        if i==max_iter:
            return lc
        li = line.strip()
        if not li.startswith('#'):
            line = line.rstrip().split(" ")
            lc.append( [int(line[0]), int(line[1]), positive] )
        i = i+1
    return lc

def save_shuffle_list(shuffle_list, save_file_path):
    save_file = open(save_file_path, 'w')
    save_file.write("%i\n" % len(shuffle_list))
    for l in shuffle_list:
        save_file.write("%i %i %i\n" % (l[0], l[1], l[2]))
    save_file.close()






print "Reading positive loop-closures"
positive_lc = read_loop_closure_files(args.positive, 10000000, 1)
print "Reading negative loop-closures"
negative_lc = read_loop_closure_files(args.negative, len(positive_lc), 0)


# Maximum number of loop-closures
print "Shuffle sub-sampling"
m = min(len(positive_lc), len(negative_lc))

print "Generating shuffle list of %i pairs: %i positives, %i negatives" % (2*m, m, m)
result = positive_lc + negative_lc
print "Shuffling shuffle_list"
random.shuffle(result)


save_shuffle_list(result, args.shuffle_list)

