#!/usr/local/bin/python3
'''
png_to_movie.py

The purpose of this script is to convert multiple pngs to a movie file. 

Original code taken from references:
    - http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    - https://pastebin.com/P5Xj8u8K

## NEED TO INSTALL CV
    - source activate py35
    - conda install cv2
    - conda install -c conda-forge opencv=3.2.0
    - conda clean --packages
## USAGE
python /c/Users/akchew/bin/pythonfiles/modules/MDDescriptors/visualization/png_to_movie.py -e png -o output.mp4 -p Mostlikely_433.15_6_nm_PDO_50_WtPercWater_spce_dmso --sort last_number_of_png -fps 4
=======
    - conda install -c conda-forge opencv=3.2.0
    - conda install -c conda-forge opencv

'''
# 

import cv2
import argparse
import os
import functools
import re
from functools import cmp_to_key
 
#Function to check if string can be cast to int
def isnum (num):
    try:
        int(num)
        return True
    except:
        return False
 
#Numerically sorts filenames
def image_sort (x,y):
    x = int(x.split(".")[0])
    y = int(y.split(".")[0])
    return x-y
 
# Construct the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-e", "--extension", required=False, default='png', help="Extension name. default is 'png'.")
arg_parser.add_argument("-o", "--output", required=False, default='output.mp4', help="Output video file.")
arg_parser.add_argument("-d", "--directory", required=False, default='.', help="Specify image directory.")
arg_parser.add_argument("-fps", "--framerate", required=False, default='10', help="Set the video framerate.")
arg_parser.add_argument("-s", "--sort", required=False, default='numeric', help="Determines the type of file-order sort that will be used.")
arg_parser.add_argument("-t", "--time", required=False, default='none', help="Sets the framerate so that the video length matches the time in seconds.")
arg_parser.add_argument("-v", "--visual", required=False, default='false', help="If 'true' then will display preview window.")
arg_parser.add_argument("-p", "--prefix", required=False, default='Most', help="Prefix you are interested in")
args = vars(arg_parser.parse_args())
 
# Arguments
dir_path = args['directory']
prefix = args['prefix']
ext = args['extension']
output = args['output']
framerate = args['framerate']
sort_type = args['sort']
time = args['time']
visual = args['visual']
 
#Flips 'visual' to a bool
visual = visual == "true"
 
#Sets the framerate to argument, or defaults to 10
if not isnum(framerate):
    framerate = 10
else:
    framerate = int(framerate)
 
#Get the files from directory
images = []
for f in os.listdir(dir_path):
    if f.endswith(ext) and f.startswith(prefix):
        images.append(f)

    
#Sort the files found in the directory
if sort_type == "numeric":
    int_name = images[0].split(".")[0]
    if isnum(int_name):
        images = sorted(images, key=cmp_to_key(image_sort))
    else:
        print("Failed to sort numerically, switching to alphabetic sort")
        images.sort()
elif sort_type == "alphabetic":
    images.sort()
elif sort_type == "last_number_of_png":
    ## FINDING ALL NUMBERS BEFORE PNG
    int_name_list = [ os.path.splitext(my_str)[0] for my_str in images]
    ## FINDING LAST NUMBER (INT)
    last_number_list = [ int(re.match('.*?([0-9]+)$', int_name).group(1)) for int_name in int_name_list]
    ## GETTING SORTING INDEX
    sorting_index = sorted(range(len(last_number_list)), key=lambda k: last_number_list[k])
    ## LOOPING THROUGH SORTING INDEX AND RECREATING IMAGES
    image_copy = images[:]
    ## RECREATING
    images = [image_copy[each_index] for each_index in sorting_index]
    
#Change framerate to fit the time in seconds if a time has been specified.
#Overrides the -fps arg
if isnum(time):
    framerate = int(len(images) / int(time))
    print("Adjusting framerate to " + str(framerate))
 
# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
 
if visual:
    cv2.imshow('video',frame)
regular_size = os.path.getsize(image_path)
height, width, channels = frame.shape
print("Height:", height)
print("Width:", width)
 
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Be sure to use lower case mp4v fmp4
# Downloading: https://github.com/cisco/openh264/releases
# Decompressing: bzip2 -d openh264-1.7.0-win64.dll.bz2 <-- allows for h264 encoding
# Different codecs: http://www.fourcc.org/codecs.php
out = cv2.VideoWriter(output, fourcc, framerate, (width, height)) # fourcc
 
for n, image in enumerate(images):
    image_path = os.path.join(dir_path, image)
    image_size = os.path.getsize(image_path)
    if image_size < regular_size / 1.5:
        print("Cancelled: " + image)
        continue
 
    frame = cv2.imread(image_path)
    out.write(frame) # Write out frame to video
    # print(out)
    if visual:
        cv2.imshow('video', frame)
 
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
    if n%1== 0: # n%100
        print("Working on frame " + str(n) + ', ' + image_path)
 
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()
 
print("The output video is {}".format(output))