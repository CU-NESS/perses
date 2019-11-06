"""
File: perses/util/MakeVideo.py
Author: Keith Tauscher
Date: 31 Oct 2019

Description: File containing function that makes video from set of images.
"""
import subprocess    

def make_video(video_file_name, frame_prefix, frame_suffix,\
    original_images_per_second, index_format='%d', slowdown_factor=1):
    """
    Makes a video from the given image files. Only works on systems where
    ffmpeg can be used from subprocess.call
    
    video_file_name: output file name (should have mp4 extension)
    frame_prefix: pre_index picture file name
    frame_suffix: post_index picture file name (including extension)
    original_images_per_second: number of original images per second in video
    index_format: format of index can be '%3d' e.g. for 3 digits: default, '%d'
    slowdown_factor: number of frames per second in video divided by number
                     original images per second in video (pretty much the
                     number of times each image is duplicated for a frame)
    """
    video_command_components =\
    [\
        'ffmpeg',\
        '-r {:d}'.format(int(original_images_per_second)),\
        '-i {0!s}{1!s}{2!s}'.format(frame_prefix, index_format, frame_suffix),\
        '-c:v libx264',\
        '-r {:d}'.format(int(original_images_per_second * slowdown_factor)),\
        '-pix_fmt yuv420p',\
        video_file_name\
    ]
    subprocess.call(' '.join(video_command_components).split(' '))

