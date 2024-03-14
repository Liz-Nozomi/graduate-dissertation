# -*- coding: utf-8 -*-
"""
delete_files.py
This script contains functions that are designed to remove files.

Written by: Alex K. Chew (02/26/2020)

FUNCTIONS:
    remove_all_hashtag_files:
        removes all hashtag files, which is useful if you are re-running 
        the scripts multiple time.

"""
import os, glob

### FUNCTION TO REMOVE ALL EXTRAS # 
def remove_all_hashtag_files(wd,
                             prefix = r"#",
                             verbose = True):
    '''
    The purpose of this function is to remove all extraneous # files. 
    This is useful when you are trying to preserve data and do not want to 
    create extremely large directories.
    INPUTS:
        wd: [str]
            path to working directory
        prefix: [str]
            prefix to hashtag
        verbose: [logical]
            True if you want to print out all the details
    OUTPUTS:
        void -- this just removes all hashtags from the files
    '''
    ## LOOPING THROUGH AND REMOVING
    for idx, filename in enumerate(glob.glob( wd + '/' + prefix + "*")):
        if verbose is True:
            if idx == 0:
                print("Current working directory: %s"%(wd) )
            print("--> Removing file: %s"%( os.path.basename( filename ) ) )
        ## REMOVING FILES ONLY
        if os.path.isfile(filename):
            os.remove(filename)
    return