# -*- coding: utf-8 -*-
"""
track_time.py
The purpose of this script is to keep track of time

CREATED ON: 05/11/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

FUNCTIONS:
    convert_sec_to_hms: converts seconds to hours, minutes, and seconds
    print_total_time: prints total time a process took, given the start time
    
USAGE:
    from MDDescriptors.core.track_time import track_time calc_tools
"""
import time
from datetime import datetime

### FUNCTION TO KEEP TRACK OF TIME
def convert_sec_to_hms( seconds ):
    '''
    This function simply takes the total seconds and converts it to hours, minutes, and seconds
    INPUTS:
        seconds: Total seconds
    OUTPUTS:
        h: hours
        m: minutes
        s: seconds
    '''
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

### FUNCTION TO PRINT TIME GIVEN INITIAL TIME
def print_total_time(start_time, string = 'Total time: '):
    '''
    The purpose of this function is to print the total time given a start time
    INPUTS:
        start_time: time from time module, e.g.
            start_time = time.time()
        string: [str] string in front of the total time
    OUTPUTS:
        prints total time
        total_time: total time that was taken in seconds
    '''
    ## FINDING TOTAL TIME
    total_time = time.time() - start_time
    ## CONVERTING TO HOURS, MINUTES, AND SECONDS
    h, m, s = convert_sec_to_hms(total_time)
    ## PRINTING
    print("%s %d hrs, %d mins, %d sec"%(string, h, m, s) )
    return total_time


### CLASS FUNCTION TO TRACK TIME
class track_time:
    '''
    The purpose of this function is to track time.
    INPUTS:
        void
        
    FUNCTIONS:
        time_elasped: 
            function to print total time elapsed
    '''
    def __init__(self):
        ## START TRACKING TIME
        self.start = datetime.now()
        return
    ## FUNCTION TO PRINT TIME
    def time_elasped(self, prefix_string = None):
        ''' Function to print time elapsed '''
        time_elapsed = datetime.now() - self.start
        if prefix_string is None:
            print( 'Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed) )
        else:
           print( '%s time elapsed' %(prefix_string) + ' (hh:mm:ss.ms) {}\n'.format(time_elapsed) ) 
        return
    
    ## UPGRADING NAME
    def time_elapsed(self, **args):
        self.time_elasped(**args)
        return
