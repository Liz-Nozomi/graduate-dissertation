# -*- coding: utf-8 -*-
"""
initialize.py
This script contains all importing and exporting functions for analysis tools. Simply load the functions by "import import_export"

FUNCTIONS:
    ### TRAJECTORY FUNCTIONS
    load_traj_from_dir: Load md trajectory given gro + xtc
    print_traj_general_info: Prints general information of the trajectory
    checkPath2Server: Checks the current path of the script that is running (Windows/ MAC/ linux, etc.)
    
    ### TIME FUNCTIONS
    convert2HoursMinSec: Converts seconds, minute, hours, etc. to more user friendly times
    getDate: Simply gets the date for today
    
    ### DIRECTORY FUNCTIONS
    check_dir: Creates a directory if it does not exist
    
    ### DICTIONARY FUNCTIONS
    make_dict_avg_std: Makes average and standard deviation as a dictionary

IMPORTANT NOTES:
    - checkPath2Server - Function may need to be changed to match the user (currently fixed for akchew)

Created on: 12/3/2017

Author(s):
    Alex K. Chew (alexkchew@gmail.com)

"""
### IMPORTING MODULES
import time
import sys
import os

### FUNCTION TO GET THE PATH ON THE SERVER
def checkPath2Server(path2AnalysisDir):
    '''
    The purpose of this function is to change the path of analysis based on the current operating system. 
    Inputs:
        path2AnalysisDir: Path to analysis directory
    Outputs:
        path2AnalysisDir (Corrected)
    '''
    ## IMPORTING MODULES
    import getpass
    
    ## CHANGING BACK SLASHES TO FORWARD SLASHES
    backSlash2Forward = path2AnalysisDir.replace('\\','/')
    
    ## CHECKING PATH IF IN SERVER
    if sys.prefix == '/usr' or sys.prefix == '/home/akchew/envs/cs760': # At server
        ## CHECKING THE USER NAME
        user_name = getpass.getuser() # Outputs $USER, e.g. akchew, or bdallin
        
        # Changing R: to /home/akchew
        path2AnalysisDir = backSlash2Forward.replace(r'R:','/home/' + user_name)
    
    ## AT THE MAC
    elif '/Users/' in sys.prefix:
        ## LISTING ALL VOLUMES
        volumes_list = os.listdir("/Volumes")
        ## LOOPING UNTIL WE FIND THE CORRECT ONE
        final_user_name =[each_volume for each_volume in volumes_list if 'akchew' in each_volume ][-1]
        ## CHANGING R: to /Volumes
        path2AnalysisDir = backSlash2Forward.replace(r'R:','/Volumes/' + final_user_name)
    
    ## OTHERWISE, WE ARE ON PC -- NO CHANGES
    
    return path2AnalysisDir

### FUNCTION TO KEEP TRACK OF TIME
def convert2HoursMinSec( seconds ):
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
    h, m = divmod(m, 60)# RADIAL DISTRIBUTION FUNCTION SCRIPT #
    return h, m, s

### FUNCTION TO GET TODAY'S DATE
def getDate():
    '''
    This function simply gets the date for figures, etc.
    INPUTS: None
    OUTPUTS:
        Date: Date as a year/month/date
    '''
    Date=time.strftime("%y%m%d") # Date for the figure name 
    
    return Date
def getDateTime():
    '''
    This function simply gets date + hour, minute, seconds
    INPUTS:
        NONE
    OUTPUTS:
        Date_time: date and time as YEAR, MONTH, DAY, HOUR, MINUTE, SECONDS (e.g. '2017-12-15 08:27:38')
    '''
    Date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return Date_time
### FUNCTION TO CREATE DIRECTORIES
def check_dir(directory):
    '''
    This function checks if the directory exists. If not, it will create one.
    INPUTS:
        directory: directory path
    OUTPUTS:
        none, a directory will be created
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

### FUNCTION TO MAKE DICTIONARY OF AVERAGE AND STANDARD DEVIATIONS
def make_dict_avg_std(average,std):
    '''
    The purpose of this script is simply to take your mean (average) and standard deviation to create a dictionary.
    INPUTS:
        average: average value(s)
        std: standard deviation
    OUTPUTS:
        dict_object: dictionary containing {'avg': value, 'std': value}
    '''
    return {'avg':average, 'std':std}
