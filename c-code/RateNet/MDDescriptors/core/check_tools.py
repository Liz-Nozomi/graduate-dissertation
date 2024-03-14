# -*- coding: utf-8 -*-
"""
check_tools.py
This script contains all checking variable/function tools.

FUNCTIONS:
        ### VARIABLE CHECKING
        check_spyder:
            checks if you are running on spyder
        check_exists: **DEPRECIATED** Checks if a value exists. If not, give it a default value.
        check_file_exist: 
            Checks if a file exists based on system path
        check_path: 
            checks path to server
        check_dir: 
            checks if directory exists
        check_testing:
            checking testing

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
UPDATES:
    20180720 - AKC - included function to see if a file exists
    20200228 - AKC - added check spyder function
"""
## IMPORTING MODULES
import sys
import os

### FUNCTION THAT CHECKS IF YOU ARE ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False

### FUNCTION TO CHECK IF VARIABLE EXISTS
def check_exists(value, default=None ):
    '''
    The purpose of this function is to see if the value exists. If not, assign the variable with a default value. If so, set the variable equal to the value
    INPUTS:
        value: Some value
        default: default value
    OUTPUTS:
        return_value: return value
    '''
    try:
        return_value = value
    except:
        return_value = default
    return return_value

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

### FUNCTION TO CHECK IF FILE EXISTS
def check_file_exist(file_path):
    '''
    The purpose of this function is to check if a file exists. 
    INPUTS:
        file_path:[str] 
        file path to check if file exists
    OUTPUTS:
        logical: [logical]
            True or false logical. True if the file exists
    '''
    import os.path
    return os.path.exists(file_path)

### FUNCTION TO CHECK EXISTENCE OF A FILE
def stop_if_does_not_exist(file_path):
    '''
    The purpose of this function is to check if file exists. If not, this will 
    raise an error message and stop the function entirely.
    INPUTS:
        file_path: [str]
            file path to see if exists
    OUTPUTS:
        void: exists if the file does not exist
    '''
    exist = check_file_exist(file_path=file_path)
    if exist == False:
        print("Error! File does not exist:")
        print(file_path)
        print("Stopping here to prevent further errors!")
        sys.exit(1)
    return

### FUNCTION TO GET THE PATH ON THE SERVER
def check_path(path):
    '''
    The purpose of this function is to change the path of analysis based on the current operating system. 
    INPUTS:
        path: [str]
            Path to analysis directory
    OUTPUTS:
        path (Corrected)
    '''
    ## IMPORTING MODULES
    import getpass
    
    ## CHANGING BACK SLASHES TO FORWARD SLASHES
    backSlash2Forward = path.replace('\\','/')
    
    ## TRYING TO FIX PATHS
    try:
    
        ## CHECKING PATH IF IN SERVER
        if sys.prefix == '/usr' or sys.prefix == '/home/akchew/envs/cs760' or '/home' in sys.prefix: # At server
            ## CHECKING THE USER NAME
            user_name = getpass.getuser() # Outputs $USER, e.g. akchew, or bdallin
            
            # Changing R: to /home/akchew
            path = backSlash2Forward.replace(r'R:','/home/' + user_name)
            if path.startswith("S:"):
                path = path.replace(r'S:','/home/shared')
            path = path.replace(r'/Volumes/','/home/' )
        
        ## AT THE MAC
        elif '/Users/' in sys.prefix or '/Applications/PyMOL.app/Contents' in sys.prefix or 'anaconda' in sys.prefix:
            ## GETTING SPLIT 
            path_split = path.split('\\')
            # backSlash2Forward.split('/')
            # path.split('\\')
            
            ## DEFINING S AND R
            initial_path = path_split[0]            
            
            ## SEEING IF VOLUMES ARE IN THE PATH
            if '/Volumes' not in initial_path:
            
                ## LISTING ALL VOLUMES
                volumes_list = os.listdir("/Volumes")
                ## LOOPING UNTIL WE FIND THE CORRECT ONE
                final_user_name =[each_volume for each_volume in volumes_list if 'akchew' in each_volume ][0]
                if initial_path != 'S:':
                    final_user_name =[each_volume for each_volume in volumes_list if 'akchew' in each_volume ][0]
                else:
                    final_user_name = 'shared'
                ## CHANGING R: to /Volumes
                path = backSlash2Forward.replace(path_split[0],'/Volumes/' + final_user_name)
    except Exception:
        pass
    ## OTHERWISE, WE ARE ON PC -- NO CHANGES
    
    return path


### FUNCTION TO CHECK MULTIPLE PATHS
def check_multiple_paths( *paths ):
    ''' 
    Function that checks multiple paths
    INPUTS:
        *paths: any number of paths        
    OUTPUTS:
        correct_path: [list]
            list of corrected paths
    '''
    correct_path = []
    ## LOOPING THROUGH
    for each_path in paths:
        ## CORRECTING
        correct_path.append(check_path(each_path))
    
    ## CONVERTING TO TUPLE
    correct_path = tuple(correct_path)
    return correct_path

### FUNCTION TO SEE IF TESTING SHOULD BE TURNED ON
def check_testing():
    '''
    The purpose of this function is to turn on testing if on SPYDER
    INPUTS:
        void
    OUTPUTS:
        True or False depending if you are on the server
    '''
    ## CHECKING PATH IF IN SERVER
    # if sys.prefix != '/Users/alex/anaconda' and sys.prefix != r'C:\Users\akchew\AppData\Local\Continuum\Anaconda3' and sys.prefix != r'C:\Users\akchew\AppData\Local\Continuum\Anaconda3\envs\py35_mat': 
    if any('SPYDER' in name for name in os.environ):
        print("*** TESTING MODE IS ON ***")
        testing = True
    else:
        testing = False
    return testing
