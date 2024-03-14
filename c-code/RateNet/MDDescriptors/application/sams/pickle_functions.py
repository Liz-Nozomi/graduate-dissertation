"""
pickle_functions.py
contains function to load and save pickle files

CREATED ON: 04/07/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import pickle

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
## FUNCTION TO LOAD PICKLE FILES
def load_pkl( path_pkl ):
    r'''
    Function to load data from pickle file
    '''
    print( "LOADING PICKLE FROM %s" % ( path_pkl ) )
    with open( path_pkl, 'rb' ) as input:
        data = pickle.load( input )
        
    return data

## FUNCTION TO SAVE PICKLE FILES
def save_pkl( data, path_pkl ):
    r'''
    Function to save data as pickle
    '''
    print( "PICKLE FILE SAVED TO %s" % ( path_pkl ) )
    with open( path_pkl, 'wb' ) as output:
        pickle.dump( data, output, pickle.HIGHEST_PROTOCOL )
            