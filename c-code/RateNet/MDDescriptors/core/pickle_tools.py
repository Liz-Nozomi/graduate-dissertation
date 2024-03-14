# -*- coding: utf-8 -*-
"""
pickle_tools.py

This script contains all pickling tools that you will need to load and store pickles.

CREATED ON: 07/26/2019

FUNCTIONS:
    load_pickle: 
        loads pickle given a file path
    pickle_results:
        function to pickle results
    load_pickle_results:
        function to load the results
    save_and_load_pickle:
        save and loads pickles. NOTE: This is useful if you want to run calculations 
        and save files -- it should speed up your workflow!

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
UPDATES:
    20200228 - added 'save_and_load_pickle' to speed up intermediate computations
## USAGE:
    import MDDescriptors.core.pickle_tools as pickle_funcs
    
"""

## IMPORTING FUNCTIONS
try:
    import cPickle as pickle # cPickle is known to speed things up
except:
    import pickle
import sys
import os

## ADDING PATH FUNCTIONS 
import MDDescriptors.core.check_tools as check_tools

### FUNCTION TO STORE RESULTS
def pickle_results(results, pickle_path, pickle_protocol = 4,verbose = False):
    '''
    This function stores the results for pickle.
    INPUTS:
        results: [list]
            list of results
        pickle_path: [str]
            path to pickle location
        verbose: [logical]
            True if you want verbosely print
    OUTPUTS:
        no output text, just store pickle
    '''
    ## VERBOSE
    if verbose is True:
        print("Storing pickle at: %s"%(pickle_path) )
    ## STORING PICKLES
    with open( os.path.join( pickle_path ), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([results], f, protocol=pickle_protocol)  # <-- protocol 2 required for python2   # -1
    return 

### MAIN FUNCTION TO LOAD PICKLE GIVEN A PICKLE DIRECTORY
def load_pickle_results( file_path, verbose = False ):
    '''
    The purpose of this function is to load pickle for a given file path
    INPUTS:
        file_path: [str]
            path to the pickle file
        verbose: [str]
            True if you want verbose pickling
    OUTPUTS:
        results: [obj]
            results from your pickle
    '''
    # PRINTING
    if verbose is True:
        print("LOADING PICKLE FROM: %s"%(file_path) )    
    ## LOADING THE DATA
    with open(file_path,'rb') as f:
        # file_path = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        if sys.version_info[0] > 2:
            results = pickle.load(f, encoding='latin1') ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        elif sys.version_info[0] == 2: ## ENCODING IS NOT AVAILABLE IN PYTHON 2
            results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        else:
            print("ERROR! Your python version is not 2 or greater! We cannot load pickle files below Python 2!")
            sys.exit()
    return results


### MAIN FUNCTION TO LOAD PICKLE GIVEN A PICKLE DIRECTORY
def load_pickle( file_path, verbose = False ):
    '''
    The purpose of this function is to load pickle for a given file path
    INPUTS:
        file_path: [str]
            path to the pickle file
        verbose: [str]
            True if you want verbose pickling
    OUTPUTS:
        results: [obj]
            results from your pickle
    '''
    # PRINTING
    if verbose is True:
        print("LOADING PICKLE FROM: %s"%(file_path) )    
    ## LOADING THE DATA
    with open(file_path,'rb') as f:
        # file_path = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        if sys.version_info[0] > 2:
            results = pickle.load(f, encoding='latin1') ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        elif sys.version_info[0] == 2: ## ENCODING IS NOT AVAILABLE IN PYTHON 2
            results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        else:
            print("ERROR! Your python version is not 2 or greater! We cannot load pickle files below Python 2!")
            sys.exit()
    return results

### FUNCTION TO LOAD PICKLE RESULTS
def load_pickle_from_descriptor( pickle_dir,
                                 descriptor_class,
                                 pickle_name,
                                 verbose = True):
    '''
    The purpose of this function is to load pickle from a descriptor that was analyzed 
    by multi_traj_by_list.py. 
    INPUTS:
        pickle_dir: [str]
            pickle directory, e.g. r"R:\scratch\nanoparticle_project\scripts\analysis_scripts\PICKLE_ROT"
        descriptor_class: [class object]
            descriptor class that you are using (we just need the name really)
        pickle_name: [str]
            name of the pickle
        verbose: [str]
            whether or not to print something
    OUTPUTS:
        results: [obj]
            results from your pickle
    '''
    ## DEFINING FULL PATH
    pickle_path = os.path.join( pickle_dir, descriptor_class.__name__,  pickle_name )
    pickle_path = check_tools.check_path(pickle_path)
    
    ## LOADING PICKLE
    results = load_pickle( file_path = pickle_path , 
                           verbose = verbose )[0]

    return results

### FUNCTION TO SAVE PICKLE
def save_and_load_pickle(function, 
                         inputs, 
                         pickle_path,
                         rewrite = False,
                         verbose = True):
    '''
    The purpose of this function is to save and load pickle whenever you 
    have a function you want to run and you want to store the outputs. 
    The idea is to lower the amount of redundant calculations by storing the 
    outputs into a pickle file, then reloading if necessary. 
    
    INPUTS:
        function: [func]
            function you want to run
        inputs: [dict]
            dictionary of inputs
        pickle_path: [str]
            path to store the pickle in
        rewrite: [logical]
            True if you want to rewrite the pickle file
        verbose: [logical]
            True if you want to print the details
    OUTPUTS:
        results: [list]
            list of your results
    '''
    ## RUNNING THE FUNCTION
    if os.path.isfile(pickle_path) == False or rewrite is True:
        ## PRINTING
        if verbose is True:
            print("Since either pickle path does not exist or rewrite is true, we are running the calculation!")
            print("Pickle path: %s"%(pickle_path) )
        ## PERFORMING THE TASK
        results = function(**inputs)
        
        
        if verbose is True:
            print("Saving the pickle file in %s"%(pickle_path) )
        ## STORING IT
        
        pickle_results(results = results,
                       pickle_path = pickle_path)

    ## LOADING THE FILE
    else:
        ## LOADING THE PICKLE
        results = load_pickle_results(file_path =pickle_path)[0]
        ## PRINTING
        if verbose is True:
            print("Loading the pickle file from %s"%(pickle_path) )
    
    return results
