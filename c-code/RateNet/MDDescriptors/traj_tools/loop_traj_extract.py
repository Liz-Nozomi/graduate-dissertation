# -*- coding: utf-8 -*-
"""
loop_traj_extract.py
The purpose of this script is to extract loop trajectory information. 


"""
## IMPORTING MAIN MODULES
import os
import numpy as np
import glob
import pandas as pd
from MDDescriptors.core.decoder import decode_name
from MDDescriptors.core.pickle_tools import load_pickle_results
from MDDescriptors.core.track_time import track_time

## DEFINING ANALYSIS DIRECTORY
ANALYSIS_FOLDER = "analysis"
## DEFINING PICKLE NAME
RESULTS_PICKLE = "results.pickle"

### FUNCTION TO LOAD THE PICKLE
def load_results_pickle(path_to_sim,
                        func,
                        analysis_folder = ANALYSIS_FOLDER,
                        results_pickle = RESULTS_PICKLE,
                        verbose = True,):
    '''
    The purpose of this function is to load the results pickle from the 
    loop_traj.py function.
    
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        func: [obj]
            function object used -- which will be used to generate the name
        analysis_folder: [str]
            analysis folder
        results_pickle: [str]
            results pickle
        verbose: [logical]
            True if you want to print verbosely
    '''    
    
    ## GETTING FUNC NAME
    func_name = func.__name__
    
    ## TRACKING TIME
    tracker = track_time()
    
    ## PRINTING
    if verbose is True:
        print("Loading pickle from: %s"%(path_to_sim) )
        print("--> Analysis folder: %s"%(analysis_folder) )
        print("--> Pickle name: %s"%(results_pickle) )
        print("--> Function: %s"%(func_name) )

    
    ## GETTING RELATIVE PATH TO PICKLE
    relative_path_to_pickle = os.path.join(analysis_folder,
                                           func_name,
                                           results_pickle)
    
    ## DEFINING PATH TO PICKLE
    path_to_pickle=os.path.join(path_to_sim,
                                relative_path_to_pickle)
    ## LOADING
    results = load_pickle_results(path_to_pickle)[0]
    
    if verbose is True:
        tracker.time_elapsed()
    
    return results
    


#############################################
### CLASS FUNCTION TO EXTRACT INFORMATION ###
#############################################
class extract_multiple_traj:
    '''
    The purpose of this function is to extract information from multiple trajectories.
    INPUTS:
        path_to_sim_list: [list]
            path to simulation list
    OUTPUTS:
    
    ALGORITHM:
        - Input folder path and directories
        - Input a decoder function
        - Loop through and search for specific folders
        - Run analysis based on these folders
        - Use the results to generate plots
    '''
    def __init__(self,
                 path_to_sim_list,
                 ):
        ## GETTING PATH TO SIMULATION
        self.path_to_sim_list = path_to_sim_list
        
        ## GETTING LIST OF SIMULATIONS WITHIN FOLDER
        self.full_sim_list =  np.array(glob.glob(self.path_to_sim_list + "/*"))
        
        return
        
    ### FUNCTION TO DECODE
    @staticmethod
    def decode_sim(sim_name,
                   decode_type='nanoparticle'):
        '''
        This function decodes the simulation list. Note that this function uses 
        the 'decode_name' function. 
        INPUTS:
            sim_name: [str]
                name of the simulation
            decode_type: [str]
                decoding type you are using
        OUTPUTS:
            decoded_name: [dict]
                dictionary of decoded names
        '''
        decoded_name = decode_name(sim_name, decode_type=decode_type)
        return decoded_name
    
    ### FUNCTION TO DECORE ALL SIMS
    def decode_all_sims(self,
                        decode_type='nanoparticle'):
        '''
        This function decodes all the simulations.
        INPUTS:
            decode_type: [str]
                decoding type you are using
        OUTPUTS:
            self.decoded_name_list: [list]
                list of decoded names
            self.decoded_name_df: [df]
                dataframe of decoded name
        '''
        self.decoded_name_list = [ self.decode_sim(os.path.basename(path_to_sim), decode_type=decode_type) for path_to_sim in self.full_sim_list]
        self.decoded_name_df = pd.DataFrame(self.decoded_name_list)
        return 
        
    ### FUNCTION TO LOAD ALL THE RESULTS
    def load_results(self,
                     idx,
                     func,
                     analysis_folder = ANALYSIS_FOLDER,
                     results_pickle = RESULTS_PICKLE,
                     verbose = True,
                     ):
        '''
        This function laods all the results for a given function, for a specific 
        index
        INPUTS:
            idx: [int]
                index in the list that you want to load
            func: [function]
                function that you want to load
            analysis_folder: [str]
                analysis folder
            results_pickle: [str]
                pickle results name
        OUTPUTS:
            results: [obj]
                results output
        '''
        ## LOADING THE RESULTS
        results = load_results_pickle(path_to_sim = self.full_sim_list[idx],
                        func = func,
                        analysis_folder = analysis_folder,
                        results_pickle = results_pickle,
                        verbose = verbose,)

        return results