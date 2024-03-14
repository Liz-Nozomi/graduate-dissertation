# -*- coding: utf-8 -*-
"""
multi_traj_by_list.py
The purpose of this script is to load all trajectories, 
use a corresponding function to analyze the trajectories, 
then save the information for subsequent plotting/analysis. This tool 
is designed for multiple-descriptor searching and tabulation. By looping through 
multiple trajectories, we can generate a large database of descriptors that 
can be informative to the underlying physical mechanisms.

Wrriten by: Alex K. Chew (06/28/2019)

# USAGE: from multi_traj import multi_traj_analysis

CLASSES:
    multi_traj_by_list: runs analysis on multiple trajectories
    
FUNCTIONS:
    
*** UPDATES ***

"""
#########################
### IMPORTING MODULES ###
#########################
import os
import sys
import time
import pickle # Used to store variables    
from datetime import datetime
import glob

######################
### CUSTOM MODULES ###
######################
## IMPORTING PATHS
from MDDescriptors.core.check_tools import check_multiple_paths, check_dir
## IMPORTING CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools
## IMPORTING TRACK TIME
from MDDescriptors.core.track_time import print_total_time
## FUNTION FOR IMPORTING
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

########################
### GLOBAL VARIABLES ###
########################
## DIVIDER
DIVIDER_STRING="---------------------"

###########################################################
### CLASS FUNCTION TO LOAD TRAJECTORIES AND RUN PROGRAM ###
###########################################################
class multi_traj_by_list:
    '''
    The purpose of this class is to take the input directories, files, and function 
    to run detailed analysis on them. Afterwards, this function will output the analyzed 
    data as a variable file.
    INPUTS:
        path_main_dir: [str]
            path to main directory that has all the simulation information. 
            This folder should have a list of files that you are analyzing.
        path_pickle: [str]
            path to pickle storage directory. Ideally, you would store 
            this somewhere in your project folder. 
        function_list: [list]
            list of classes/functions 
        function_inputs: [list]
            list of dictionary inputs
        specific_dir: [list]
            list of a specific set of directories to look into. This would 
            override globbing over all directories.
        verbose [logical, default=False]
            True if you want to print out
    OUTPUTS:
        null: this code outputs a pickle with the results.
    ALGORITHM:
        - Check if the inputs make sense
        - Check to see if the directory exists
        - Reorganize gro and xtc file as necessary
        - Find all directories
        - Loop through each directory
            - Loop through each function
            - Load trajectory (if necessary)
            - Run analysis tool
            - Export result into a pickle
    '''
    ### INITIALIZATION
    def __init__(self, 
                 path_main_dir, 
                 function_list,
                 function_inputs,
                 path_pickle_dir,
                 specific_dir = [],
                 verbose = False
                 ):
        ## COLLECTING INPUTS
        self.function_list = function_list
        self.function_inputs = function_inputs
        self.specific_dir = specific_dir
        self.verbose = verbose
        ## CHECKING PATHS
        self.path_main_dir, self.path_pickle_dir = \
            check_multiple_paths(path_main_dir, path_pickle_dir)
        
        if self.verbose is True:
            self.print_header()
        
        ## CHECKING INPUTS
        self.check_inputs()
        
        ## FINDING ALL PATHS
        self.find_all_files()
        
        ## RUNNING DESCRIPTOR APPROACH
        self.run_descriptors()
        
        ## PRINTING SUMMARY
        if self.verbose is True:
            self.print_summary()
        
        return
    
    ### FUNCTION THAT PRINTS SUMMARY
    def print_summary(self):
        '''This function prints summary of what was done for this class'''
        print(DIVIDER_STRING)
        self.print_header()
        print("Summary:")
        print("   Directory path: %s"%( self.path_main_dir ) )
        print("   Number of trajectories: %s"%(len(self.list_all_directories)))
        print("   Number of functions: %d"%(len(self.function_list)) )
        self.print_functions()
        return
        
    ### FUNCTION THAT PRINTS HEADER
    def print_header(self):
        ''' This function prints the title '''
        print("-----------------------------------")
        print("------- multi_traj_analysis  ------")
        print("-----------------------------------")
        return
    
    ### FUNCTION TO FIND ALL FILES
    def find_all_files(self):
        '''
        The purpose of this function is to look into the directories and
        find all directories.
        INPUTS:
            self: [obj]
                self object
        OUTPUTS:
            self.list_all_directories: [list]
                list of all directories
            self.list_all_directories_basename: [list]
                list of all basename directories
        '''
        ## SEEING IF DIRECTORY IS SPECIFIED
        if len(self.specific_dir) == 0:
            ## FINDING FULL LIST
            self.list_all_directories = glob.glob(os.path.join(self.path_main_dir + '/*'))
        else:
            self.list_all_directories = [ os.path.join(self.path_main_dir, each_directory) for each_directory in self.specific_dir]
        ## FINDING BASENAME
        self.list_all_directories_basename =[ os.path.basename(directory)  for directory in self.list_all_directories ]
        
        ## PRINTING
        if self.verbose is True:
            print(DIVIDER_STRING)
            print("Path: %s"%(self.path_main_dir) )
            print("Working on directories:")
            print("\n  ".join(['']+self.list_all_directories_basename) )
        return
    
    ### FUNCTION TO RUN FUNCTIONS
    def run_descriptors(self):
        '''
        The purpose of this function is to run the descriptors after 
        checking the inputs. 
        INPUTS:
            self: [obj]
                self object
        OUTPUTS:
            outputs descriptors in a form of a pickle
        '''
        ## LOOPING THROUGH TRAJECTORIES
        for idx, directory in enumerate(self.list_all_directories):
            ## PRINTING
            if self.verbose is True:
                print("\n--- Dir: %s ---"%(self.list_all_directories_basename[idx]) )
            
            ## LOOPING THROUGH EACH GRO AND XTC FILE
            for struct_idx, each_structure_xtc_combination in enumerate(self.unique_structure_xtc_list):
                ## DEFINING STRUCTURE AND XTC FILE
                structure_file = each_structure_xtc_combination[0]
                xtc_file = each_structure_xtc_combination[1]
                ## PRINTING
                if self.verbose is True:
                    print("\n Working on GRO / XTC file: %s, %s"%(structure_file, xtc_file) )
                
                ## DEFINING FUNCTION INDEXES
                function_indexes = self.index_structure_list[struct_idx]
                
                ## SEEING IF GRO AND XTC AVAILABLE IS THERE
                if structure_file is not None and xtc_file is not None:
                    ## SETTING WANT DIRECTORIES TO FALSE
                    want_only_directories = False
                else:
                    want_only_directories = True
                    
                ## LOADING THE TRAJECTORIES
                traj_data = import_tools.import_traj(directory = directory, # Directory to analysis
                                                     structure_file = structure_file, # structure file
                                                     xtc_file = xtc_file, # trajectories
                                                     verbose = False, # Verbosity
                                                     want_only_directories = want_only_directories,
                                                     )
                    
                    
                ## LOOPING THROUGH EACH INDEX
                for each_function_idx in function_indexes:
                    ## DEFINING FUNCTION
                    run_function = self.function_list[each_function_idx]
                    if self.verbose is True:
                        print("\n     Running function: %s"%( run_function.__name__ ) )
                    ## DEFINING INPUTS
                    function_inputs = self.function_inputs[each_function_idx]
                    ## REMOVING STRUCTURE AND XTC
                    function_inputs.pop("STRUCTURE_FILE", None)
                    function_inputs.pop("XTC_FILE", None)
                    ## SEEING IF GRO AND XTC AVAILABLE IS THERE
                    # if structure_file is not None and xtc_file is not None:
                    function_inputs['traj_data'] = traj_data

                    #########################
                    ###  RUNNING FUNCTION ###
                    #########################
                    ## KEEPING TRACK OF TIME
                    start_time = time.time()
                    if self.verbose is True:
                        print(DIVIDER_STRING)
                    ## RUNNING FUNCTION
                    results = run_function(**function_inputs)
                    if self.verbose is True:
                        print(DIVIDER_STRING)
                    ## PRINTING THE TIME
                    print_total_time( start_time, string = '    Total time for analysis: ')
                    ## FINDING STORAGE LOCATION
                    storage_location_path = self.get_pickle_info(run_function = run_function)
                    ## STORING PICKLE
                    self.store_pickle( results = results, 
                                       storage_location_path = storage_location_path,
                                       directory_name = self.list_all_directories_basename[idx])
                    
                    
    ### FUNCTION TO STORE ALL INFORMATION WITHIN A PICKLE FILE
    def store_pickle(self, results, storage_location_path, directory_name):
        '''
        The purpose of this function is to store all the results into a pickle file
        INPUTS:
            self: [obj]
                self object
            results: [obj]
                results object
            storage_location_path: [str]
                storage location path
            directory_name: [str]
                name of the directory
        OUTPUTS:
            pickle file within the pickle directory under the name of the class used
        '''
        ## DEFINING PICKLE PATH
        pickle_storage_path = os.path.join( storage_location_path, directory_name )
        ## PRINTING PICKLE
        with open( os.path.join( pickle_storage_path ), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([results], f, protocol=2)  # <-- protocol 2 required for python2   # -1
        if self.verbose is True:
            print("    Data collection was complete at: %s"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            print("    Pickle path: %s"%( pickle_storage_path ) )
        return 
                    
    ### FUNCTION TO GET STORAGE INFORMATION FOR PICKLE
    def get_pickle_info(self, run_function):
        '''
        The purpose of this script is to get location of storage for 
        your pickle.
        INPUTS:
            self: [obj]
                self object
            run_function: [func]
                function
        OUTPUTS:
            storage_location_path: [str]
                storage location for a specific function
            self.path_working_dir: current working directory path
            self.pickle_file_path: pickle file path
            self.path_pickle_file: Path to pickle file
            self.pickle_file_path_date: Path to pickle file and date
        '''
        ## GETTING DATE
        # date=time.strftime("%y%m%d") # Finding date
        ## DEFINING STORAGE LOCATION
        # storage_location_path = os.path.join( self.path_pickle_dir, run_function.__name__, date )
        storage_location_path = os.path.join( self.path_pickle_dir, run_function.__name__)
        # storage_location_path = os.path.join( storage_location_dir, date  )
        ## CHECK IF LOCATION DIRECTORY EXISTS
        check_dir(storage_location_path)            
        return storage_location_path
    
    ### FUNCTION TO CHECK INPUTS
    def check_inputs(self):
        ''' 
        This function simply checks the inputs
        INPUTS:
            self: [obj]
                self object
        OUTPUTS:
            self.function_input_names: [list]
                list of function input names
        '''
        ## CHECKING IF DIRECTORY EXISTS
        if os.path.isdir(self.path_main_dir) is False:
            print("Error! Your simulation path is incorrect!")
            print("Check path: %s"%(self.path_main_dir)  )
            sys.exit()
        ## CHECKING INPUTS
        elif len(self.function_list) == 0:
            print("Error! function_list does not have any input classes. Perhaps there is something wrong with your input!")
            print("Stopping here! Please check your inputs!")
            sys.exit()
        ## CHECKING TO SEE IF THE LENGTH OF INPUTS AND DESCRIPTORS MATCH
        elif len(self.function_list) != len(self.function_inputs):
            print("Your input descriptor class and inputs do not match!")
            print("Total descriptor classes: %d"%(len(self.function_list) ) )
            print("Total descriptor inputs: %d"%(len(self.function_inputs) ) )
            print("Please double-check your inputs")
            sys.exit()
        else:
            print(DIVIDER_STRING)
            print("No errors in inputs!")
            print("Total functions: %d"%(len(self.function_list)))
            
        ## CHECKING GRO AND XTC
        self.check_gro_xtc_availability()
        ## REORDERING GRO AND XTC
        self.find_order_gro_xtc()            
        ## FINDING FUNCTION INPUT NAMES
        self.function_names = [ each_function.__name__ for each_function in self.function_list ]
        ## FUNCTION NAME ORDERS
        self.function_names_ordered = [ self.function_names[each_index] for each_index in self.index_structure_list_flatten ]
        
        ## PRINTING 
        if self.verbose is True:
            self.print_functions()
        return
    
    ### FUNCTION TO PRINT FUNCTIONS
    def print_functions(self):
        '''
        This function prints the functions list
        '''
        print(DIVIDER_STRING)
        print("Function name order of running:")
        print(", ".join(self.function_names_ordered))
        # print("Structure / XTC files:")
        print("Structure files  :", ", ".join(str(each_entry[0]) for each_entry in self.structure_xtc_file_list_ordered))
        print("XTC files        :", ", ".join(str(each_entry[1]) for each_entry in self.structure_xtc_file_list_ordered))
        return
    
    ### FUNCTION TO CHECK CONSISTENCY OF GRO AND XTC FILE
    def check_gro_xtc_availability(self):
        ''' 
        The purpose of this funtion is to check gro and xtc availability. 
        If not available, we will input a "None" into the gro/xtc dictionary
        '''
        if self.verbose is True:
            print(DIVIDER_STRING)
            print("*** Checking GRO / XTC file names for each function ***")
        ## LOOPING THROUGH EACH INPUT
        for idx, inputs in enumerate(self.function_inputs):
            ## CHECKING IF EXISTS
            if 'STRUCTURE_FILE' not in inputs.keys() or 'XTC_FILE' not in inputs.keys():
                   ## INSERTING TO GRO AND XTC
                   self.function_inputs[idx]['STRUCTURE_FILE'] = None
                   self.function_inputs[idx]['XTC_FILE'] = None
                   if self.verbose is True:
                       ## PRINTING
                       print("")
                       print("Gro, xtc not found for '%s' function, setting them to None!"%( self.function_list[idx].__name__  ) )
                       print("If this is an error, please input 'STRUCTURE_FILE' and 'XTC_FILE' for function ")
                       print("")
                   
        return
                
    ### FUNCTION TO FIND ALL GRO FILES
    def find_order_gro_xtc(self):
        '''
        The purpose of this function is to find matching gro and xtc files. 
        If they are matching, then we reorder so that we limit the number 
        of gro and xtc loadings. The "None" gro files and xtc files are 
        completed first as they are the fastest ones. 
        INPUTS:
            self: [obj]
                self object
        OUTPUTS:
            self.unique_structure_xtc_list: [list]
                unique structure and xtc list
            self.index_structure_list: [list of list]
                list of list containing function indexes that matches the unique structure
            self.index_structure_list_flatten: [list]
                list of order that the functions will run in
            self.structure_xtc_file_list_ordered: [list]
                list of structured file
            self.function_inputs_ordered: [list]
                list of inputs ordered
            self.function_list_ordered: [list]
                list of functions ordered
        '''
        ## LOOPING TO FIND LIST OF GRO AND XTC FILES
        self.structure_xtc_file_list = [ [each_input['STRUCTURE_FILE'],
                                      each_input['XTC_FILE']] 
                                        for each_input in self.function_inputs]
        
        ## FINDING UNIQUE LIST
        self.unique_structure_xtc_list = [list(x) for x in set(tuple(x) for x in self.structure_xtc_file_list)]
        
        ## CREATING INDEX
        self.index_structure_list = []
        
        ## LOOPING THROUGH EACH INDEX
        for each_unique in self.unique_structure_xtc_list:
            ## LOOPING THROUGH STRUCTURE LIST
            self.index_structure_list.append([idx for idx, x in \
                                    enumerate(tuple(x) for x in self.structure_xtc_file_list) if x == tuple(each_unique) ])
        
        ## FLATTENING LIST OF LIST
        self.index_structure_list_flatten = calc_tools.flatten_list_of_list(self.index_structure_list)
        
        ## REORDERING
        self.structure_xtc_file_list_ordered = [ self.structure_xtc_file_list[each_index] for each_index in self.index_structure_list_flatten]
        self.function_inputs_ordered = [ self.function_inputs[each_index] for each_index in self.index_structure_list_flatten]
        self.function_list_ordered = [ self.function_list[each_index] for each_index in self.index_structure_list_flatten]
        return

### DUMMY FUNCTION 1
def dummy_func1( gro, xtc ):
    ''' dummy function that just prints all gro and xtc file'''
    print("STRUCTURE file: %s"%(gro) )
    print("XTC file: %s "%(xtc) )
    return

### DUMMY FUNCTION 2
def dummy_func2( traj_data, gro, xtc ):
    ''' dummy function that just prints all gro and xtc file'''
    print("traj_data: ", traj_data )
    print("STRUCTURE file: %s"%(gro) )
    print("XTC file: %s "%(xtc) )
    return

### FUNCTION TO DEBUG
def load_vars_to_debug_multi_traj():
    '''
    This function loads all variables necessary to debug multitraj. This contains 
    three dummy function inputs:
        1 - input with No structure / xtc file inputted
        2 - another function with gro and xtc file
        3 - same as function 1, but with structure / xtc set to None
    OUTPUTS:
        function_list: [list]
            function list
        function_inputs: [list]
            function input list
        
    '''
    ## DEFINING EMPTY LIST FOR FUNCTIONS AND INPUTS
    function_list = []
    function_inputs = []
    
    ## IMPORTING DUMMY FILE 1
    dummy_inputs = {
            # 'STRUCTURE_FILE': None,
            # 'XTC_FILE': None,
            'gro': 'gro_file',
            'xtc': 'xtc_file',
            }
    
    ## APPENDING
    function_list.append( dummy_func1 )
    function_inputs.append( dummy_inputs )
        
    ## IMPORTING DUMMY FILE 2
    dummy_inputs = {
            'STRUCTURE_FILE': 'sam_equil.gro',
            'XTC_FILE': 'sam_equil.xtc',
            'gro': 'gro_file',
            'xtc': 'xtc_file',
            }
    
    ## APPENDING
    function_list.append( dummy_func2 )
    function_inputs.append( dummy_inputs )
    
    ## IMPORTING DUMMY FILE 3
    dummy_inputs = {
            'STRUCTURE_FILE': None,
            'XTC_FILE': None,
            'gro': 'gro_file',
            'xtc': 'xtc_file',
            }
    ## APPENDING
    function_list.append( dummy_func1 )
    function_inputs.append( dummy_inputs )
    return function_list, function_inputs

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING MAIN PATH
    path_main_dir = r"R:\scratch\nanoparticle_project\simulations\190520-2nm_ROT_Sims_updated_forcefield_new_ligands"
    path_pickle_dir = r"/Volumes/akchew/scratch/nanoparticle_project/scripts/analysis_scripts/PICKLE"
    
    ## DEFINING FUNCTIONS
    function_list, function_inputs = load_vars_to_debug_multi_traj()

    ## DEFINING SPECIFIC DIRECTORY
    specific_dir = ['EAM_300.00_K_2_nmDIAM_ROT014_CHARMM36jul2017_Trial_1',
                    'EAM_300.00_K_2_nmDIAM_ROT005_CHARMM36jul2017_Trial_1',]    

    ## DEFINING INPUTS FOR ANALYSIS
    analysis_inputs = {
            'path_main_dir'     : path_main_dir,
            'path_pickle_dir'   : path_pickle_dir,
            'function_list'     : function_list,
            'function_inputs'   : function_inputs,
            'specific_dir'      : specific_dir,
            'verbose'           : True,
            }    
    ## DEFINING ANALYSIS
    analysis = multi_traj_by_list( **analysis_inputs)
    
    
    
