# -*- coding: utf-8 -*-
"""
import_tools.py
This contains codes on importing functions.

Created by: Alex K. Chew (alexkchew@gmail.com, 02/27/2018)

*** UPDATES ***
20180313 - AKC - Added total residues to trajectory function
20180622 - AKC - Added total frames to trajectory function
20180706 - AKC - Added functions to read file as a line

FUNCTIONS:
    read_file_as_line: reads a file as a line

CLASSES:
    import_traj: [class]
        class that can import trajectory information (uses md.load from mdtraj module)
    read_gromacs_xvg: class to read gromacs xvg information
        USAGE EXAMPLE: 
            ## READING FILES
            self.output_xvg = import_tools.read_gromacs_xvg(    traj_data = traj_data,
                                                                xvg_file = xvg_file,
                                                                variable_definition = self.variable_definition
                                                            )
    read_plumed_covar:
        Function that reads plumed covar files
        
"""
import os
# MDTRAJ TO READ TRAJECTORIES
import mdtraj as md
import pandas as pd
## IMPORTING PATHS
from MDDescriptors.core.check_tools import check_path
# FUNCTION TO MEASURE TIME
import time
## MATH MODULE
import numpy as np
## IMPORTING TRACK TIME
from MDDescriptors.core.track_time import print_total_time
## IMPORTING CALCULATION TOOLS
import MDDescriptors.core.calc_tools as calc_tools




## DEFINING VARIABLES
GMX_XVG_VARIABLE_DEFINITION = {'GYRATE': 
                                    [ 
                                    [0, 'frame',    int],
                                    [1, 'Rg',       float ],
                                    [2, 'Rg_X',     float],
                                    [3, 'Rg_Y',     float],
                                    [4, 'Rg_Z',     float],
                                ],
                            'density.xvg':
                                [
                                [0, 'distance', float,], # kg/m^3
                                [1, 'density', float,],  #  kg/m^3
                                ],
                            'potential.xvg':
                                [
                                [0, 'distance', float,], # nm
                                [1, 'potential', float,],  #  Volts
                                ],
                               }


### FUNCTION TO READ FILE AND CONVERT THE INTO LINES
def read_file_as_line(file_path, want_clean = True, verbose = False):
    '''
    The purpose of this function is to read a file and convert them into lines
    INPUTS:
        file_path: [str] full path to your file
        want_clean: [logical, default = True] True if you want to clean the data of '\n'
    OUTPUTS:
        data_full: [list] Your file as a list of strings
    '''
    ## PRINTING
    if verbose is True:
        print("READING FILE FROM: %s"%(file_path))
    ## OPENING FILE AND READING LINES
    with open(file_path, 'r') as file:
        data_full= file.readlines()
    ## CLEANING FILE OF ANY '\n' AT THE END
    if want_clean == True:
        data_full = [s.rstrip() for s in data_full]
    return data_full
    
### FUNCTION TO READ THE XVG FILE
def read_xvg(file_path):
    '''
    The purpose of this function is to read the file and eliminate all comments
    INPUTS:
        file_path: [str] full file path to xvg file
    OUTPUTS:
        self.data_full: [list] 
            full list of the original data
        self.data_extract: [list] 
            extracted data in a form of a list (i.e. no comments)
    '''
    ## PRINTING
    print("READING FILE FROM: %s"%(file_path))
    ## OPENING FILE AND READING LINES
    with open(file_path, 'r') as file:
        data_full= file.readlines()
        
    ## EXTRACTION OF DATA WITH NO COMMENTS (E.G. '@')
    try:
        final_index =[i for i, j in enumerate(data_full) if '@' in j][-1]
    except IndexError:
        # This means that no '@' was found
        final_index = 0
    data_extract = [ x.split() for x in data_full[final_index+1:] ]
    return data_full, data_extract


####################################
#### CLASS FUNCTION: import_traj ###
####################################
# This class imports all trajectory information
class import_traj:
    '''
    INPUTS:
        directory: [str]
            directory where your information is located
        structure_file: [str]
            name of your structure file
        xtc_file: [str]
            name of your xtc file
        stride: [int]
            stride to load trajectory
        want_only_directories: [logical, default = False] 
            If True, this function will no longer load the trajectory. It will simply get the directory information
        verbose: [logical, default = True]
            False if you want no output 
        index: [int]
            Frame to load the trajectory. If None, all the frames are loaded. 
        discard_overlapping_frames: [logical]
            True if you want to discard overlapping frames (False by mdtraj default)
    OUTPUTS:
        ## FILE STRUCTURE
            self.directory: directory your file is in
            
        ## TRAJECTORY INFORMATION
            self.traj: trajectory from md.traj
            self.topology: toplogy from traj
            self.residues: Total residues as a dictionary
                e.g. {'HOH':35}, 35 water molecules
            self.num_frames: [int] total number of frames
        
    FUNCTIONS:
        load_traj_from_dir: Load trajectory from a directory
        print_traj_general_info: prints the current trajectory information
    '''
    ### INITIALIZING
    def __init__(self, directory, 
                       structure_file, 
                       xtc_file, 
                       stride = None,
                       standard_names = False,
                       want_only_directories = False,
                       verbose = True,
                       index = None,
                       discard_overlapping_frames = False):
        ## STORING
        self.verbose = verbose
        self.index = index
        
        ### STORING INFORMATION
        self.directory = directory
        self.file_structure = structure_file
        self.file_xtc = xtc_file
        self.stride = stride
        self.standard_names = standard_names
        self.discard_overlapping_frames = discard_overlapping_frames
        
        if want_only_directories == False:
            ### START BY LOADING THE DIRECTORY
            self.load_traj_from_dir()
            
            ## GENERATING RESIDUE NAMES
            self.generate_residue_names()
            
            # PRINTING GENERAL TRAJECTORY INFORMATION
            if self.verbose is True:
                self.print_traj_general_info()
    
    
    ### FUNCTION TO LOAD TRAJECTORIES
    def load_traj_from_dir(self):
        '''
        The purpose of this function is to load a trajectory given an xtc, gro file, and a directory path
        INPUTS:
            self: class object
        OUTPUTS:
            self.traj: [class] trajectory from md.traj
            self.topology: [class] toplogy from traj
            self.num_frames: [int] total number of frames
        '''
        ## CREATING PATHS
        _EXT = os.path.splitext( self.file_xtc )[1]
        _EXT_STR = os.path.splitext( self.file_structure )[1]
        self.path_structure = os.path.join( self.directory, self.file_structure )
        self.path_xtc = os.path.join( self.directory, self.file_xtc )

        ## PRINT LOADING TRAJECTORY
        if self.verbose is True:
            print("")
            print('Loading trajectories from: %s'%(self.directory))
            print('   Structure file: %s' %(self.file_structure) )        
            print('   XTC file: %s' %(self.file_xtc))
            if self.stride != None:
                print("   Loading with stride of: %d"%(self.stride) )

        ## LOADING TRAJECTORIES
        start_time = time.time()
        
        ## LOADING WHOLE TRAJECTORIES
        if self.index is None:
                    
            if _EXT == ".arc":
                ## TINKER TRAJECTORY
                self.traj =  md.load( self.path_xtc,
                                      stride = self.stride)
                traj_structure = md.load( self.path_structure ) # structure should be pdb
                self.traj.topology = traj_structure.topology
            
            
            else: # if _EXT == ".xtc" or _EXT == ".gro":
            
                if _EXT_STR == ".pdb": # structure file is pdb, uses standard_names
                    ## GROMACS TRAJECTORY 
                    self.traj =  md.load( self.path_xtc, 
                                          top = self.path_structure,
                                          stride = self.stride,
                                          standard_names = self.standard_names )  # Prevent MDTraj from changing residue names
                else: # structure file is gro, no standard_names option
                    ## GROMACS TRAJECTORY
                    self.traj =  md.load( self.path_xtc, 
                                          top = self.path_structure,
                                          stride = self.stride )

        else:
            ## XTC FILES
            if _EXT == ".xtc":
                ## FOR PDB FILES
                if _EXT_STR == ".pdb": # structure file is pdb, uses standard_names
                    ## GROMACS TRAJECTORY 
                    self.traj =  md.load_frame(self.path_xtc, 
                                               top = self.path_structure,
                                               index = self.index,
                                               standard_names = self.standard_names,  # Prevent MDTraj from changing residue names
                                               discard_overlapping_frames =self.discard_overlapping_frames ) 
                else: # structure file is gro, no standard_names option
                    ## GROMACS TRAJECTORY
                    self.traj =  md.load_frame( self.path_xtc, 
                                                top = self.path_structure,
                                                index = self.index )
                    
            elif _EXT == ".arc":
                ## TINKER TRAJECTORY
                self.traj =  md.load( self.path_xtc,
                                      index = self.index)
                traj_structure = md.load( self.path_structure ) # structure should be pdb
                self.traj.topology = traj_structure.topology
            if self.verbose is True:
                print("Loading single frame, index: %d"%(self.index))

        ## PRINTING TOTAL TIME
        if self.verbose is True:
            print_total_time( start_time, string = '   Total for MD load: ')

        ## GETTING TOPOLOGY
        try:
            self.topology=self.traj.topology
        except AttributeError:
            pass
        
        ## GETTING TOTAL TIME
        self.num_frames = len(self.traj)
        
        return 
    
    ### FUNCTION TO GENERATE RESIDUE NAMES
    def generate_residue_names(self):
        '''
        This function generates residue names for imported traj
        '''
        ## STORING TOTAL RESIDUES
        self.residues={}
        
        # Finding unique residues
        unique_res_names = calc_tools.find_unique_residue_names(traj = self.traj)
        for current_res_name in unique_res_names:
            # Finding total number of residues, and their indexes    
            num_residues, index_residues = calc_tools.find_total_residues(traj = self.traj, 
                                                                          resname = current_res_name)
            
            ## STORING
            self.residues[current_res_name] = num_residues
        return
        
    
    ### FUNCTION TO PRINT GENERAL TRAJECTORY INFORMATION
    def print_traj_general_info(self):
        '''This function simply takes your trajectory and prints the residue names, corresponding number, and time length of your trajectory
        INPUTS:
            self: class object
        OUTPUTS:
            Printed output
        '''
        
        ## Main Script ##
        print("---- General Information about your Trajectory -----")
        print("%s\n"%(self.traj))
        
        ## PRINTING RESIDUES
        for each_residue in self.residues.keys():
            # Printing an output
            print("Total number of residues for %s is: %s"%(each_residue, self.residues[each_residue]))
            
        # Finding total time length of simulation
        print("\nTime length of trajectory: %s ps"%(self.traj.time[-1] - self.traj.time[0]))
 
        return
    


########################################################
### DEFINING GENERALIZED XVG READER FOR GMX COMMANDS ###
########################################################
class read_gromacs_xvg:
    '''
    The purpose of this class is to read xvg files in a generalized fashion. Here, you will input the xvg file, 
    define the bounds of the xvg such that the columns are defined. By defining the columns, we will read the xvg file, 
    then extract the information. 
    INPUTS:
        traj_data: [object]
            trajectory data indicating location of the files
        xvg_file: [str]
            name of the xvg file
        variable_definition: [list]
            Here, you will define the variables such that you define the column, name, and type of variable.
            Note: the name of the variable will be used as a dictionary.
    OUTPUTS:
        ## INPUT INFORMATION
            self.variable_definition: [list]
                same as input -- definition of variables
        ## FILE PATHS
            self.file_path: [str]
                full path to the xvg file
        ## FILE INFORMATION
            self.data_full: [list]
                data with full information
            self.data_extract: [list]
                extracted data (no comments)
        ## VARIABLE EXTRACTION
            self.output: [dict]
                output data from defining the variables in a form of a dictionary
    FUNCTIONS:
        define_variables: this function extracts variable details
            
    '''
    ## INITIALIZING
    def __init__(self, traj_data, xvg_file, variable_definition ):
        
        ## STORING INPUTS
        self.variable_definition = variable_definition
        
        ## DEFINING FULL PATH
        self.file_path = traj_data.directory + '/' + xvg_file
        
        ## READING THE FILE
        self.data_full, self.data_extract = read_xvg(self.file_path)

        ## VARIABLE EXTRACTION
        self.define_variables()
    
    ## EXTRACTION OF VARIABLES
    def define_variables(self,):
        '''
        The purpose of this function is to extract variables from column data
        INPUTS:
            self: [object]
                class property
        OUTPUTS:
            self.output: [dict]
                output data from defining the variables in a form of a dictionary
        '''
        ## DEFINING EMPTY DICTIONARY
        self.output={}
        ## LOOPING THROUGH EACH CATEGORY
        for each_variable_definition in self.variable_definition:
            ## DEFINING CURRENT INPUTS
            col = each_variable_definition[0]
            name = each_variable_definition[1]
            var_type = each_variable_definition[2]
            
            ## EXTRACTING AND STORING
            self.output[name] = np.array([ x[col] for x in self.data_extract]).astype(var_type)
        return
        
### FUNCTION TO READ COVER FILE FROM PLUMED
def read_plumed_covar(path_data,
                      verbose = True):
    '''
    This function reads the covar file outputted from plumed.
    INPUTS:
        path_data: [str]
            path to the data file
        verbose: [logical]
            True if you want to print out details
    OUTPUTS:
        data: [dataframe]
            pandas dataframe, e.g.
                               time       coord  ...  restraint.coord_kappa  restraint.work
                0          0.000000    0.194326  ...                    0.0        0.000000
                1          0.020000    0.186519  ...                    0.1        0.001768
    '''
    ## READING FIRST LINE    
    with open(path_data, 'r') as f:
        lines = f.readlines()
    ## CLEANING ALL LINES
    clean_lines = [  each_line.rstrip() for each_line in lines ]
    
    ## FINDING ALL ROWS THAT HAS FIELDS
    rows_with_fields = [each_idx for each_idx, line in enumerate(clean_lines) if "FIELDS" in line ]
    # rows_with_fields_plus_1 = np.array(rows_with_fields) + 1 # Needed for skipping rows
    
    ## FINDING HEADERS
    first_line_split = clean_lines[rows_with_fields[0]].split(" ")
    fields_index = first_line_split.index("FIELDS")
    headers = first_line_split[fields_index:]
    
    ## READING THE DATA (skip first row)
    data = pd.read_csv(path_data, sep = ' ', 
                       skiprows = rows_with_fields, # 1 
                       header = None,
                       names = headers
                       ) # , skiprows = 1
    ## REMOVING A COLUMN
    data = data.drop(columns=["FIELDS"])
    
    if verbose is True:
        print("Reading CSV from: %s"%(path_data) )
    return data



