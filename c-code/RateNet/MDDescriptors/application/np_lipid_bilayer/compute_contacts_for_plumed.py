#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_contacts_for_plumed.py

The purpose of this function is to compute the plumed sij values against rij 
and compare them against the contacts code that was used previously. The idea 
would be to measure contacts versus rij for both continuous and discrete cases, 
then try to optimize the number of contacts. 

Algorithm:
    - Convert the trajectory (remove all waters)
    - Load trajectory
    - Divide the groups between nanoparticle and lipid membrane
    - For each time step, compute the distances and generate a rij matrix of size (T, N, M),
      where T is the total time, N is the number of atoms for the nanoparticle, 
      and M is the number of atoms for the lipid membrane
    - Then, compute contacts based on the cutoff radius
    - Similarly, compute coord from plumed the equation
    
    
Written by: Alex K. Chew (05/13/2020)

"""
## IMPORTING OS
import os
import numpy as np
import mdtraj as md
## IMPORTING FUNCTION FOR GENERATING GROUPS
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import get_nplm_heavy_atom_details
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups

## PATH TO PARENT SIM
from MDDescriptors.application.np_lipid_bilayer.global_vars import PARENT_SIM_PATH, IMAGE_LOC

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## DEFINING PICKLING TOOLS
from MDDescriptors.core.pickle_tools import load_pickle_results, save_and_load_pickle

## IMPORTING PLOT TOOLS
import matplotlib.pyplot as plt
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv
import MDDescriptors.core.import_tools as import_tools

## IMPORTING PARALLEL FUNCTIONS
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

## IMPORTING LAST FRAME TOOL
from MDDescriptors.core.traj_itertools import get_traj_from_last_frame

## LOADING PANDAS
import pandas as pd

## IMPORTING
from MDDescriptors.application.np_lipid_bilayer.plot_covar import read_covar

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()


## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

### FUNCTION TO COMPUTE SIJ
def compute_coord_from_plumed(distance,
                              d0 = 0,
                              n = 6,
                              m = 0,
                              r0 = 0.5):
    '''
    Tnhis function computes the coordination number from plumed.
    INPUTS:
        distance: [np.array]
            distance array
        d0: [float]
            d0 switching parameters
        n: [int]
            exponent on numerator
        m: [int]
            exponent in denominator. By default, if m = 0, then this is 2 * n
        r0: [float]
            radius of cutoff for the coordination number
    OUTPUTS:
        sij: [np.array]
            sij numpy array matrix
        
    '''
    if m == 0:
        m = 2 * n

    ## FINDING RDIST
    rdist = (distance - d0) / r0
    ## GETTING SIJ
    sij = ( 1 - (rdist ** n) ) / ( 1 -  (rdist**m)  )

    return sij

##############################################
### CLASS FUNCTION TO GET INDEXES FOR NPLM ###
##############################################
class nplm_get_indices:
    '''
    The purpose of this function is to get the nanoparticle-lipid membrane
    indices. 
    INPUTS:
        traj: [obj]
            trajectory object
        lipid_membrane_resname: [str]
            resname of lipid membrane
        verbose: [logical]
            True if you want to print out verbose information
    OUTPUTS:
        self.ligand_names: [list]
            list of ligand names
        self.np_heavy_atom_index: [np.array]
            nanoparticle heavy atom index
        self.np_heavy_atom_names: [np.array]
            nanoparticle heavy atom names
        self.lm_heavy_atom_index: [np.array]
            lipid mbembrane atom index
        self.lm_heavy_atom_names  : [np.array]
            lipid membrane heavy atom index
            
        self.np_groups: [dict]
            dictionary of nanoparticle groups and their atom names
        self.lm_groups: [dict]
            dictionary of lipid membrane groups 
            
    FUNCTIONS:
        find_np_and_tail_group_atom_indices:
            finds all nanoparticle ligand atoms and lipid membrane tail group atoms
    '''
    ## INITIALIZING
    def __init__(self, 
                 traj,
                 lipid_membrane_resname = "DOPC",
                 verbose = False):
        ## STORING
        self.lipid_membrane_resname = lipid_membrane_resname
        
        

        ## GETTING INFORMATION FOR LIGAND DETAILS
        self.ligand_names, \
        self.np_heavy_atom_index, \
        self.np_heavy_atom_names, \
        self.lm_heavy_atom_index, \
        self.lm_heavy_atom_names = get_nplm_heavy_atom_details(traj = traj,
                                                               lm_res_name = self.lipid_membrane_resname)
        ## GETTING GROUPS
        self.np_groups = generate_rotello_np_groups(traj = traj,
                                               atom_names_np =  self.np_heavy_atom_names,
                                               np_heavy_atom_index = self.np_heavy_atom_index, 
                                               verbose = verbose)
        
        ## GETTING LM GROUPS
        self.lm_groups = generate_lm_groups(traj = traj,
                                       atom_names = self.lm_heavy_atom_names,
                                       lm_heavy_atom_index = self.lm_heavy_atom_index,
                                       verbose = verbose,
                                       )            
        
        return
    
    ### FUNCTION TO GET NP AND TAIL GROUP ATOM INDEXES ONLY
    def find_np_and_tail_group_atom_indices(self,
                                            traj,
                                            np_group_keys_exclude = ['GOLD'],
                                            lm_group_keys_include = ['TAILGRPS']):
        '''
        This function simply looks for specific nanoparticle groups and lipid 
        membrane groups, then outputs the atomindices for them. 
        INPUTS:
            traj: [obj]
                trajectory object
            np_group_keys_not_in: [list]
                list of nanoparticle group keys to exclude
            lm_group_keys_include: [list]
                list of lipid membane keys to include
        OUTPUTS:
            np_atom_index: [np.array]
                nanoparticle atom index
            lm_atom_index: [np.array]
                lipid membrane atom index
        '''
    
        ## DEFINING SPECIFIC GROUPS
        np_groups_key = [each_key for each_key in self.np_groups.keys() if each_key not in np_group_keys_exclude]
        lm_groups_key = [each_key for each_key in self.lm_groups.keys() if each_key in lm_group_keys_include]


        ## GETTING ALL ATOM NAMES
        np_atom_names = calc_tools.flatten_list_of_list([self.np_groups[each_key] for each_key in np_groups_key])
        lm_atom_names = calc_tools.flatten_list_of_list([self.lm_groups[each_key] for each_key in lm_groups_key])

        ## GETTING ALL ATOM INDICES THAT MATCH
        np_atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                           residue_name = self.ligand_names[0], # assuming one
                                                           atom_names = np_atom_names)[1]
        lm_atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                           residue_name = self.lipid_membrane_resname, # assuming one
                                                           atom_names = lm_atom_names)[1]

        ## FLATTENING
        np_atom_index = np.array(calc_tools.flatten_list_of_list(np_atom_index))
        lm_atom_index = np.array(calc_tools.flatten_list_of_list(lm_atom_index))
        
        return np_atom_index, lm_atom_index

#######################################
### CLASS TO COMPUTE NPLM DISTANCES ###
#######################################
class compute_nplm_distances:
    '''
    The purpose of this function is to compute the rij between nanoparticles 
    and lipid membranes. The idea would be to use the distances to compute 
    contacts, and so forth.
    
    INPUTS:
        traj: [obj]
            trajectory object
        lipid_membrane_resname: [str]
            resname of the lipid membrane
    OUTPUTS:
        self.nplm_indices: [obj]
            indices object that contains all the information required
        self.np_atom_index: [np.array]
            nanoparticle atom index
        self.lm_atom_index: [np.array]
            lipid membrane atom index
        self.atom_pairs: [np.array, shape = (N, 2)]
            atom pairs between nanoparticle atom index and lipid membrane atom index
    
    FUNCTIONS:
        compute:
            computes all the distances
    '''
    ## INITIALIZING
    def __init__(self,
                 traj,
                 lipid_membrane_resname = "DOPC"):
        
        ## STORING TIME
        self.traj_time = traj.time
        
        ## GETTING INDICES
        self.nplm_indices = nplm_get_indices(traj = traj,
                                        lipid_membrane_resname = "DOPC")
        
        ## FINDING ATOM INDICES FOR NP AND TAIL GROUP
        self.np_atom_index, self.lm_atom_index = self.nplm_indices.find_np_and_tail_group_atom_indices(traj = traj, 
                                                                                                       np_group_keys_exclude = ['GOLD'],
                                                                                                       lm_group_keys_include = ['TAILGRPS'])
        
        ## GENERATING ATOM PAIRS
        self.atom_pairs = calc_tools.create_atom_pairs_list(atom_1_index_list = self.np_atom_index, 
                                                            atom_2_index_list = self.lm_atom_index)

        return
    
    ### FUNCTION TO COMPUTE DISTANCES
    def compute(self,
                traj,
                frames = [],
                periodic = True,
                verbose = True):
        '''
        This function computes the distances for a given set of atom pairs
        INPUTS:
            traj: [md.traj]
                trajectory ojbect
            frames: [list]
                list of frames to run
            periodic: [logical]
                True if you want to account for PBC
        OUTPUTS:
            distances: [np.array, shape = (T, N, M)]
                Distances array with the shape of num frames x num nanoparticle atoms x num lipid membrane atoms
        '''
        ## FINDING LENGTH OF THE FRAMES
        if len(frames) > 0:
            traj = traj[frames]
        ## TOTAL FRAME
        total_frames = len(traj)
        
        ## PRINTING
        if verbose is True:
            print("Computing distances for times: %d ps - %d ps"%(traj.time[0], traj.time[-1]) )
        
        ## COMPUTING DISTANCES
        distances = md.compute_distances(
                                        traj = traj,
                                        atom_pairs = self.atom_pairs,
                                        periodic = True,
                ) ## RETURNS TIMEFRAME X (NUM_ATOM_1 X NUM_GOLD) NUMPY ARRAY
        
        ## RESHAPING THE DISTANCES
        distances = distances.reshape(total_frames, 
                                      len(self.np_atom_index), 
                                      len(self.lm_atom_index))
        return distances
    
### MAIN FUNCTION TO COMPUTE NPLM DISTANCES
def main_compute_nplm_distances(path_to_sim,
                                input_prefix = "nplm_prod",
                                lipid_membrane_resname = "DOPC",
                                last_time_ps = 50000,
                                selection = 'non-Water',
                                n_procs = 1,
                                gro_output_time_ps = 0,
                                frames = [],
                                rewrite = False):
    '''
    This is the main function for computing nplm distances. For now, we assume 
    that you only care about the distance between nanoparticle ligands and the 
    tail groups of the DOPC lipid membrane.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        selection: [str]
            selection to run the contacts with
        func_inputs: [dict]
            dictionary for the main contacts function
        input_prefix: [str]
            input prefix for tpr and gro files
        rewrite: [logical]
            True if you want to rewrite
        n_procs: [int] 
            number of processors used to compute number of contacts
        frame_rate: [int]
            frame rate for number of contacts script
        gro_output_time_ps: [float]
            time to output in picoseconds
        frames: [list]
            frames you want distances for, if this is empty, we use all frames
    OUTPUTS:
        nplm_distances: [obj]
            distance object containing all indices information
        distances: [np.array, shape = (T, N, M)]
            full distance array across time, number of nanoparticle atoms, and number of lipid membrane atoms
    '''
    ## CONVERTING TRAJECTORY
    trjconv_func = convert_with_trjconv(wd = path_to_sim)
    ## GETTING ONLY SPECIFIC SELECTION
    gro_file, xtc_file, ndx_file = trjconv_func.generate_gro_xtc_specific_selection(input_prefix = input_prefix,
                                                                                    selection = selection,
                                                                                    rewrite = rewrite,
                                                                                    gro_output_time_ps = gro_output_time_ps)
    
    ## LOADING FILES
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = gro_file,
                                         xtc_file = xtc_file,
                                         )
    
    ## UPDATING TRAJECTORY BASED ON TIME
    traj_data.traj = get_traj_from_last_frame(traj = traj_data.traj,
                                              last_time_ps = last_time_ps)
    
    ## DEFINING TRAJECTORY
    traj = traj_data.traj
    
    if len(frames) > 0:
        traj = traj[frames]
    
    ## COMPUTING NPLM DISTANCES
    nplm_distances = compute_nplm_distances(traj = traj,
                                            lipid_membrane_resname = lipid_membrane_resname)
    
    ## COMPUTING OUTPUT BY SPLITTING TRAJECTORY
    distances = parallel_analysis_by_splitting_traj(traj = traj, 
                                                  class_function = nplm_distances.compute, 
                                                  n_procs = n_procs,
                                                  combine_type="concatenate_axis_0",
                                                  want_embarrassingly_parallel = True)
    
    ## TURNING OFF ATOM PAIRS
    nplm_distances.atom_pairs = []
    
    return nplm_distances, distances

### FUNCTION ATTEMPTING TO DEBUG SIJ -- NOT WORKING
def debug_attempt_for_sij(distances):
    '''
    This is simply a function that stores the debugging attempts for 
    sij. The conclusion is that I could not get sum of sij to agree with PLUMED
    since I do not know what the switching function is. Therefore, I will 
    just use PLUMED to compute coordination number and see how it compares 
    to my expected values. This code does work when you specify "SWITCH={RATIONAL}".
    '''

    ## DEFINING PARAMETERS
    coordination_parms = {
            'd0': 0.0,
            'n': 6.0,
            'm': 12.0,
            'r0': 0.35}
    
    ## FINDING DMAX
    # dmax=d0+r0*pow(0.00001,1./(nn-mm));
    dmax = coordination_parms['d0'] + coordination_parms['r0'] * 0.00001**(1/(coordination_parms['n']-coordination_parms['m']) )
    # dmax = 3

    # dmax = 3
    # coordination_parms['d0']+ coordination_parms['r0']*pow(0.00001,1./((coordination_parms['n']-(coordination_parms['m'])) ) )
    
    ## dmax = 2.7
    
    # Expected: 5586.210784
    

    # result=result*stretch+shift
    
    ## EXCLUDING THOSE DISTANCES
    
    ## SHIFTING DISTANCE
    # distance = distance * stretch + shift
    # distance = distances[0]
    distance = distances[0]
    ## GETTING MAXIMA    
    # dmax = np.max(distance)
    # coordination_parms['d0']+ coordination_parms['r0']*pow(0.00001,1./((coordination_parms['n']-(coordination_parms['m'])) ) )
    # np.max(distances)
    
    ## DEFINING STRETCH AND SHIFT
    s0 = 0
    sd = dmax
    # 1
    # dmax
    stretch = 1.0/(s0-sd)
    shift= -sd*stretch;
    # distance =  stretch * (distance + shift)
    # distance = distance[distance <= dmax]
    # distance = distances[0][ distances[0] <= dmax ]

    ## COMPUTING SIJ
    sij = compute_coord_from_plumed(distance = distance,
                                    **coordination_parms)
    print(sij.sum())
    return sij

### CLASS TO READ PLUMED INPUT
class read_plumed_input:
    '''
    The purpose of this function is to read the plumed input file.
    INPUTS:
        path_plumed_input:
            path to plumed input file
    OUTPUTS:
        
    '''
    ## INITIALIZING
    def __init__(self,
                 path_plumed_input):
        ## STORING INPUTS
        self.path_plumed_input = path_plumed_input
        
        ## READING
        self.lines = self.read()
        
        return

    ## DEFINING READ FILE
    def read(self):
        ''' FUNCTION THAT READS THE INPUT FILE '''
        with open(self.path_plumed_input, 'r') as f:
            lines = f.readlines()
        ## CLEANING ALL LINES
        clean_lines = [  each_line.rstrip() for each_line in lines ]
        ## CLEANING ALL COMMENTS
        lines_with_no_comments = [each_line for each_line in clean_lines if each_line.startswith("#") is False]
        
        
        return lines_with_no_comments
    
    ## FUNCTION TO READ THE INPUTS
    def list_lines_with_covar(self,
                              covar_list = []):
        '''
        This function lists the lines with the desired covar. 
        INPUTS:
            covar_list: [list]
                list of covar
        OUTPUTS:
            lines_with_covar: [list]
                list with covar inside
        '''
        lines_with_covar = []
        ## LOOPING
        for covar in covar_list:
            ## SEARCHING
            lines_with_covar.extend([each_line for each_line in self.lines if covar in each_line ])
        
        return lines_with_covar
    
    ### FUNCTION TO CREATE A DICT FOR KEYS
    @staticmethod
    def create_dict_given_covar_string(covar_string='r_0: COORDINATION GROUPA=coms1 GROUPB=coms2 NN=6 MM=12 D_0=0.0 R_0=0.1'):
        '''
        This function will create a dictionary given a covar string.
        INPUTS:
            covar_string: [str]
                covar string
        OUTPUTS:
            output_dict: [dict]
                dictionary with each part
        '''
        
        split_lines = covar_string.split(' ')
        ''' RETURNS:
            ['r_0:',
             'COORDINATION',
             'GROUPA=coms1',
             'GROUPB=coms2',
             'NN=6',
             'MM=12',
             'D_0=0.0',
             'R_0=0.1']
        '''
        
        ## DEFINING OUTPUT DICT
        output_dict = {
                'label': split_lines[0][:-1], # Removing the ":"
                'keyword': split_lines[1],
                }
        
        ## DEFINING LINES THAT ARE NEXT (AFTER KEYWORD)
        next_lines = split_lines[2:]
                    
        
        ## LOOPING THROUGH
        for each_line in next_lines:
            ## REPLACING ANY '{' OR '}'
            if '{' in each_line or '}' in each_line:
                each_line = each_line.replace("{", "")
                each_line = each_line.replace("}", "")

            split_by_equal = each_line.split("=")
            output_dict[split_by_equal[0]] = split_by_equal[1]
        
        return output_dict
    
    ### FUNCTION TO SEARCH FOR PRINT STRIDE
    def extract_print_details(self):
        '''
        This function looks for the print stride, e.g.
            PRINT ...
                STRIDE=100
                ARG=*
                FILE=COVAR.dat
            ... PRINT
        OUTPUT:
            output_dict: [dict]
                dictionary with the output details of each argument, e.g.
                    {'STRIDE': '100', 'ARG': '*', 'FILE': 'COVAR.dat'}
        '''
        ## FINDING IDX WITH PRINT
        idx_with_print = [idx for idx, each_line in enumerate(self.lines) if 'PRINT' in each_line]
        
        ## GETTING LINES
        if len(idx_with_print) > 1:
            line_index = np.arange( idx_with_print[0], idx_with_print[-1])
        elif len(idx_with_print) == 1:
            line_index = [idx_with_print[0]]
        else:
            print("Warning! No PRINT command found in the input file")
            
        ## GETTING THE LINES
        lines_with_print = [self.lines[each_line] for each_line in line_index ]
        
        ## GETTING LINES WITH ARGUMENTS
        lines_with_arguments = [each_line for each_line in lines_with_print if '=' in each_line ]
        
        ## GETTING DICTIONARY
        output_dict = {}
        
        ## LOOPING THROUGH EACH LINE WITH ARGS AND OUTPUTTING
        for each_line in lines_with_arguments:
            ## REMOVE ALL WHITE SPACES
            removed_spaces = each_line.replace(" ", "")
            split_args = removed_spaces.split("=")
            ## STORING
            output_dict[split_args[0]] = split_args[1]
        
        return output_dict

### fUNCTION TO EXTRACT COVAR DETAILS
def extract_plumed_input_and_covar(path_to_sim,
                                   plumed_input_file="plumed.dat",
                                   time_stride=100 ):
    '''
    This function extracts the plumed input file, and reads the COVAR file. 
    This is useful for subsequent plotting.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        plumed_input_file: [str]
            plumed input file name                
        time_stride: [int]
            stide in your simulation output trajectory. This will be multiplied 
            with time to get the correct time.
    OUTPUTS:
        plumed_input: [obj]
            plumed input object class
        df_extracted: [dataframe]
            dataframe output from the input file, e.g.
                  label       keyword GROUPA GROUPB NN  MM  D_0   R_0
                0   r_0  COORDINATION  coms1  coms2  6  12  0.0  0.05
                1   r_1  COORDINATION  coms1  coms2  6  12  0.0  0.10
        covar_output: [dataframe]
            dataframe containing all CV's with their values across time, e.g.
                      time       r_0       r_1  ...          r_7          r_8          r_9
                0      0.0  0.000023  0.021002  ...   158.773418   306.397751   545.617277
                1  10000.0  0.000017  0.026273  ...   198.003159   377.444710   663.202755
    '''
    ## DEFINING PATH TO PLUMED INPUT FILE
    path_to_plumed_input = os.path.join(path_to_sim,
                                        plumed_input_file)
    
    
    ## READING PLUMED FILE
    plumed_input = read_plumed_input(path_plumed_input = path_to_plumed_input)
    
    ## GETTING PRINT DETAILS
    print_details = plumed_input.extract_print_details()
    
    ## GETTING COVAR FILE
    covar_file = print_details['FILE']
    
    ## PATH TO COVAR AND PLUEMD
    path_to_covar = os.path.join(path_to_sim,
                                 covar_file)
    
    ## READING COVER
    covar_output = read_covar(path_to_covar)
    
    ## DEFINING LIST OF COLUMNES
    cols_covar = list(covar_output.columns[1:]) # Ignoring time column        
    
    ## FINDING LINES
    lines_with_covar = plumed_input.list_lines_with_covar(covar_list = cols_covar)
    
    ## GETTING DICT
    extracted_lines = [ plumed_input.create_dict_given_covar_string(each_line) for each_line in lines_with_covar ]
    
    ## FINDING DATAFRAME
    df_extracted = pd.DataFrame(extracted_lines)
    
    ## MULTIPLYING TIME
    covar_output['time'] = covar_output['time'].apply(lambda x: x*float(time_stride))
    
    return plumed_input, df_extracted, covar_output


### FUCNTION TO PLOT COORD VERSUS R_0
def plot_coord_vs_r_0(covar_output,
                      df_extracted,
                      current_time = 100000.0,
                      distance_below_cutoff = None):
    '''
    The purpoes of this function is function is to plot the coordination 
    number versus r_0 given the plumed inputs
    INPUTS:
        covar_output: [df]
            dataframe containing covar.data
        df_extracted: [df]
            extracted dataframe from the plumed input file
        current_time: [float]
            time to print, default is 100000 ps
        distance_below_cutoff: [float]
            total contacts below the cutoff
    OUTPUTS:
        figure and axis
    '''

    ## GETTING ROW OF CURRENT TIEM
    covar_row = covar_output.loc[covar_output['time'] == current_time]
    
    ## GETTING THE R_0
    r_0_value = np.array(df_extracted['R_0'])
    coord_values = covar_row[df_extracted['label'].to_list()].values[0]
    
    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_xlabel("R_0 (nm)")
    ax.set_ylabel("Total coordination number")
    
    ## PLOTTING
    ax.plot(r_0_value, coord_values,linestyle = "-", marker = '.',  color = 'k')
    
    ## DRAWING LINE
    if distance_below_cutoff is not None:
        ax.axhline(y=distance_below_cutoff, color='b', linestyle='--', label="Total contacts")
    
    ## ADDING LEGEND
    ax.legend()
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT COORD VERSUS DMAX
def plot_coord_vs_dmax(covar_output,
                      df_extracted,
                      current_time = 100000.0,):
    '''
    This function varys dmax and generates a plot for it.
    INPUTS:
        covar_output: [df]
            dataframe containing covar.data
        df_extracted: [df]
            extracted dataframe from the plumed input file
        current_time: [float]
            time to print, default is 100000 ps
    '''
    
    ## SEEING IF THERE IS A REFERENCE
    ref_available = df_extracted['label'].str.contains('ref').sum() > 0
    if ref_available:
        ref_values = df_extracted.loc[df_extracted['label'].str.contains('ref')]
        ## GETTING INDEXES
        ref_indexes = ref_values.index.to_list()
    else:
        ref_values = None
        ref_indexes = []
    
    ## DROPPING ROWS
    df = df_extracted.drop(ref_indexes)
    covar_df = covar_output.drop(columns=['ref'])
    
    ## DEFINING TIME
    current_time = nplm_distances.traj_time[0]
    
    ## GETTING ROW OF CURRENT TIEM
    covar_row = covar_df.loc[covar_df['time'] == current_time]
    
    ## DEFINING X AND Y 
    x = np.array(df['D_MAX']).astype('float')
    y = covar_row[df['label'].to_list()].values[0]
    
    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_xlabel("D_MAX (nm)")
    ax.set_ylabel("Total coordination number")
    
    ## PLOTTING
    ax.plot(x,y, linestyle='-', marker='.', color='k')
    
    ## PLOTTING REFERENCE LINE
    if ref_values is not None:
        y_line_ref = covar_output.loc[covar_output['time'] == current_time]['ref'].values
        ax.axhline(y=y_line_ref, color='b', linestyle='--', label="No dmax set")
                
    ## ADDING LEGEND
    ax.legend()
    
    ## TIGHT LAYOUT
    fig.tight_layout()    
    
    return fig, ax


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    
    ## DEFINING PARENT SIM
    parent_dir="20200430-debugging_nplm_plumed_ROT012_neighbor_list"
    specific_sim = "NPLMplumedcontactspulling-5.100_2_25_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    parent_dir = r"20200430-debugging_nplm_plumed_ROT012_neighbor_list_pt2"
    specific_sim = r"NPLMplumedcontactspulling-5.100_2_25_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    parent_dir = r"20200505-debugging_nplm_plumed_ROT001_neighbor_list_pt3"
    specific_sim = r"NPLMplumedcontactspulling-5.100_2_50_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"

    parent_dir = r"20200505-full_pulling_plumed"
    specific_sim = r"NPLMplumedcontactspulling-5.100_2_50_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"   
    
    ## US FORWARD ROT012
    parent_dir = r"20200120-US-sims_NPLM_rerun_stampede"
    specific_sim = r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"  
    
    ## US FORWARD ROT001
#    parent_dir="20200427-From_Stampede"
#    specific_sim="US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"
    
    relative_sim_path =r"4_simulations/1.300"
    # r"4_simulations/2.900"
    # r"4_simulations/1.300"
    


    
    ## DEFINING SIMULATION PATH
    path_to_sim = os.path.join(PARENT_SIM_PATH,
                               parent_dir,
                               specific_sim,
                               relative_sim_path)
    
    #%%
    
    ## COMPUTING DISTANCES    
    frame = -1
    nplm_distances, distances = main_compute_nplm_distances(path_to_sim = path_to_sim,
                                                            frames = [frame])
    
    #%%
    
    # Working directory:
    # /home/akchew/scratch/nanoparticle_project/nplm_sims/20200120-US-sims_NPLM_rerun_stampede/US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/4_simulations/1.300
    
    ## DEFINING DISTANCE
    distance = distances[0]
    
    ## COMPUTING ALL DISTANCES LESS THAN 0.5
    distance_below_cutoff = np.sum(distance < 0.5)
    
    ## SUMMING
    print(distance_below_cutoff)


    #%%
    
    ## DEFINING SIM DICT
    sim_dict={
            'US_R12_z1.300':
                {
                        'parent_dir': r"20200120-US-sims_NPLM_rerun_stampede",
                        'specific_sim': r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"  ,
                        'relative_sim_path': r"4_simulations/1.300",
                        },
            'US_R12_z1.900':
                {
                        'parent_dir': r"20200120-US-sims_NPLM_rerun_stampede",
                        'specific_sim': r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"  ,
                        'relative_sim_path': r"4_simulations/1.900",
                        },
            'US_R12_z2.900':
                {
                        'parent_dir': r"20200120-US-sims_NPLM_rerun_stampede",
                        'specific_sim': r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"  ,
                        'relative_sim_path': r"4_simulations/2.900",
                        },
            
            }
                
    ## DEFINING PLUMED INPUT FILE
    plumed_input_file = "plumed_vary_D_MAX.dat"
    # plumed_input_file = "plumed_vary_R_0.dat"
    plumed_input_file = "plumed_vary_INDEX.dat"
    frame = -1
    
    if plumed_input_file == "plumed_vary_INDEX.dat":
        ## CHANGING SIM DICT
        sim_dict={
                'US_R12_z5.100':
                    {
                            'parent_dir': r"20200120-US-sims_NPLM_rerun_stampede",
                            'specific_sim': r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"  ,
                            'relative_sim_path': r"4_simulations/5.100",
                            },
                            }
        
                
    ## LOOPING THROUGH EACH
    for current_key in sim_dict:
        ## GETTING ALL PATHS
        current_dict = sim_dict[current_key]
        parent_dir = current_dict['parent_dir']
        specific_sim = current_dict['specific_sim']
        relative_sim_path = current_dict['relative_sim_path']
        
        ## DEFINING SIMULATION PATH
        path_to_sim = os.path.join(PARENT_SIM_PATH,
                                   parent_dir,
                                   specific_sim,
                                   relative_sim_path)
    
        ## COMPUTING DISTANCES
        if plumed_input_file != "plumed_vary_INDEX.dat":
            nplm_distances, distances = main_compute_nplm_distances(path_to_sim = path_to_sim,
                                                                    frames = [frame])
    
            ## DEFINING DISTANCE
            distance = distances[0]
            
            ## COMPUTING ALL DISTANCES LESS THAN 0.5
            distance_below_cutoff = np.sum(distance < 0.5)
            
            ## SUMMING
            print(distance_below_cutoff)
                
            ## DEFINING TIME
            current_time = nplm_distances.traj_time[0]
            
        ## EXTRACTING PLUMED INPUT FILE
        plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim,
                                                                                  plumed_input_file=plumed_input_file,
                                                                                  time_stride=100 )
        
        ## DEFINING SUFFIX
        fig_suffix = specific_sim + "-" +  relative_sim_path.split("/")[-1]
        if plumed_input_file != "plumed_vary_INDEX.dat":
            fig_suffix += "-%d"%(current_time)
        
        ## VARYING R_0
        if plumed_input_file == "plumed_vary_R_0.dat":

            ### VARYING COORD VS R_0
            fig, ax = plot_coord_vs_r_0(covar_output,
                                        df_extracted,
                                        current_time = current_time,
                                        distance_below_cutoff = distance_below_cutoff)    
            ## FIGURE NAME
            figure_name = "rmax_optimization-" + fig_suffix
            
            ## SETTING AXIS
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(IMAGE_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)
        ## GETTING D MAX
        elif plumed_input_file == "plumed_vary_D_MAX.dat":

            ## PLOTTING COORD VERSUS D MAX    
            fig, ax = plot_coord_vs_dmax(covar_output,
                                         df_extracted,
                                         current_time = current_time)
            
            ## FIGURE NAME
            figure_name ="dmax_optimization-" + fig_suffix
            
            ## SETTING AXIS
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(IMAGE_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)
        elif plumed_input_file == "plumed_vary_INDEX.dat":
            ## CREATING FIGURE
            figsize=plot_funcs.cm2inch( *FIGURE_SIZE )
            ax = covar_output.plot(x='time', figsize = figsize)
            
            ## ADDING Y LABEL
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Coordination number")
            
            ## GETTING FIG
            fig = plt.gcf()
            
            ## TIGHT LAYOUT
            fig.tight_layout()
            
            ## FIGURE NAME
            figure_name = "plumed_vary_index-" + fig_suffix
            
            ## SETTING AXIS
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(IMAGE_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)
            
            ## GETTING TIME
            
            
    
    #%% VARYING R_0 


    
    
    #%%
    
#    ## DEFINING FIGURE SIZE
#    FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
#    SAVE_FIG = False
#    # False
#    ## PLOTTING SIJ VS DISTANCE
#    def plot_sij_vs_distance(sij, distance, r0=0.5):
#        ''' Plots sij vs. distance '''
#        ## CREATING FIGURE
#        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
#        
#        ## ADDING LABELS
#        ax.set_xlabel("$r_{ij}$ (nm)")
#        ax.set_ylabel("$s_{ij}$")
#        
#        ## DEFINING X AND Y
#        x = distance
#        y = sij
#        
#        ## PLOTTING
#        ax.plot(x, y,linestyle = "None", marker = '.',  color = 'k')
#        
#        ## DRAWING LINE
#        ax.axvline(x=r0, color='b', linestyle='--')
#        
#        ## GETTING TIGHT LAYOUT
#        fig.tight_layout()
#        return fig, ax
#    ## PLOTTING
#    fig, ax = plot_sij_vs_distance(sij = sij, 
#                                   distance =distance, 
#                                   r0=dmax)
    