# -*- coding: utf-8 -*-
"""
compute_NPLM_contacts_extract.py
The purpose of this script is to compute the number of contacts between 
nanoparticle-lipid membrane systems. 

Written by: Alex K. Chew (01/27/2020)

## DEFINING METHOD TO COMPUTE SUMMARY
gmx distance -s nplm_push.tpr -f nplm_push.xtc -oxyz push_summary.xvg -select 'com of group DOPC plus com of group AUNP' -n push.ndx

"""
## LOADING
import os
import glob
import matplotlib.pyplot as plt
from MDDescriptors.core.pickle_tools import load_pickle_results, save_and_load_pickle
import MDDescriptors.core.plot_tools as plot_funcs
import numpy as np
import pandas as pd
    
## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools
from MDDescriptors.core.import_tools import read_file_as_line
from MDDescriptors.core.read_write_tools import read_xvg

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## IMPORTING EXTRACT TRAJ FUNCTION
from MDDescriptors.traj_tools.loop_traj_extract import load_results_pickle, ANALYSIS_FOLDER, RESULTS_PICKLE

## IMPORTING FUNCTION
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import main_compute_contacts

## IMPORTING FUNCTION
from MDDescriptors.application.np_lipid_bilayer.compute_com_distances import main_compute_com_distances

## PICKLING RESULTS
from MDDescriptors.core.pickle_tools import pickle_results

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

##########################
#### GLOBAL VARIABLES ####
##########################

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
SAVE_FIG = False
# False

ANALYSIS_FOLDER = ANALYSIS_FOLDER
# ANALYSIS_FOLDER = "analysis_part2"

## DEFINING COLOR FOR HEAD AND TAIL GROUPS
COLOR_LM_GROUPS={
        'HEADGRPS': 'red',
        'TAILGRPS': 'blue',
        'TAIL_GRPS': 'blue'
        }

### FUNCTION TO CHANGE EACH GROUP TYPE
def convert_atomnames_to_groups(array,
                                dict_conversion,
                                check_array = True):
    '''
    The purpose of this function is to convert array using some dictionary 
    conversion method
    INPUTS:
        array: [np.array]
            array of strings that you want to convert
        dict_conversion: [dict]
            dictionary containing list of keys with each key containing 
            a list of strings
        check_array: [logical, default = True]
            True if you want to check array to see if the converted array 
            has any same names as the original array. This is a useful 
            check to ensure that you are not missing any conversions.
    OUTPUTS:
        converted_array: [np.array]
            converted array according to the dictionary
    '''
    ## CREATING COPY
    converted_array = array.copy()
    
    ## LOOPING THROUGH EACH KEY
    for each_key in dict_conversion:        
        ## GETTING EACH GROUP
        indices_for_group = np.isin(array, dict_conversion[each_key])
        ## CHANGING GROUP NAMES
        converted_array[indices_for_group] = each_key

    ## CHECKING IF THE CONVERTED ARRAY HAS ANY MATCHES
    if check_array is True:
        matches = np.equal(converted_array, array)
        any_matches = np.any(matches)
        if any_matches is True:
            print("Warning! Some of your names are not converted")
            print("Names that are not converted are listed below:")
            print(array[matches])
            print("Check your dictionary")
    
    return converted_array

### FUNCTION TO GET THE ATOM INDICES GIVEN THE GROUPS AND INDICES
def get_atom_indices_from_atomnames(traj,
                                    atom_indices,
                                    atom_names,
                                    ):
    '''
    This function gets the atom indices that matches the atom indices specified. 
    INPUTS:
        traj: [obj]
            trajectory object
        atom_indices: [list]
            list of atom indices that you would like to check
        atom_names: [list]
            list of atom names you are interested in
    OUTPUTS:
        indices: [list]
            list of atom indices that match your atom name
    '''
    indices = []
    
    ## LOOPING THROUGH INDICES
    for each_index in atom_indices:
        ## GETTING ATOM NAMES
        current_atom_name = traj.topology.atom(each_index).name
        if current_atom_name in atom_names:
            ## GETTING ATOM INDEX
            current_atom_idx = traj.topology.atom(each_index).index
            indices.append(current_atom_idx)
    ## CONVERTING INDICES TO NP ARRAY
    indices = np.array(indices)
    return indices

### FUNCTION TO CONVERT GROUP KEYS INTO ATOM INDICES
def convert_group_keys_to_atomindex(traj,
                                    groups,
                                    atom_indices,
                                    ):
    '''
    The purpose of this function is to convert a dictionary with group 
    keys into a dictionary with the atom indices. 
    INPUTS:
        traj: [obj]
            trajectory object
        groups: [dict]
            dictionary with atom names
        atom_indices: [list]
            list of atom indices that you would like to check
            
    OUTPUTS:
        atom_index_dict: [dict]
            dicitionary of atom indices for each group
    '''
    ## CREATING EMPTY DICT
    atom_index_dict = {}

    ## LOOOPING THROUGH EACH KEY
    for each_key in groups:
        ## DEFINING ATOM NAMES
        current_atom_names = groups[each_key] 
        ### FINDING ATOM INDICES
        indices = get_atom_indices_from_atomnames(traj = traj,
                                                  atom_indices = atom_indices,
                                                  atom_names = current_atom_names)
        ## STORING
        atom_index_dict[each_key] = indices
        
    return atom_index_dict

### FUNCTION TO GENERATE NANOPARTICLE GROUPS
def generate_rotello_np_groups(
                               np_heavy_atom_index,
                               atom_names_np = None,
                               traj = None,
                               verbose = True,
                               want_atom_index = False,
                               combine_alkane_and_R_group=False,
                               combine_N_and_R_group= False):
    '''
    The purpose of this function is to generate Rotello NP groups based on
    the structure. 
    INPUTS:
        traj: [obj]
            trajectory object
        np_heavy_atom_index: [np.array]
            heavy atom index
        atom_names_np: [list]
            list of atom names
        want_atom_index: [logical]
            True if you want atom indices as a return
        combine_alkane_and_R_group: [logical]
            True if you want to combine alkane and R group
        combine_N_and_R_group: [logical]
            True if you want to combine N and R groups
    OUTPUTS:
        np_groups: [dict]
            dictionary of different groups
            Labels are here:
                GOLD: all gold atoms
                ALK: all alkane groups
                PEG: 4 peg chains
                R_GRP: R group
        np_groups_indices: [dict]
            dictionary of nanoparticle group atom indices
    TO GET GROUPS:
        VMD: open MDligand structure
        Run: show_atom_labels_atomname "residue R12", where residue is your res name
        Then, use the atom labels to decide which parts are which
    '''
    ## DEFINING GROUPS
    np_groups = {
            'GOLD': ['Au'],
            'ALK': ['S1', 'C2', 'C5', 'C8', 'C11', 'C14', 'C17', 'C20', 'C23', 'C26', 'C29', 'C32'],
            'PEG': [ 'O35', 'C36', 'C39', 'O42', 'C43', 'C46', 'O49', 'C50', 'C53', 'O56', 'C57', 'C60' ],
            'NGRP': [ 'N63', 'C68', 'C64' ]
            }
    
    ## GETTING ALL UNIQUE ATOM NAME
    if atom_names_np is None:
        atom_names_np = np.array([ traj.topology.atom(each_atom).name  for each_atom in np_heavy_atom_index])
    np_unique_atom_names = np.unique(atom_names_np)
    
    ## GETTING ALL DEFINED ATOM NAMES
    defined_atom_names = calc_tools.flatten_list_of_list([np_groups[each_key] for each_key in np_groups])
    
    ## GETTING DIFFERENCES
    diff_atom_names = np.setdiff1d(np_unique_atom_names, defined_atom_names)

    ## GETTING ATOM INDICES THAT ARE NOT WITHIN GROUPS
    np_groups['RGRP'] = list(diff_atom_names[:])
    if verbose is True:
        print("Setting R group to: %s"%( ', '.join(list(diff_atom_names)) ) )
        print("Total R group atoms: %d"%(len(diff_atom_names) ) )
    
    ## COMBINING R AND ALKANE GROUP
    if combine_alkane_and_R_group is True:
        np_groups['ALK_RGRP'] = np_groups['ALK'] + np_groups['RGRP']
        print("Combining alkane and R groups to: ALK_RGRP")
    
    ## COMBINING N AND R GROUPS
    if combine_N_and_R_group is True:
        np_groups['NGRP_RGRP'] = np_groups['NGRP'] + np_groups['RGRP']
        print("Combining N and R groups to: NGRP_RGRP")
        
    ## PRINTING ATOM INDEX
    if want_atom_index is True:
        np_groups_indices = convert_group_keys_to_atomindex(traj = traj,
                                                            groups = np_groups,
                                                            atom_indices = np_heavy_atom_index,
                                                            )
        return np_groups, np_groups_indices
    else:
        return np_groups



### GENERATING GROUPS FOR DOPC
def generate_lm_groups(
                       lm_heavy_atom_index,
                       atom_names = None,
                       traj = None,
                       verbose = True,
                       want_atom_index = False
                       ):
    '''
    The purpose of this function is to select the lipid membrane groups. 
    INPUTS:
        traj: [obj]
            trajectory object
        lm_heavy_atom_index: [np.array]
            heavy atom index
        atom_names: [list]
            list of atom names for lipid membranes
        want_atom_index: [logical]
            True if you want atom indices as a return
    OUTPUTS:
        lm_groups: [dict]
            lipid membrane group atom names
        lm_groups_indices: [dict]
            dictionary with lipid membrane indices
    '''
    # on vmd: show_atom_labels_atomname "resid 161 and not name \"H.*\""
    lm_groups = {
            'HEADGRPS': [
                    "P", "O12", "O13", "O14",
                    "C11", "C12",
                    "N", "C13", "C15", "C14"
                    ]
            }
    ## GETTING ALL UNIQUE ATOM NAME
    if atom_names is None:
        atom_names = np.array([ traj.topology.atom(each_atom).name  for each_atom in lm_heavy_atom_index])
    unique_atom_names = np.unique(atom_names)
    
    ## GETTING ALL DEFINED ATOM NAMES
    defined_atom_names = calc_tools.flatten_list_of_list([lm_groups[each_key] for each_key in lm_groups])
    
    ## GETTING DIFFERENCES
    diff_atom_names = np.setdiff1d(unique_atom_names, defined_atom_names)
    
    ## GETTING ATOM INDICES THAT ARE NOT WITHIN GROUPS
    lm_groups['TAILGRPS'] = list(diff_atom_names[:])
    if verbose is True:
        print("Setting tail groups group to: %s"%( ', '.join(list(diff_atom_names)) ) )
        print("Total tail groups group atoms: %d"%(len(diff_atom_names) ) )
    
    ## PRINTING ATOM INDEX
    if want_atom_index is True:
        lm_groups_indices = convert_group_keys_to_atomindex(traj = traj,
                                                            groups = lm_groups,
                                                            atom_indices = lm_heavy_atom_index,
                                                            )
        return lm_groups, lm_groups_indices
    else:
        return lm_groups

### FUNCTION TO GET PERMUTATIONS
def get_permutation_groups(list1, list2):
    '''
    The purpose of this function is to get the permutation of two groups.
    INPUTS:
        list1: [list]
            list one
        list2: [list]
            list two
    OUTPUTS:
        list of all permutations
    '''
    import itertools
    perm_list = [ r[0] + '-' + r[1] for r in itertools.product(list1, list2) ]
    return perm_list


### FUNCTION TO GET CONTACTS
def compute_contacts_for_groups(
                                contact_results,
                                class_results,
                                np_groups,
                                lm_groups,
                                verbose = True,
                                traj = None,
                                np_atom_names = None,
                                lm_atom_names = None,):
    '''
    The purpose of this function is to compute contacts for specific groups. 
    This will loop thorugh all the frames and basically count the pairs 
    that are in contact.
    INPUTS:
        traj: [obj]
            trajectory object (any number of frames). We will simply use the trajectory 
            to compute number of contacts quickly.
        contact_results: [list]
            list of atoms that are in contact per frame
        class_results: [obj]
            class object result
        np_groups: [dict]
            dictionary of nanoparticle groups with their atom names
        lm_groups: [dict]
            dictionary of lipid membrane groups with their atom names
        verbose: [logical]
            True if you want to print contacts calculations
        np_atom_names: [list]
            list of atom names
        lm_atom_names: [list]
            list of lipid membrane atom names
    OUTPUTS:
        contacts_storage: [np.array, shape = (num_permutation, num_frames)]
            contacts that are stored across time per permuation example
        permutation_list: [list]
            list of permutation groups
    '''
    ## GETTING PERMUTATION LIST
    permutation_list = get_permutation_groups(list1 = np_groups,
                                              list2 = lm_groups)
    
    ## GETTING THE EMPTY ARRAY
    contacts_storage = np.zeros( (len(permutation_list), len(contact_results) ) )
    
    ## DEFINING DATA TYPE    
    dt = np.dtype('object')
    
    ## DEFINING STRING TYPE
    string_type = "U25"
    
    ## GETTING THE ATOM NAMES
    if np_atom_names is None:
        np_atom_names = [ traj.topology.atom(each_idx).name for each_idx in class_results.np_heavy_atom_index ]

    if lm_atom_names is None:
        lm_atom_names = [ traj.topology.atom(each_idx).name for each_idx in class_results.lm_heavy_atom_index ]
        
    ## CONVERTING TO NUMPY ARRAY STRING ( LONG )    
    np_atom_names = np.array(np_atom_names,dtype = dt)
    lm_atom_names = np.array(lm_atom_names, dtype = dt)
    
    ## DEFINING FRAME
    # LOOPING THROUGH EACH FRAME
    for frame in range(len(contact_results)):
        ## PRINTING
        if frame % 100 == 0:
            print("Working on frame: %d of %d"%(frame, len(contact_results) ) )
        
        ## DEFINING ATOM PAIRS
        pairs_in_contact = contact_results[frame]
        
        ## CONVERTING ATOM PAIRS TO ATOM NAMES
        np_current_atoms_in_contact = np_atom_names[pairs_in_contact[:,0]]
        lm_current_atoms_in_contact = lm_atom_names[pairs_in_contact[:,1]]

        ## CONVERTING ATOM NAMES TO GROUPS
        np_converted_groups = convert_atomnames_to_groups(array = np_current_atoms_in_contact,
                                                          dict_conversion = np_groups).astype(string_type)
        lm_converted_groups = convert_atomnames_to_groups(array = lm_current_atoms_in_contact,
                                                          dict_conversion = lm_groups).astype(string_type)
        
        ## GETTING ARRAY LIST
        combined_array_names = np.core.defchararray.add(np_converted_groups, '-', )
        combined_array_names = np.core.defchararray.add(combined_array_names, lm_converted_groups )
        
        ## GETTING COUNTS
        unique, counts = np.unique(combined_array_names, return_counts=True)
        
        ## UPDATING STORAGE
        for uniq_idx, each_key in enumerate(unique):
            ## GETTING LIST INDEX
            row_idx = permutation_list.index(each_key)
            contacts_storage[row_idx, frame] = counts[uniq_idx]
    
    return contacts_storage, permutation_list

### FUNCTION TO COMPUTE CONTACTS MAIN
def main_compute_contacts_for_groups(
                                     path_class_pickle = None,
                                     path_log = None,
                                     path_analysis = None,
                                     class_object = None,
                                     contact_results = None,
                                     index = 0,
                                     path_to_traj = None,
                                     gro_file = None,
                                     xtc_file = None,):
    '''
    The purpose of this function is to compute contacts for group -- this will load 
    the trajectory and so forth as a main code to output contacts storage and 
    permutation list. 
    INPUTS:
        class_object: [obj]
            class object containing contact information
        contact_results: [list]
            list containing contact results
        path_to_traj: [str]
            path to the trajectory
        gro_file: [str]
            gro file
        xtc_file: [str]
            xtc file
        path_class_pickle: [str]
            path to the class pickle
        path_log: [str]
            path to the log file
        path_analysis: [str]
            path to analysis folder
        index: [int]
            index to load the gro / xtc, which is used to get atom indexes
    OUTPUTS:
        output_dict: [dict]
            output dictionary for computing number of contacts
    '''
    ## LOAD TRAJECTORY IF NO INFORMATION ABOOUT THE OBJECT IS AVAILABLE
    if class_object is None:
        ## LOADING SINGLE FRAME
        traj_data = import_tools.import_traj(directory = path_to_traj,
                                             structure_file = gro_file,
                                             xtc_file = xtc_file,
                                             index = index)
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        ## SETTING ATOM NAMES TO NONE
        atom_names_np = None
        atom_names_lm = None
        
        ## LOADING CLASS OBJECT
        class_results = load_pickle_results(file_path = path_class_pickle,
                                             verbose = True)[0][0]
        
        
    else:
        traj = None
        atom_names_np = class_object.np_heavy_atom_names
        atom_names_lm = class_object.lm_heavy_atom_names
        ## REDEFINING CLASS RESULTS
        class_results = class_object
    
    ## SEEING IF CONTACT RESULTS ARE AVAILABLE
    if contact_results is None:

        ## LOADING CONTACT RESULTS
        contact_results = read_pickle_based_on_log_file(path_log = path_log, 
                                                        path_analysis = path_analysis)[0][0]
    
    ## GETTING GROUPS
    np_groups = generate_rotello_np_groups(traj = traj,
                                           atom_names_np = atom_names_np,
                                           np_heavy_atom_index = class_results.np_heavy_atom_index )
    
    ## GETTING LM GROUPS
    lm_groups = generate_lm_groups(traj = traj,
                                   atom_names = atom_names_lm,
                                   lm_heavy_atom_index = class_results.lm_heavy_atom_index,
                                   verbose = True,
                                   )
    
    ## DEFINING INPUTS
    group_contacts_inputs = {
            'np_atom_names': atom_names_np,
            'lm_atom_names': atom_names_lm,
            'traj': traj,
            'contact_results': contact_results,
            'class_results': class_results,
            'np_groups': np_groups,
            'lm_groups': lm_groups,
            }
    
    ## RUNNING THE FUNCTION
    pickle_name = compute_contacts_for_groups.__name__
    path_pickle_contacts_permutations = os.path.join(path_analysis, pickle_name)
    
    ## PERFORMING THE RESULTS
    contacts_storage, permutation_list = save_and_load_pickle(function = compute_contacts_for_groups, 
                                                               inputs = group_contacts_inputs, 
                                                               pickle_path = path_pickle_contacts_permutations,
                                                               rewrite = True)
    
    ## DEFINING OUTPUT
    output_dict = {
            'contacts_storage': contacts_storage,
            'permutation_list': permutation_list,
            'np_groups' : np_groups,
            'lm_groups': lm_groups,
            }
    
    return output_dict

### FUNCTION TO PLOT
def plot_nplm_contacts_vs_time(contacts_storage,
                               permutation_list,
                               np_groups,
                               time_array = None,
                               frame_rate = 10):
    '''
    The purpose of this function is to plot nplm contacts as a function of 
    time. 
    INPUTS:
        contacts_storage: [np.array]
            storage for contacts
        permutation_list: [list]
            list of permutations which is directly linked to contacts_storage
        np_groups: [dict]
            dictionary of nanoparticle groups
        lm_groups: [dict]
            dictionary of lipid membrane groups
        frame_rate: [int]
            frame rate of contacts
    OUTPUTS:
        fig, ax: 
            figure and axis
    
    '''

    ## DEFINING NANOPARTICLE GROUPS
    np_group_list = list(np_groups.keys())
    
    ## DIVIDING PERMUTATION LIST
    premutation_list_divided = [ each_string.split('-') for each_string in permutation_list ]
    
    ## CREATING FIGURES
    fig, axs = plt.subplots(nrows=len(np_group_list), sharex=True)
    
    ## FIXING FOR WHEN AXS IS ONE
    if len(np_group_list)==1:
        axs = [axs]
    
    ## DEFINING TIME ARRAY
    if time_array is None:
        x_values =  np.arange( len(contact_results )) * frame_rate
    else:
        x_values = time_array[:]
    
    ## LOOPING THROUGH GROUPS
    for idx, each_grp in enumerate(np_group_list):
        
        ## GETTING INDEX
        index_of_grp = [list_idx for list_idx,each_list in enumerate(premutation_list_divided) if each_list[0] == each_grp ]
        ## LOOPING THROUGH EACH INDEX
        for idx_grp in index_of_grp:
            ## DEFINING AXIS
            ax = axs[idx]
            ## DEFINING Y VALUES
            y_values = contacts_storage[idx_grp]
            ## DEFINING CURRENT NAME FOR LM
            label = premutation_list_divided[idx_grp][-1]
            ## DEFINING COLOR
            color = COLOR_LM_GROUPS[label]
            
            ## ADDING TITLE
            ax.text(.5,.8,each_grp,
                horizontalalignment='center',
                transform=ax.transAxes)
            ## PLOTTING
            ax.plot(x_values, 
                    y_values, 
                    linestyle = '-', 
                    linewidth=2, 
                    color = color, 
                    label = label)
            ## SETTING Y LABELS
            ax.set_ylabel("# contacts")
    ## ADJUSTING SPACE OF SUB PLOTS
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ## ADDING AXIS LABELS
    axs[len(np_group_list)-1].set_xlabel("Time (ps)")
    axs[-1].legend(loc='upper right')
    
    return fig, ax

### FUNCTION TO READ PICKLE FROM LOG FILE
def read_pickle_based_on_log_file(path_log, path_analysis):
    '''
    This function reads the pickle file using the log output
    INPUTS:
        path_log: [str]
            path to log file
        path_analysis: [str]
            path to analysis file
    OUTPUTS:
        results: [obj]
            results object
    '''

    ## GETTING PICKLE FILE FROM LOG FILE
    output_log = read_file_as_line(path_log)
    
    ## DEFINING PICKLE NAME
    pickle_file = str(output_log[0].split(", ")[-1])

    ## DEFINING FULL PATH
    path_pickle = os.path.join(path_analysis,
                               pickle_file)
    ## LOADING PICKLE
    results = load_pickle_results(file_path = path_pickle,
                                         verbose = True)
    return results
        
#### FUNCTION TO GET TIME ARRAY AND Z DISTANCE
def get_time_array_and_z_dist_from_com_distances(path_to_sim):
    '''
    The purpose of this function is to get the COM distances in the z-dimension.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
    OUTPUTS:
        time_array: [np.array]
            time array in ps
        z_dist: [np.array]
            center-of-mass z distance
    '''
    
    ## LOADING DATA FOR COM DISTANCES
    xvg, xvg_data, time_array, z_dist = load_results_pickle(path_to_sim = path_to_sim,
                                                            func = main_compute_com_distances,
                                                            analysis_folder = ANALYSIS_FOLDER)
    
    ## GETTING TIME ARRAY
    time_array = xvg_data[:,0]
    
    ## GETTING DISTANCE
    z_dist = xvg_data[:,-1]
    
    ## COMPUING COM DISTANCE
    # com_distance = np.linalg.norm(xvg_data, axis = 1)
    
    return time_array, z_dist
#
#### FUNCTION TO PLOT NUMBER OF CONTACTS VERSUS TIME
#def plot_num_contacts_vs_time(path_simulations,
#                              config_library,
#                              relative_analysis_dir = "",
#                              distance_xvg = "",
#                              fig_size_cm = FIGURE_SIZE,
#                              frame_rate = 10):
#    '''
#    The purpose of this function is to plot the number of contacts versus time. 
#    INPUTS:
#        path_simulations: [str]
#            path to the simulations
#        config_library: [list]
#            list of libraries to look for
#        relative_analysis_dir: [str]
#            relative path to the analysis directory
#        fig_size_cm: [tuple]
#            figure size
#        distance_xvg: [str]
#            distance xvg that has the COM distances
#        frame_rate: [float]
#            frame rate that you are using for COM distances
#    OUTPUTS:
#        fig, ax: 
#            figure and axis for the number of contacts
#    '''
#    ## CREATING FIGURE
#    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
#    
#    ax.set_xlabel("Time (ps)")
#    ax.set_ylabel("Number of contacts")
#    
#    ## GETTING COLOR MAP
#    cmap = plot_funcs.get_cmap( len(config_library) )
#    
#    ## LOOPING THROUGH CONFIG LIBRARY
#    for idx, specific_config in enumerate(config_library):
#        print("Working on: %s"%(specific_config) )
#        
#        ### NUMBER OF CONTACTS LOADING
#        ## DEFINING PATH TO SIMULATION
#        path_to_sim = os.path.join(path_simulations,
#                                   specific_config)
#        
#        ## LOADING NUM CONTACTS
#        class_object, contact_results = load_num_contacts(path_to_sim)
#        
#        ## DEFINING TIME ARRAY
#        time_array = class_object.traj_time
#        
#        ## GETTING NUMBER OF CONTACTS
#        num_contacts_array = np.array([len(each_array) for each_array in contact_results])
#
#        ## GETTING COLOR
#        current_group_color = cmap(idx)
#        ## ADDING TO PLOT
#        ax.plot(time_array, num_contacts_array, color=current_group_color, label = specific_config)
#            
#        ## ADDING LEGEND 
#        if len(config_library) == 1:
#        
#            ### COM DISTANCE LOADING
#            time_array, z_dist = get_time_array_and_z_dist_from_com_distances(path_to_sim = path_to_sim)
#    
#            ## ADD SECOND Y AXIS
#            ax2 = ax.twinx() 
#            
#            ## ADDING TO PLOT
#            ax2.plot(time_array, z_dist, ':', color='k', label = "Distance" ) # specific_config + "_dist"
#            ax2.set_ylabel('COM Distance (nm)')
#            
#    ## ADDING LEGEND 
#    if len(config_library) > 1:
#    
#        ## GETTING TICK LABELS
#        x = np.array(config_library).astype('float')
#        normalized = (x-min(x))/(max(x)-min(x))
#        # norm=plt.Normalize(vmin=0, vmax=1)
#        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax = ax,
#                            ticks =  normalized)
#        cbar.ax.set_yticklabels(config_library)
#        
#        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    
#    ## ADDING LEGEND 
#    try:
#        ax2.legend()
#        ax.grid()
#    except Exception:
#        pass
#    ## TIGHT LAYOUT
#    fig.tight_layout()
#    
#    return fig, ax


### FUNCTION TO LOAD CONTACTS
def load_num_contacts(path_to_sim):
    '''
    This function simply loads the number of contacts and cleans if necesssary.
    iNPUTS:
        path_to_sim: [str]
            path to the simulation
    '''
    ## LOADING RESULTS
    class_object, contact_results = load_results_pickle(path_to_sim = path_to_sim,
                                                        func = main_compute_contacts,
                                                        analysis_folder = ANALYSIS_FOLDER)

    ## CLEANING OBJECT DATA
    class_object, contact_results = turn_off_atom_pairs_in_contacts_object(path_to_sim, class_object, contact_results)
    
    return class_object, contact_results

### FUNCTION TO TURN OFF ATOM PAIRS
def turn_off_atom_pairs_in_contacts_object(path_to_sim, class_object, contact_results):
    '''
    This just deletes the atom_pairs, which slows down the loading of 
    pickles.
    '''
    
    ## CHECKING IF ATOM PAIRS ARE DEFINED
    if len(class_object.atom_pairs) > 0:
        class_object.atom_pairs = []
    
        ## REPICKLING
        ## DEFINING PATH TO PICKLE
        path_pickle = os.path.join(path_to_sim, ANALYSIS_FOLDER, main_compute_contacts.__name__,  RESULTS_PICKLE)
        ## STORING THE RESULTS
        pickle_results(results = (class_object, contact_results),
                       pickle_path = path_pickle,
                       verbose = True)

    return class_object, contact_results

### FUNCTION TO TRUNCATE TIME ARRAY
def get_indices_for_truncate_time(time_array, last_time_ps = 0):
    '''
    This function finds the indices that you want for a desired time array. 
    It will truncate the last N ps to use for an analysis
    INPUTS:
        time_array: [np.array]
            time array you want to truncate
        last_time_ps: [float]
            last time you are interested in
    OUTPUTS:
        indices: [np.array]
            indices to run the truncation of time
    '''
    ## SEEING IF YOU HAVE LAST TIME
    if last_time_ps > 0:
        ## NORMALIZE BY FIRST VALUE
        time_array = time_array - time_array[0]
        ## GETTING DIFF
        diff = np.abs(time_array - time_array[-1])
        indices = np.where(diff <= last_time_ps)[0]
    else:
        indices = np.arange(len(time_array))
    
    return indices

### FUNCTION TO PLOT NPLM AVG CONTACTS VERSUS DISTANCE
def plot_nplm_avg_contacts_vs_distances(extract_groups_vs_dist,
                                        ):
    '''
    The purpose of this function is to plot the average number of contacts
    versus distance.
    INPUTS:
        extract_groups_vs_dist: [dict]
            dictionary containing all information for plotting
    OUTPUTS:
        fig, ax:
            figure and axis
    '''
    
    ## DEFINING NANOPARTICLE GROUPS
    np_group_list = list(extract_groups_vs_dist['np_groups'].keys())
    
    permutation_list = extract_groups_vs_dist['permutation_list']
    
    ## DIVIDING PERMUTATION LIST
    premutation_list_divided = [ each_string.split('-') for each_string in permutation_list ]
        
    ## CREATING FIGURES
    fig, axs = plt.subplots(nrows=len(np_group_list), sharex=True)
    
    ## FIXING FOR WHEN AXS IS ONE
    if len(np_group_list)==1:
        axs = [axs]
    
    ## DEFINIGN X VALUES
    x_values = extract_groups_vs_dist['com_z_mean_dist']
    
    ## LOOPING THROUGH GROUPS
    for idx, each_grp in enumerate(np_group_list):
        ## GETTING INDEX
        index_of_grp = [list_idx for list_idx,each_list in enumerate(premutation_list_divided) if each_list[0] == each_grp ]
        
        ## LOOPING THROUGH EACH INDEX
        for idx_grp in index_of_grp:
            ## GETTING PERMUTATION NAME
            permutation_name = permutation_list[idx_grp]
            
            ## DEFINING VALUES
            y_values = extract_groups_vs_dist[permutation_name]
            
            ## DEFINING AXIS
            ax = axs[idx]
            
            ## DEFINING CURRENT NAME FOR LM
            label = premutation_list_divided[idx_grp][-1]
            ## DEFINING COLOR
            color = COLOR_LM_GROUPS[label]
            
            ## ADDING TITLE
            ax.text(.5,.8,each_grp,
                horizontalalignment='center',
                transform=ax.transAxes)
            ## PLOTTING
            ax.scatter(x_values, 
                    y_values, 
                    linestyle = '-', 
                    linewidth=2, 
                    color = color, 
                    label = label)
            ## SETTING Y LABELS
            ax.set_ylabel("Avg. contacts")
            
    ## ADJUSTING SPACE OF SUB PLOTS
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ## ADDING AXIS LABELS
    axs[len(np_group_list)-1].set_xlabel("COM distance (nm)")
    axs[-1].legend(loc='upper right')
    
    return fig, ax

### FUNCTION TO GET DATA RELATIVE TO DISTANCE
def extract_data_for_groups_vs_distance(avg_contacts_storage):
    '''
    This function basically takes the storage of dictionaries and converts 
    them to a dictionary with each permutation group. Output example:
        {'GOLD-HEADGRPS': [0.0],
         'GOLD-TAIL_GRPS': [0.3033932135728543],
         'ALK-HEADGRPS': [12.015968063872256],
         'ALK-TAIL_GRPS': [542.3872255489022],
         'PEG-HEADGRPS': [68.35528942115768],
         'PEG-TAIL_GRPS': [312.8043912175649],
         'NGRP-HEADGRPS': [50.02794411177645],
         'NGRP-TAIL_GRPS': [43.34530938123753],
         'RGRP-HEADGRPS': [17.21556886227545],
         'RGRP-TAIL_GRPS': [16.087824351297407],
         'com_z_mean_dist': 3.894409181636727}
    INPUTS:
        avg_contacts_storage: [list]
            list of average contact information
    OUTPUTS:
        storage_dict: [dict]
            storage dictionary contianing center of mass distance and 
            number of contacts
    '''
    ## CREATING EMPTY DICTIONARY
    storage_dict = { each_key: [] for each_key in avg_contacts_storage[0]['permutation_list'] }
    storage_dict['com_z_mean_dist'] = []
    ## STORING NP GROUPS
    storage_dict['np_groups'] = avg_contacts_storage[0]['np_groups']
    storage_dict['permutation_list'] = avg_contacts_storage[0]['permutation_list']
    
    ## LOOPING AND ADDING
    for idx, current_storage in enumerate(avg_contacts_storage):
        ## STORING EACH PERMUTATION LIST ITEM
        [storage_dict[each_key].append( current_storage['avg_contacts'][each_idx] ) 
                for each_idx, each_key in enumerate(current_storage['permutation_list']) ]
        ## STORING COM DIST
        storage_dict['com_z_mean_dist'].append(current_storage['com_z_mean_dist'])
    return storage_dict

####################################################
### CLASS FUNCTION TO ANALYZE NUMBER OF CONTACTS ###
####################################################
class extract_num_contacts:
    '''
    The purpose of this class is to store all number of contacats code.
    INPUTS:
        job_info: [obj]
            object containing information about the job
        last_time_ps: [float]
            Last time in picoseconds to take from. For example, if this is 
            50,000, it will take the last 50 ns of the trajectory and 
            compute things like contacts and so forth. If this is zero, 
            no truncations will be done
    OUTPUTS:
        
    FUNCTIONS:
        analyze_num_contacts:
            analyzes number of contacts and outputs the details as a dictionary
        plot_num_contacts_vs_time_for_config:
            plots number of contacts versus time
        plot_num_contacts_vs_time_all_config:
            plots contacts versus time for all configuratoins
    '''
    ## INITIALIZING
    def __init__(self,
                 job_info,
                 last_time_ps = 0):
        
        ## STORING JOB INFORMATION
        self.job_info = job_info
        self.last_time_ps = last_time_ps
        
        return
    
    ### FUNCTION TO ANALYZE NUMBER OF CONTACTS
    def analyze_num_contacts(self,
                             path_to_sim,
                             want_nplm_grouping = False,
                             want_com_distance = False,
                             skip_load_contacts = False):
        '''
        This function analyzes the number of contacts for a single input. 
        INPUTS:
            path_to_sim: [str]
                path to the simulation directory.
            want_nplm_grouping: [logical]
                True if you want the number of contacts to then be grouped.
            want_com_distance: [logical]
                True if you want com distances
            skip_load_contacts: [logical]
                True if you want to skip number of contacts
        OUTPUTS:
            output_dict: [dict]
                output dictionary containing the following:
                    class_object:
                        class object containing all information about topology
                    contact_results:
                        list object containing all contact info
                    num_contacts_array: 
                        array containing number of contacts per frame
                    time_array:
                        array containing the time information
                    avg_contacts:
                        array containing average contact information
        '''
        ## SKIPPING THE LOADING OF CONTACTS
        if skip_load_contacts is False:
            ## LOADING NUM CONTACTS
            class_object, contact_results = load_num_contacts(path_to_sim)
        
            ## DEFINING TIME ARRAY
            time_array = class_object.traj_time
        
            ## GETTING INDICES
            indices = get_indices_for_truncate_time(time_array,
                                                last_time_ps = self.last_time_ps)
        
            ## REDEFINING TIME ARRAY (starting at zero)
            time_array = time_array[indices] - time_array[indices][0]
            
            ## REDEFINING CONTACT ARRAY
            contact_results = [contact_results[each_index] for each_index in indices]
            
            ## GETTING NUMBER OF CONTACTS
            num_contacts_array = np.array([len(each_array) for each_array in contact_results])
            
            ## OUTPUT DICT
            output_dict = {
                    'class_object': class_object,
                    'contact_results': contact_results,
                    'num_contacts_array': num_contacts_array,
                    'time_array': time_array,
                    }
        else:
            print("Skipping contacts loading")
            output_dict = {}
        
        ## SEEING IF YOU WANT THE GROUPING
        if want_nplm_grouping is True and skip_load_contacts is False:
            print("Generating NPLM groups...")
            
            ## DEFINING PATH TO ANALYSIS
            path_analysis = os.path.join(path_to_sim,
                                         ANALYSIS_FOLDER,
                                         main_compute_contacts.__name__)
            
            ## DEFINING INPUTS
            input_main_func={
                    'class_object': class_object,
                    'contact_results': contact_results,
                    'path_analysis': path_analysis,
                    }
            ## DEFINING PICKLE PATH
            pickle_name = main_compute_contacts_for_groups.__name__
            path_pickle = os.path.join(path_analysis, pickle_name)
            
            ## PERFORMING THE RESULTS
            group_dict = save_and_load_pickle(function = main_compute_contacts_for_groups, 
                                               inputs = input_main_func, 
                                               pickle_path = path_pickle,
                                               rewrite = False)
            
            ## GETTING AVERAGE CONTACTS
            avg_contacts = np.mean(group_dict['contacts_storage'], axis = 1)
            
            ## STORING
            group_dict['avg_contacts'] = avg_contacts
            
            ## STORING THE INFORMATION
            output_dict={**output_dict, **group_dict}
            
        ## SEEING IF YOU WANT COM DISTANCES
        if want_com_distance is True:
            ## GETTING TIME ARRAY AND Z DISTANCE
            time_array, z_dist = get_time_array_and_z_dist_from_com_distances(path_to_sim = path_to_sim)
            ## DEFINING DISTANCE AND TIME 
            
            ## GETTING INDICES
            indices = get_indices_for_truncate_time(time_array,
                                                    last_time_ps = self.last_time_ps)
            ## REDEFINING TIME ARRAY (starting at zero)
            time_array = time_array[indices] - time_array[indices][0]
            z_dist = z_dist[indices]
            
            ## GETTING AVG Z DISTANCE
            z_mean_dist = np.mean(z_dist)
            
            ## DEFINING DICTIONARY
            group_dict = {
                    'com_time_array' : time_array,
                    'com_z_dist' : z_dist,
                    'com_z_mean_dist': z_mean_dist,
                    }
            ## STORING THE INFORMATION
            output_dict={**output_dict, **group_dict}

        return output_dict
    
    ### FUNCTION TO GET AVERAGE CONTACTS FOR EACH GROUP
    @staticmethod
    def compute_contacts_for_lm_groups(contacts_dict):
        '''
        The purpose of this function is to compute average contacts for lipid membrane groups, 
        e.g. tail and head groups.
        INPUTS:
            contacts_dict: [dict]
                dictionary of contacts
        OUTPUTS:
            output_dict: [dict]
                dictionaty of avg contacts, contact array, etc.
        '''
        output_dict = {}
        ## LOOPING THROUGH EACH LM GROUP
        for each_group in contacts_dict['lm_groups'].keys():
            ## SEARCHING FOR INDEX IN PERMUTATION LIST
            indices = np.array([ idx for idx, key in enumerate(contacts_dict['permutation_list']) if each_group in key ])
            ## GETTING CONTACTS ARRAY
            contacts_array = contacts_dict['contacts_storage'][indices]
            ## SUMMING
            sum_contacts = np.sum(contacts_array, axis = 0)
            ## AVGING CONTACT
            avg_contacts = np.mean(sum_contacts)
            
            if each_group == "TAIL_GRPS":
                output_name="TAILGRPS"
            else:
                output_name = each_group
            
            ## ADDING TO OUTPUT
            output_dict[output_name] = {
                    'contacts_array': contacts_array,
                    'sum_contacts': sum_contacts,
                    'avg_contacts': avg_contacts,
                    }
            
        return output_dict
    
    ### FUNCTION TO LOOP THROUGH AND GET NUMBER OF CONTACTS VS. TIME
    def plot_num_contacts_vs_time_for_config(self,
                                             path_to_sim,
                                             label = None,
                                             current_group_color = 'k',
                                             want_com_distance = False,
                                             want_com_distance_abs = False,
                                             want_tail_groups = False,
                                             color_com_axis='red',
                                             fig = None,
                                             ax = None,):
        '''
        The purpose of this function is to plot number of contacts versus time. 
        INPUTS:
            path_to_sim: [str]
                path to the simulation directory.
            want_com_distance: [logical]
                com distances plotted as a second y axis
            want_com_distance_abs: [logical]
                True if you want com distance to be positive
            want_tail_groups: [logical]
                True if you want tail groups only plotted as contacts
            label: [str]
                label for the plot
            current_group_color: [str]
                color for the image
            fig, ax:
                figure and axis object
        OUTPUTS:
            fig, ax:
                figure and axis for the plot
        '''
        if fig is None or ax is None:
            ## CREATING FIGURE
            fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
            
            ## ADDING TO AXIS
            ax.set_xlabel("Time (ps)")
            if want_tail_groups is True:
                ax.set_ylabel("Number of DOPC tail contacts")                
            else:
                ax.set_ylabel("Number of contacts")

        if want_tail_groups is True:
            want_nplm_grouping = True
        else:
            want_nplm_grouping = False

        ## GET THE CONTACTS INFORMATION
        contacts_dict = self.analyze_num_contacts(path_to_sim,
                                                  want_com_distance = want_com_distance,
                                                  want_nplm_grouping = want_nplm_grouping)
        
        
        
        if want_tail_groups is True:
            ## COMPUTING LM CONTACTS
            lm_contact_dict = self.compute_contacts_for_lm_groups(contacts_dict = contacts_dict)
            ## X AND Y
            y = lm_contact_dict['TAILGRPS']['sum_contacts']
            
        else:
            y = contacts_dict['num_contacts_array']
        ## DEFINING X VALUES
        x = contacts_dict['time_array']
        ## ADDING TO PLOT
        ax.plot(x,
                y, 
                color=current_group_color, 
                label = label)
        
        ## ADDING SECOND Y
        if want_com_distance is True:
            ## ADD SECOND Y AXIS
            ax2 = ax.twinx() 
            
            ## DEFINING COM DISTANCE
            com_dist = contacts_dict['com_z_dist']
            if want_com_distance_abs is True:
                com_dist = np.abs(com_dist)
            
            ## ADDING TO PLOT
            ax2.plot(contacts_dict['com_time_array'],
                     com_dist,
                     linestyle='-', 
                     color=color_com_axis, 
                     label = "Distance" ) # specific_config + "_dist"
            ax2.set_ylabel('COM Distance (nm)', color = color_com_axis)
            ax2.tick_params(axis='y', labelcolor=color_com_axis)
        else:
            ax2 = None

        return fig, ax, ax2
    
    ### FUNCTION TO LOOP THROUGH EACH CONFIG
    def plot_num_contacts_vs_time_all_config(self):
        '''
        This function simply plots all contacts versus time
        '''
        
        ## GETTING COLOR MAP
        cmap = plot_funcs.get_cmap( len(self.job_info.config_library) )
        fig, ax = None, None
        
        ## LOOPING THROUGH EACH
        for idx, path_to_sim in enumerate(self.job_info.path_simulation_list):
            
            ## GETTING CONFIG NAME
            if len(self.job_info.config_library) > 1:
                config_name = self.job_info.config_library[idx]
                want_com_distance = False
            else:
                config_name = None
                want_com_distance = True
            
            ## GENERATING PLOT
            fig, ax, ax2 = self.plot_num_contacts_vs_time_for_config(path_to_sim = path_to_sim,
                                                                    label = config_name,
                                                                    current_group_color = cmap(idx),
                                                                    want_com_distance = want_com_distance,
                                                                    fig = fig,
                                                                    ax = ax
                                                                    )
        
        ## ADDING COLORBAR 
        if len(self.job_info.path_simulation_list) > 1:
            ## GETTING TICK LABELS
            x = np.array(self.job_info.config_library).astype('float')
            normalized = (x-min(x))/(max(x)-min(x))
            # norm=plt.Normalize(vmin=0, vmax=1)
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax = ax,
                                ticks =  normalized)
            cbar.ax.set_yticklabels(self.job_info.config_library)
        
        ## ADDING LEGEND 
        try:
            ax2.legend()
            ax.grid()
        except Exception:
            pass
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        ## SAVING FIGURE
        figure_name = self.job_info.specific_sim + "_num_contacts_vs_time"
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        return fig, ax
    
    ### FUNCTION TO LOOP AND GENERATE ALL CONFIG PLOTS FOR GROUPED
    def plot_num_contacts_grouped_single_config(self,
                                                path_to_sim = None,
                                                contacts_dict = None,
                                                fig_name = ''):
        '''
        This function plots number of contacts for a single configuration.
        INPUTS:
            path_to_sim: [str]
                path to the simulation directory.
            contacts_dict: [dict]
                dictionary containing the information for contacts
        OUTPUTS:
            fig, ax: 
                figure and axis for the image
        '''
        
        ## GET THE CONTACTS INFORMATION
        if contacts_dict is None:
            contacts_dict = self.analyze_num_contacts(path_to_sim = path_to_sim,
                                                      want_nplm_grouping = True,
                                                      want_com_distance = False)
        
        ## PLOTTING NPLM CONTACTS
        fig, ax = plot_nplm_contacts_vs_time(contacts_storage = contacts_dict['contacts_storage'],
                                             permutation_list = contacts_dict['permutation_list'],
                                             np_groups = contacts_dict['np_groups'],
                                             time_array = contacts_dict['time_array'],
                                             )
        
        ## DEFINING FIGURE NAME
        figure_name = fig_name + "_numcontactsgrouped_vs_time"
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        return fig, ax
    
    ### FUNCTION TO LOOP AND GENERATE ALL NUM CONTACTS
    def plot_num_contacts_grouped_all_configs(self,
                                              want_single_config_plot = True,
                                              want_avg_plot = False):
        '''
        This function loops through all configurations and generates configurations 
        for them.
        INPUTS:
            self: [obj]
                class object
            want_single_config_plot: [logical]
                True if you want a single config plot
            want_avg_plot: [logical]
                True if you want average plot across distance
            
        OUTPUTS:
            
        '''
        ## DEFINING STORAGE
        avg_contacts_storage = []
        fig_list_single_config = []
        ax_list_single_config = []
        
        ## SEEING IF YOU WANT AVERAGE PLOT
        if want_avg_plot is True:

            ## STORE THE INFORMATION
            storage_list = ['com_z_mean_dist', 
                            'avg_contacts', 
                            'permutation_list',
                            'np_groups', 
                            'avg_contacts']
            ## TURNING ON COM DISTANCES
            want_com_distance = True
            
        ## LOOPING THROUGH EACH
        for idx, path_to_sim in enumerate(self.job_info.path_simulation_list):
            ## GET THE CONTACTS INFORMATION
            contacts_dict = self.analyze_num_contacts(path_to_sim = path_to_sim,
                                                      want_nplm_grouping = True,
                                                      want_com_distance = want_com_distance)
            ## PLOTTING
            if want_single_config_plot is True:
                fig, ax = self.plot_num_contacts_grouped_single_config(contacts_dict = contacts_dict,
                                                           fig_name = self.job_info.specific_sim + '-' + self.job_info.config_library[idx])
                ## APPENDING
                fig_list_single_config.append(fig)
                ax_list_single_config.append(ax)
            ## SEEING IF YOU WANT AVERAGE PLOT
            if want_avg_plot is True:
                storage_dict = {
                        key: contacts_dict[key] for key in storage_list
                        }
                ## STORING
                avg_contacts_storage.append(storage_dict)
        try:
            ## PLOTTING AVERAGE
            if want_avg_plot is True:
                ## GETTING EXTRACTED DATA
                extract_groups_vs_dist = extract_data_for_groups_vs_distance(avg_contacts_storage)
                
                ## PLOTTING
                fig_avg, ax_avg = plot_nplm_avg_contacts_vs_distances(extract_groups_vs_dist)
                
                ## STORING THE FIGURE
                ## DEFINING FIGURE NAME
                figure_name = self.job_info.specific_sim + "_avgcontacts_vs_distance"
                ## SAVING FIGURE
                plot_funcs.store_figure( fig = fig_avg,
                                         path = os.path.join(IMAGE_LOC,
                                                             figure_name),
                                         save_fig = SAVE_FIG,
                                         )
                
                
        except ValueError:
            print("Error in x y sizes! Check average_contacts_storage")
            fig_avg, ax_avg = None, None
            
        return avg_contacts_storage, fig_list_single_config, ax_list_single_config, fig_avg, ax_avg

### FUNCTION TO CREATE DATAFRAME FROM CONTACS DICT
def create_group_avg_contacts_df(contacts_dict):
    '''
    This function creates the contacts dataframe from contacts dict. 
    INPUTS:
        contacts_dict: [dict]
            dictionary containing all the information for contacts
    OUTPUTS:
        df: [dataframe]
            pandas datafrmae containing the average contacts in a 
            dataframe form, for example:
                          GOLD         ALK         PEG       NGRP        RGRP
            HEADGRPS   0.00000    0.411178   54.155689  61.692615   20.205589
            TAIL_GRPS  0.01996  188.293413  303.694611  65.255489  498.654691
        The HEADGRPS and TAIL_GRPs are for the lipid membrane
    '''

    
    ## GETTING PERMUTATION LIST/AVG CONTACTS
    perm_list = contacts_dict['permutation_list']
    avg_contacts = contacts_dict['avg_contacts']
    
    ## LM GROUP KEYS
    lm_group_keys = contacts_dict['lm_groups'].keys()
    # np_group_keys = contacts_dict['np_groups'].keys()
    
    ## CREATING EMPTY DICTIONARY
    dict_storage = {}
    
    ## LOOPING PERM LIST AND GETTING EACH GROUP
    for each_group in lm_group_keys:
        ## FINDING ALL OF THE GROUP
        idxes_in_group = [idx_perm for idx_perm, each_perm in enumerate(perm_list) if each_group in each_perm]
        
        ## GETTING ALL CONTACTS 
        np_names_in_lm_group = [ perm_list[each_index].split('-')[0] for each_index in idxes_in_group]
        np_avg_contacts = [ avg_contacts[each_index] for each_index in idxes_in_group]
        
        ## STORING
        dict_storage[each_group] = { each_name : np_avg_contacts[idx] for idx, each_name in enumerate(np_names_in_lm_group) }
        
    # CREATING PANDAS DATAFRAME
    df = pd.DataFrame(dict_storage).T
    
    return df


## DEFINING PARENT SIMULATION PATH
parent_sim_path = PARENT_SIM_PATH



#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":

    #### DEFINING SIMULATION TYPE
    ## GOING THROUGH ALL SIMULATIONS
    sim_type_list = [
#             'us_forward_R12',
#             'us_reverse_R12',
#            'us_forward_R01',
#            'us_reverse_R12',
            'us_reverse_R01',
#            
#            'unbiased_ROT12_1.300',
#            'unbiased_ROT12_1.900',
#            'unbiased_ROT12_2.100',
#            
#            'unbiased_ROT001_1.700',
#            'unbiased_ROT001_1.900',
#            'unbiased_ROT001_2.100',
#            'unbiased_ROT001_2.300', 
            ]
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
                 
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING INPUTS
        contact_inputs= {
                'job_info': job_info,
                'last_time_ps': 50000,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs)
        
        ## PLOTTING ALL CONTACTS
        fig, ax = extract_contacts.plot_num_contacts_vs_time_all_config()
        
        ## DEFINING AVERAGE CONTACTS
        avg_contacts_storage, fig_list_single_config, ax_list_single_config, fig_avg, ax_avg =\
                     extract_contacts.plot_num_contacts_grouped_all_configs(want_single_config_plot = True,
                                                                                      want_avg_plot = True)
        
#    
#    
#    ## PLOTTING CONTACTS FOR A SPECIFIC CONFIG
#    fig, ax, ax2 = extract_contacts.plot_num_contacts_vs_time_for_config(path_to_sim = job_info.path_simulation_list[job_idx])
#    
#    fig, ax = extract_contacts.plot_num_contacts_grouped_single_config(path_to_sim = job_info.path_simulation_list[job_idx])
#
#    

    #%% PLOTTING FOR UNBIASED SIMS D = 5.3 NM, ROT012
    
    ## GOING THROUGH ALL SIMULATIONS
    sim_type_list = [
#             'unbiased_ROT012_5.300',
#            'unbiased_ROT012_5.300_rev',
#              'unbiased_ROT012_1.300',
#             'unbiased_ROT012_2.100',
#            'unbiased_ROT001_1.900',
#            'unbiased_ROT001_2.100',
#            'unbiased_ROT012_1.900',
#             'unbiased_ROT012_3.500',
#              'unbiased_ROT001_3.500',
#             'us_forward_R12',
#            'pullthenunbias_ROT001',
            'pullthenunbias_ROT012',
            ]
    
    ## GETTING ALL UNBIASED SIMS
    # sim_type_list = [each_key for each_key in NPLM_SIM_DICT if 'unbiased' in each_key]    
    
    ## CREATING STORAGE
    storage_list = []
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING INPUTS
        contact_inputs= {
                'job_info': job_info,
                'last_time_ps': 50000,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs)
        
        ## DEFINING PATH TO SIMULATION
        path_to_sim = job_info.path_simulation_list[0]
        
        ## GETTING CONTACTS PER GROUP
        contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                                  want_nplm_grouping = True,
                                                  want_com_distance = True)
        
        ## GETTING AVERAGE END DISTANCE
        output_dict={
                'sim_name': sim_type,
                'avg_output_com': np.mean( np.abs(contacts_dict['com_z_dist'][-100:]) ) # last 10 ns
                }
        
        ## SAVING FIGURE
        if sim_type  == 'us_forward_R12':
            figure_name = "Biased_contacts-%s"%(specific_sim)
        else:
            figure_name = "unbiased_num_contacts-%s"%(specific_sim)
        
        ## PLOTTING CONTACT VERSUS TIME
        fig, ax, ax2 = extract_contacts.plot_num_contacts_vs_time_for_config(path_to_sim = path_to_sim,
                                                                             want_com_distance = True,
                                                                             want_com_distance_abs = True,
                                                                             want_tail_groups = True)
        fig.tight_layout()
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name),
                                 save_fig = True,
                                 )
        
        
        
        
        #%%
        
        
        ### GETTING DETAILS FROM UMBRELLA SAMPLING SIMULATIONS
        if "unbiased" in sim_type:
            ## FINDING UMBRELLA SAMPLING THAT MATCHES IT
            if "rev" in sim_type:
                if "ROT001" in sim_type:
                    us_dir_key = "us_reverse_R01"
                elif "ROT012" in sim_type:
                    us_dir_key = "us_reverse_R12"
                else:
                    print("Error! No reverse sim of this type is found: %s"%(sim_type) )
            else:
                if "ROT001" in sim_type:
                    us_dir_key = "us_forward_R01"
                elif "ROT012" in sim_type:
                    us_dir_key = "us_forward_R12"
                else:
                    print("Error! No reverse sim of this type is found: %s"%(sim_type) )
        
        ## FINDING CONFIG
        config_key = sim_type.split('_')[2]
        
        print("Umbrella sampling extraction point (%s): %s, config: %s"%(sim_type, us_dir_key, config_key) )
        
        ## DEFINING MAIN SIM DIR
        us_main_sim_dir= NPLM_SIM_DICT[us_dir_key]['main_sim_dir']
        us_specific_sim= NPLM_SIM_DICT[us_dir_key]['specific_sim']
        
        ## GETTING JOB INFORMATION
        job_info_us = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = us_main_sim_dir,
                                  specific_sim = us_specific_sim)
        
        idx = job_info_us.config_library.index(config_key)
        path_to_sim = job_info_us.path_simulation_list[idx]
        
        ## DEFINING INPUTS
        contact_inputs_us= {
                'job_info': job_info_us,
                'last_time_ps': 50000,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs_us)
        
        ## GETTING CONTACTS PER GROUP
        contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                                  want_nplm_grouping = True,
                                                  want_com_distance = True)


        ## LOOPING THROUGH EACH LM GROUP
        for each_group in contacts_dict['lm_groups'].keys():
            ## SEARCHING FOR INDEX IN PERMUTATION LIST
            indices = np.array([ idx for idx, key in enumerate(contacts_dict['permutation_list']) if each_group in key ])
            ## GETTING CONTACTS ARRAY
            contacts_array = contacts_dict['contacts_storage'][indices, -100:] # Last 10 ns
            ## SUMMING
            sum_contacts = np.sum(contacts_array, axis = 0)
            ## AVGING CONTACT
            avg_contacts = np.mean(sum_contacts)
            
            if each_group == "TAIL_GRPS":
                output_name="TAILGRPS"
            else:
                output_name = each_group
            
            ## ADDING TO OUTPUT
            output_dict['contacts_' + output_name] = avg_contacts
            
        ## APPENDING
        storage_list.append(output_dict)
        
    ## CREATING PANDAS DATAFRAME
    df = pd.DataFrame(storage_list)
    
    ## OUTPUTTING
    csv_name="unbiased_sim_contacts_using_US.csv"
    csv_location=os.path.join(IMAGE_LOC,
                           csv_name)
    df.to_csv(csv_location)
    print("Outputting csv to: %s"%(csv_location) )
    
        #%%
        
        
    ## GETTING JOB INFOMRATION
    job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                              main_sim_dir = main_sim_dir,
                              specific_sim = specific_sim)
    
    ## DEFINING INPUTS
    contact_inputs= {
            'job_info': job_info,
            'last_time_ps': 50000,
            }
    
    ## DEVELOPING SCRIPT FOR CONTACTS
    extract_contacts = extract_num_contacts(**contact_inputs)
    
    if sim_type  == 'us_forward_R12':
        idx = job_info.config_library.index('5.100')
        path_to_sim = job_info.path_simulation_list[idx]
    else:
        path_to_sim = job_info.path_simulation_list[0]
    
    ## SAVING FIGURE
    if sim_type  == 'us_forward_R12':
        figure_name = "Biased_contacts-%s"%(specific_sim)
    else:
        figure_name = "unbiased_num_contacts-%s"%(specific_sim)
    
    # '''
    ## PLOTTING CONTACT VERSUS TIME
    fig, ax, ax2 = extract_contacts.plot_num_contacts_vs_time_for_config(path_to_sim = path_to_sim,
                                                                         want_com_distance = True,
                                                                         want_com_distance_abs = True)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(IMAGE_LOC,
                                                 figure_name),
                             save_fig = True,
                             )
    # '''
                             
                             
    ## GETTING CONTACTS PER GROUP
    contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                              want_nplm_grouping = True,
                                              want_com_distance = True)
    ## GETTING AVERAGE END DISTANCE
    output_dict={
            'sim_name': sim_type,
            'avg_output_com': np.mean( np.abs(contacts_dict['com_z_dist'][-100:]) ) # last 10 ns
            }
    

        
        
    
    
    ## LOOPING THROUGH EACH LM GROUP
    for each_group in contacts_dict['lm_groups'].keys():
        ## SEARCHING FOR INDEX IN PERMUTATION LIST
        indices = np.array([ idx for idx, key in enumerate(contacts_dict['permutation_list']) if each_group in key ])
        ## GETTING CONTACTS ARRAY
        contacts_array = contacts_dict['contacts_storage'][indices, :100]
        ## SUMMING
        sum_contacts = np.sum(contacts_array, axis = 0)
        ## AVGING CONTACT
        avg_contacts = np.mean(sum_contacts)
        
        if each_group == "TAIL_GRPS":
            output_name="TAILGRPS"
        else:
            output_name = each_group
        
        ## ADDING TO OUTPUT
        output_dict['contacts_' + output_name] = avg_contacts
        
    ## APPENDING
    storage_list.append(output_dict)
                             
    if sim_type  == 'us_forward_R12':
        fig, ax = extract_contacts.plot_num_contacts_grouped_single_config(path_to_sim = path_to_sim)
    
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name + '-grouped' ),
                                 save_fig = True,
                                 )
    
    #%%
    ## CREATING PANDAS DATAFRAME
    df = pd.DataFrame(storage_list)
    
    ## OUTPUTTING
    csv_name="unbiased_sim_contacts.csv"
    csv_location=os.path.join(IMAGE_LOC,
                           csv_name)
    df.to_csv(csv_location)
    print("Outputting csv to: %s"%(csv_location) )
    
    
    
        
        #%%
        
#        output_dict =  extract_contacts.analyze_num_contacts(
#                             path_to_sim = path_to_sim,
#                             want_nplm_grouping = False,
#                             want_com_distance = True)    
    

        
    #%% FORWARD AND REVERSE SIM COMPARISON
    
    ## DEFINING JOB TYPES
    forward_reverse_dict = {
            'forward': 'us_forward_R12',
            'reverse': 'us_reverse_R12',
            }
    
    ## GETTING SPECIFIC DISTANCE
    dist="5.300"

    ## STORING CONTACTS DICT
    contacts_dict_df_storage = {}
    ## LOOPING THROUGH EACH DICTIONARY
    for each_key in forward_reverse_dict:
        ## DEFINIGN SIMULATION
        sim_type = forward_reverse_dict[each_key]
        
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
    
    
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING INPUTS
        contact_inputs= {
                'job_info': job_info,
                'last_time_ps': 50000,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs)
        

        ## FINDING JOB
        name_list_basename = [ os.path.basename(each_job) for each_job in job_info.path_simulation_list ]
        
        ## LOCATING
        job_idx = name_list_basename.index(dist)

        ## DEFINING PATH TO SIM
        path_to_sim = job_info.path_simulation_list[job_idx]
        
        ## GET THE CONTACTS INFORMATION
        contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                                              want_nplm_grouping = True,
                                                              want_com_distance = True)
        ## CREATING GROUP DF
        df = create_group_avg_contacts_df(contacts_dict = contacts_dict)
        
        ## STORING
        contacts_dict_df_storage[each_key] = df
    
    
    #%% PLOTTING BAR PLOT
    
    ## MAKING PLOTS
    
    ## DEFINING PLLOT WIDTH
    bar_width=0.3
    
    ## FINDING POSIITON OF BAR
    bar_positions = [ np.arange(contacts_dict_df_storage[each_df].shape[1]) + idx * bar_width 
                         for idx,each_df in enumerate(contacts_dict_df_storage)]
    
    # Bar plot reference
    # https://python-graph-gallery.com/11-grouped-barplot/
    
    color_dict={
            'forward': 'b',
            'reverse': 'k',
            }
    
    ## PLOTTING BAR PLOT
    ## CREATING FIGURES
    fig, axs = plt.subplots(nrows=len(contacts_dict_df_storage[next(iter(contacts_dict_df_storage))]), 
                            sharex=True)
    
    ## PLOTTING
    for type_idx, each_key in enumerate(contacts_dict_df_storage):
        ## DEFINING COLOR
        color = color_dict[each_key]
        
        ## GETTING DF
        df = contacts_dict_df_storage[each_key]
        idx = 0
        for label, row in df.iterrows(): # .iterrows()
            if idx == 0: #  and type_idx == 0
                bar_label = each_key
            else:
                bar_label = None
            
            ## PLOTTING
            axs[idx].bar(bar_positions[type_idx], 
                         row.to_numpy(), 
                         width = bar_width, 
                         color = color,
                         label = bar_label)
            
            ## ADDING TITLE ONCE
            if type_idx == 0:
                ## ADDING TITLE
                axs[idx].set_title(label, loc='left', y = 0.85, x = 0.05)
                ## ADDING GRID
                axs[idx].xaxis.grid(True)
                ## ADDING Y LABEL
                axs[idx].set_ylabel("Avg. contacts")
            ## ADDING TO INDEX
            idx+=1
            
        ## SETTING X LABEL
#         ax.set_xticks
    ## ADDING LEGEND
    axs[0].legend(loc = 'upper right')
    
    ## SETTING XTICKS
    plt.xticks([r + bar_width/len(contacts_dict_df_storage) for r in range(len(bar_positions[0]))], df.columns )
        
    ## ADJUSTING SIZE
    width, height = plot_funcs.cm2inch( 10, 8 )
    fig.set_figheight(height)
    fig.set_figwidth(width)

    ## TIGHTLAYOUT
    fig.tight_layout()
    
    ## ADJUSTING SPACE OF SUB PLOTS
    plt.subplots_adjust(wspace=0, hspace=0)

    figure_name = "avg_contacts_barplot_%s_%s"%(forward_reverse_dict['forward'],dist)

    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(IMAGE_LOC,
                                                 figure_name),
                             save_fig = True,
                             )
    
    #%% PLOTTING AVERAGE COM DISTANCE
    
    ## DEFINING JOB TYPES
    forward_reverse_dict = {
            'forward': 'us_forward_R12',
            'reverse': 'us_reverse_R12',
            }
    
    ## DEFINING JOB TYPES
    forward_reverse_dict = {
            'forward': 'us_forward_R01',
            'reverse': 'us_reverse_R01',
            }
    
    ## DEFINING STORAGE
    contacts_dict_df_storage = {}
    
    ## LOOPING THROUGH EACH DICTIONARY
    for each_key in forward_reverse_dict:
        ## DEFINIGN SIMULATION
        sim_type = forward_reverse_dict[each_key]
        
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
    
    
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING INPUTS
        contact_inputs= {
                'job_info': job_info,
                'last_time_ps': 50000,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs)
        

        ## FINDING JOB
        name_list_basename = [ os.path.basename(each_job) for each_job in job_info.path_simulation_list ]
        
        ## STORING
        contacts_dict_df_storage[each_key] = {}
        ## LOOPING
        for job_idx, basename in enumerate(name_list_basename):
            
            ## DEFINING PATH TO SIM
            path_to_sim = job_info.path_simulation_list[job_idx]
            
            ## GET THE CONTACTS INFORMATION
            contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                                                  want_nplm_grouping = True,
                                                                  want_com_distance = True)
            ## CREATING GROUP DF
            df = create_group_avg_contacts_df(contacts_dict = contacts_dict)
            
            ## STORING
            contacts_dict_df_storage[each_key][basename] ={
                    'df': df,
                    'com_z_mean_dist': contacts_dict['com_z_mean_dist']
                    }
            
    #%%
    ## PLOTTING THE IMAGE
    ## DEFINING COLORS
    GROUP_COLOR_DICT={
        'GOLD':  'gold',
        'RGRP': 'black',
        'ALK': 'tab:pink',
        'PEG': 'cyan',
        'NGRP': 'purple',
        }
    
    ## DEFINING LIMITS
    AX_LIMITS_DICT = {
            'HEADGRPS':{
                    'y_ticks': np.arange(0, 400, 100)
                      },
            'TAIL_GRPS':{
                    'y_ticks': np.arange(0, 1500, 200)
                      },
            'TAILGRPS':{
                    'y_ticks': np.arange(0, 1500, 200)
                      },
            }
    
    ## DEFINING DICTIONARY
    for contacts_dict_key in contacts_dict_df_storage:
        ## DEFINING EACH DICT
        each_dict = contacts_dict_df_storage[contacts_dict_key]
     
        ## STORING
        x_data = []
        y_data = {}
        
        ## EXTRACTING THE DATA
        for key_idx, each_key in enumerate(each_dict):
            ## GETTING ALL X DATA
            x_data.append(each_dict[each_key]['com_z_mean_dist'])
            
            ## DEFINING DF
            current_df = each_dict[each_key]['df']
           
            ## GETTING ALL Y DATA
            for row_idx, each_row in enumerate(current_df.iterrows()):
                
                ## TAIL GROUP RENAMING
                row_name = each_row[0]
                if row_name == "TAIL_GRPS":
                    row_name = "TAILGRPS"
                    
                if key_idx == 0:
                    ## CREATING EMPTY DICTIONARY FOR Y
                    y_data[row_name] = { each_col: [] for each_col in current_df.columns}
                    
                ## STORING THE DATA
                for each_col in current_df.columns:
                    y_data[row_name][each_col].append(each_row[1][each_col])
                    
        ## SORTING THE DATA
        idx_sorted = np.argsort(x_data)
        
        ## SORTED DATA
        x_data_sorted = np.array(x_data)[idx_sorted]
        y_data_sorted = { key_0: {key_1: np.array(y_data[key_0][key_1])[idx_sorted] for key_1 in y_data[key_0]  } for key_0 in y_data  }
        
        ## LOOPING THROUGH EACH TYPE
        for each_type in y_data_sorted:
            ## PLOTTING
            ## CREATING FIGURE
            fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
            
            ## ADDING AXIS
            ax.set_xlabel("z (nm)")
            ax.set_ylabel("Avg. contacts")
            
            ## PLOTTING
            for each_key in y_data_sorted[each_type]:
                ## GETTING COLOR
                color = GROUP_COLOR_DICT[each_key]
                
                y_values = y_data_sorted[each_type][each_key]
                x_values = x_data_sorted
                
                ## PLOTTING
                ax.plot(x_values, y_values, label=each_key, color = color)
            
            ## SETTING AX XLIM
            ax.set_xticks(np.arange(1,7,1))
            ax.set_yticks(AX_LIMITS_DICT[each_type]['y_ticks'])
               
            ## ADDING LEGEND
            ax.legend()
          
            ## TIGHT LAYOUT
            fig.tight_layout()
            
            ## SAVING FIGURE
            figure_name = "avg_contacts_vs_time-%s_%s"%(forward_reverse_dict[contacts_dict_key],each_type)
        
            ## SAVING FIGURE
            plot_funcs.store_figure( fig = fig,
                                     path = os.path.join(IMAGE_LOC,
                                                         figure_name),
                                     save_fig = True,
                                     )
    
        
        
        # , columns = np_group_keys
        
        
        ## 
        
