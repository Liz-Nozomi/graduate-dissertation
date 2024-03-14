# -*- coding: utf-8 -*-
"""
surfactant_analysis.py
this is the main script for surfactant analysis

CREATED ON: 02/24/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os, sys
import numpy as np
import mdtraj as md
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path

## IMPORT COM COMPUTING TOOL
from MDDescriptors.core.calc_tools import compute_com 

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv

## IMPORTING ANALYSIS COMMANDS
from MDDescriptors.application.surfactants.rdf import compute_rdf, compute_interfacial_rdf
from MDDescriptors.application.surfactants.h_bond import compute_hbonds_configurations
from MDDescriptors.application.surfactants.hydration_fe import compute_hydration_fe

### FUNCTION TO CHECK IF YOU ARE RUNNING ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False

def read_ndx( filename ):
    r'''
    '''
    with open( filename, 'r' ) as outputfile: # Opens gro file and make it writable
        file_data = outputfile.readlines()
    
    # SPLIT FILE INTO LINES
    groups = {}
    lines = [ n.strip( '\n' ) for n in file_data ]
    indices = []
    for nn, line in enumerate(lines):
        if "[" in line and len(indices) < 1:
            ## ADD FIRST GROUP TO DICT
            key = line.strip('[ ')
            key = key.strip(' ]')
        elif "[" in line and len(indices) > 1:
            ## ADD GROUP TO DICT
            groups[key] = np.array([ int(ndx)-1 for ndx in indices ])
            indices = []
            key = line.strip('[ ')
            key = key.strip(' ]')
        else:
            ## ADD INDICES TO GROUP
            indices += line.split()

    ## GET LAST GROUP
    groups[key] = np.array([ int(ndx)-1 for ndx in indices ])
    
    return groups

def load_md_traj( path_traj,
                  input_prefix,
                  make_whole = True,
                  add_bonds = True,
                  index_file = None,
                  center_residue_name = 'SUR',
                  output_name = 'System',
                  want_rewrite = False,
                  want_water = True,
                  water_residue_name = 'SOL',
                  keep_ndx_components = True,
                  ndx_components_file = 'surfactant.ndx',
                  ):
    r'''
    '''
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv( wd = path_traj )
    
    ## CHECK IF SIMULATION IS BULK OR CONTAINS SURFACTANT
    if "bulk" not in sd: ## SIMULATION CONTAINS SURFACTANT
        ## CENTER SURFACTANT IN BOX AND MAKE MOLECULES WHOLE
        func_inputs = { "index_file"          : index_file,
                        "center_residue_name" : center_residue_name,
                        "output_name"         : output_name,
                        "rewrite"             : want_rewrite,
                        "want_water"          : want_water,
                        "water_residue_name"  : water_residue_name,
                        "keep_ndx_components" : keep_ndx_components,
                        "ndx_components_file" : ndx_components_file,
                       }
        ## EXECUTE TRJCONV COMMANDS
        path_gro, path_xtc, path_tpr, index_file = trjconv_conversion.generate_center_pbc_mol( input_prefix + ".tpr",
                                                                                               input_prefix + ".xtc",
                                                                                               **func_inputs )    
        ## MAKE MOLECULES WHOLE BEFORE LOADING
        new_prefix = path_gro.strip('.gro')
        path_pdb, path_xtc = trjconv_conversion.generate_pdb_from_gro( new_prefix,
                                                                       make_whole = make_whole,
                                                                       rewrite = want_rewrite )
    else: ## SIMULATION IS BULK
        ## DETERMINE TRAJECTORY TYPE
        ## CHECK IF SIMULATION IS GROMACS OR TINKER
        if "amoeba" in sd:
            ## SIMPLY LOAD TINKER TRAJ
            path_pdb = os.path.join( path_to_sim, input_prefix + ".pdb" )
            path_xtc = os.path.join( path_to_sim, input_prefix + ".arc" )
        else:
            ## MAKE MOLECULES WHOLE BEFORE LOADING
            path_pdb, path_xtc = trjconv_conversion.generate_pdb_from_gro( input_prefix,
                                                                           make_whole = make_whole,
                                                                           rewrite = want_rewrite )
    ## DETERMINE TRAJECTORY TYPE
    ## CHECK IF SIMULATION IS GROMACS OR TINKER
    if os.path.splitext( path_xtc )[-1] != ".arc":
        ## LOAD FULL GROMACS TRAJECTORY
        print( "LOADING GROMACS TRAJECTORY")
        traj = md.load( path_xtc,
                        top = path_pdb )
    else:
        ## LOAD FULL TINKER TRAJECTORY
        print( "LOADING TINKER TRAJECTORY")
        traj = md.load( path_xtc )
        traj_pdb = md.load( path_pdb )
        traj.topology = traj_pdb.topology 
                        
    return traj

def compute_r_cutoff( traj,
                      path_pkl,
                      ref_ndx,
                      atom_indices,
                      atom_group_key,
                      use_com = False,
                      r_range = ( 0, 2.0 ),
                      bin_width = 0.02,
                      periodic = True,
                      residue_list = [ 'HOH' ],
                      verbose = True,
                      print_freq = 100,
                      want_rewrite = False,
                      plot_fig = False,
                      ):
    r'''
    '''
    ## UPDATE TRAJ SO HEAVY ATOM HAS COM POSITION
    com = compute_com( traj, atom_indices )
    traj.xyz[:,ref_ndx,:] = com
    
    ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
    analysis_kwargs = { 'traj'         : traj,
                        'path_pkl'     : path_pkl,
                        'ref_ndx'      : ref_ndx,   # COM ndx for atom group, -1 for bulk
                        'use_com'      : use_com,
                        'r_range'      : r_range,
                        'bin_width'    : bin_width,
                        'periodic'     : periodic,
                        'residue_list' : residue_list,
                        'verbose'      : verbose,
                        'print_freq'   : print_freq,
                        'rewrite'      : want_rewrite }

    results = compute_rdf(**analysis_kwargs)
    
    ## SOLVER TO FIND MINIMUM
    ## GET ALL PEAKS(THROUGHS)
    peaks, _ = find_peaks( -results[:,1] )
    ## DETERMINE PEAK PROMINENCES
    prominences = peak_prominences( -results[:,1], peaks )[0]
    ## SELECT MAX PROMINENCE FOR PEAK FINDER
    prominence = prominences.max()
    ## GET PEAK WITH MAX PROMINENCE
    peak, _ = find_peaks( -results[:,1], prominence = prominence )
    r_ref = results[:,0][peak][0]

    ## PLOT RDF IF PLOT_FIG IS TRUE    
    if plot_fig is True:
        plt.figure()
        plt.plot( results[:,0], results[:,1] )
        plt.plot( [ r_ref, r_ref ], [ 0, 2 ], color = "black" )
    
    return r_ref

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## USER SPECIFIED INPUTS
    dist_cutoff  = 0.35      # nm, hbond distance cutoff
    angle_cutoff = 0.523598  # radians, hbond angle cutoff
    use_com      = True      # use com of water, false oxygen atom used as ref
    n_procs      = 20        # number of processors to use in parallel
    periodic     = True      # use pbc in calculations
    residue_list = [ "HOH" ] # list of residues to include in calculations
    verbose      = True      # print out to cmd line
    print_freq   = 1000       # print after N steps
    want_rewrite = False     # rewrite files
    plot_fig = False         # create rdf figure
    
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Documents\Box Sync\univ_of_wisc\manuscripts\hexyl_surfactants\raw_figures"    
    
    ## GROMACS OUTPUT
    input_prefix = "bulk_prod" # "bulk_prod"
    
    ## MAIN DIRECTORY
    main_dir = r"R:\surfactants\simulations\unbiased"
    
    ## SUB DIRECTORIES
    sub_directories = [ 
                        "bulk_300K_spce",
                        "bulk_300K_tip3p",
                        "bulk_300K_tip4p",
#                        "bulk_300K_amoeba",
#                        "surfactant_300K_heptane_spce_CHARMM36",
#                        "surfactant_300K_hexylamine_spce_CHARMM36",
#                        "surfactant_300K_hexylammonium_spce_CHARMM36",
#                        "surfactant_300K_hexylguanidinium_spce_CHARMM36",
#                        "surfactant_300K_heptane_tip3p_CHARMM36",
#                        "surfactant_300K_hexylamine_tip3p_CHARMM36",
#                        "surfactant_300K_hexylammonium_tip3p_CHARMM36",
#                        "surfactant_300K_hexylguanidinium_tip3p_CHARMM36",
#                        "surfactant_300K_heptane_tip4p_CHARMM36",
#                        "surfactant_300K_hexylamine_tip4p_CHARMM36",
#                        "surfactant_300K_hexylammonium_tip4p_CHARMM36",
#                        "surfactant_300K_hexylguanidinium_tip4p_CHARMM36",                        
                       ]

    ## LIST OF ANALYSIS FUNCTIONS
    analysis_list = { 
                      "hydration_fe"          : compute_hydration_fe,
#                      "interfacial_rdf"       : compute_interfacial_rdf,
#                      "hbonds_configurations" : compute_hbonds_configurations,
                      }

    ## LOOP THROUGH DIRECTORIES
    for sd in sub_directories:
        ## PREPARE DIRECTORY FOR ANALYSIS
        path_to_sim = check_server_path( os.path.join( main_dir, sd ) )
        
        ## LOAD TRAJECTORY CONVERT TRAJECTORY BY MAKING WHOLE AND ADDING BONDS
        traj = load_md_traj( path_traj = path_to_sim,
                             input_prefix = input_prefix,
                             make_whole = True,
                             add_bonds = True,
                             want_rewrite = want_rewrite )
            
#        ## LOAD TRAJECTORY
#        if check_spyder() is True:
#            traj = md.load( os.path.join( path_to_sim, input_prefix + ".pdb" ) )
#        else:
#            ## LOAD TRAJECTORY CONVERT TRAJECTORY BY MAKING WHOLE AND ADDING BONDS
#            traj = load_md_traj( path_traj = path_to_sim,
#                                 input_prefix = input_prefix,
#                                 make_whole = True,
#                                 add_bonds = True,
#                                 want_rewrite = want_rewrite )
        ## LOOP THROUGH ANALYSIS TYPES
        for analysis_key, function_obj in analysis_list.items():
            ## CHECK IF SIMULATION IS BULK OR CONTAINS SURFACTANT
            if "bulk" not in sd: ## SIMULATION CONTAINS SURFACTANT 
                ## LOAD INDEX FILE
                path_ndx = os.path.join( path_to_sim, "surfactant.ndx" )
                atom_groups = read_ndx( path_ndx )
                ## REMOVE FULL SURFACTANT GROUP, ONLY WANT ATOM GROUPS
                atom_groups = { key : group for key, group in atom_groups.items() if key != "SUR" }
            
                ## LOOP THROUGH ATOM GROUP TYPES
                for atom_group_key, atom_indices in atom_groups.items():
                    print( "\n--- GETTING R_REF FROM RDF ---" )
                    path_pkl = os.path.join( path_to_sim, r"output_files", r"{:s}_rdf_{:s}.pkl".format(input_prefix, atom_group_key.lower()) )
                    r_cutoff = compute_r_cutoff( traj,
                                                 path_pkl,
                                                 ref_ndx = atom_indices[0],
                                                 atom_indices = atom_indices,
                                                 atom_group_key = atom_group_key,
                                                 plot_fig = plot_fig )
                    ## PATH TO OUTPUT PICKLE FILE
                    path_pkl = os.path.join( path_to_sim, r"output_files", r"{:s}_{:s}_{:s}.pkl".format(input_prefix, analysis_key, atom_group_key.lower()) )
                    
                    ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
                    analysis_kwargs = { 'traj'         : traj,
                                        'path_pkl'     : path_pkl,
                                        'ref_ndx'      : atom_indices[0], # COM ndx for atom group, -1 for bulk
                                        'r_cutoff'     : 0.5, #r_cutoff,        # base on first hydration shell
                                        'dist_cutoff'  : dist_cutoff,
                                        'angle_cutoff' : angle_cutoff,
                                        'use_com'      : use_com,
                                        'n_procs'      : n_procs,
                                        'periodic'     : periodic,
                                        'residue_list' : residue_list,
                                        'verbose'      : verbose,
                                        'print_freq'   : print_freq,
                                        'rewrite'      : want_rewrite }
        
                    if analysis_key == "triplet_angle_distribution":
                        analysis_kwargs["use_com"] = False
                            
                    function_obj(**analysis_kwargs)
                    
            else: ## SIMULATION IS BULK
                ## PATH TO OUTPUT PICKLE FILE
                path_pkl = os.path.join( path_to_sim, r"output_files", r"{:s}_{:s}.pkl".format(input_prefix, analysis_key) )
                
                ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
                analysis_kwargs = { 'traj'         : traj,
                                    'path_pkl'     : path_pkl,
                                    'ref_ndx'      : -1,             # COM ndx for atom group, -1 for bulk
                                    'r_cutoff'     : 0.33,           # base on first hydration shell
                                    'dist_cutoff'  : dist_cutoff,
                                    'angle_cutoff' : angle_cutoff,
                                    'use_com'      : use_com,
                                    'n_procs'      : n_procs,
                                    'periodic'     : periodic,
                                    'residue_list' : residue_list,
                                    'verbose'      : verbose,
                                    'print_freq'   : print_freq,
                                    'rewrite'      : want_rewrite }
    
                if analysis_key == "triplet_angle_distribution":
                    analysis_kwargs["use_com"] = False
    
                if analysis_key == "hydration_fe":
                    analysis_kwargs["ref_coords"] = list(0.5*traj.unitcell_lengths[0,:])
                    
                function_obj(**analysis_kwargs)

