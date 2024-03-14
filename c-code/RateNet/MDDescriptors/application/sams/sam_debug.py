"""
sam_analysis.py
this is the main script for SAM analysis

CREATED ON: 02/18/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os
import pickle
import numpy as np
import mdtraj as md

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv
from MDDescriptors.core.read_write_tools import extract_itp     # Loading itp reading tool

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

## CREATING GRID FUCNTIONS
from MDDescriptors.surface.core_functions import create_grid
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## IMPORTING ANALYSIS FUNCTION
from MDDescriptors.application.sams.rdf import compute_interfacial_rdf
from MDDescriptors.application.sams.density import compute_density
from MDDescriptors.application.sams.triplet_angle import compute_triplet_angle_distribution
from MDDescriptors.application.sams.h_bond_properties import compute_num_hbonds

### FUNCTION TO CHECK IF YOU ARE RUNNING ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False

def add_bonds_to_traj( traj, itp_data = [], bond_water = True ):
    r'''
    '''
    if not itp_data:
        print( "no itp data loaded! Load itp data." )
    
    for itp in itp_data:
        ## DOUBLE CHECK THAT MDTRAJ LOADS IN CORRECT RESIDUE
        res_atoms = [ [ atom for atom in residue.atoms ] for residue in traj.topology.residues if residue.name == itp.residue_name ]
        for res in res_atoms:
            for a1, a2 in itp.bonds:
                traj.topology.add_bond( res[a1-1], res[a2-1] )

    ## ADD WATER BONDS
    if bond_water is True:
        ## GATHER WATER GROUPS
        res_atoms = [ [ atom for atom in residue.atoms ] for residue in traj.topology.residues if residue.name == "SOL" ]
        ## LOOP THROUGH WATER GROUPS
        for res in res_atoms:
            h1 = 0
            ## LOOP THROUGH MOLECULE
            for atom in res:
                if atom.element.symbol == "O":
                    oxygen = atom
                elif atom.element.symbol == "H" and h1 < 1:
                    hydrogen1 = atom
                    h1 = 1
                elif atom.element.symbol == "H" and h1 > 0:
                    hydrogen2 = atom
            ## BIND WATER
            traj.topology.add_bond( oxygen, hydrogen1 )
            traj.topology.add_bond( oxygen, hydrogen2 )
            
    return traj

def find_itp_files( directory = '.', extension = 'itp' ):
    r'''
    '''
    itp_files = []
    extension = extension.lower()
    
    for dirpath, dirnames, files in os.walk( directory ):
        for name in files:
            if extension in name:
                itp_files.append( os.path.join( dirpath, name ) )
    
    return itp_files

def load_pkl( path_pkl ):
    r'''
    Function to load data from pickle file
    '''
    print( "LOADING PICKLE FROM %s" % ( path_pkl ) )
    with open( path_pkl, 'rb' ) as input:
        data = pickle.load( input )
        
    return data

def compute_z_ref( path_wd,
                   input_prefix,
                   want_rewrite = False ):
    r'''
    '''
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv(wd = path_to_sim)
            
    func_inputs = {
                    'input_prefix': input_prefix,
                    'center_residue_name': "None",
                    'only_last_ns': True,
                    'rewrite': want_rewrite
                   }
    
    ## CONVERTING TRJ TO HEAVY WATER ATOMS FOR WC ANALYSIS
    wc_gro_file, wc_xtc_file = trjconv_conversion.generate_water_heavy_atoms(**func_inputs)

    ## LOADING TRAJECTORY
    traj = md.load( os.path.join(path_to_sim, wc_xtc_file),
                    top = os.path.join(path_to_sim, wc_gro_file))
            
    ## GENERATING GRID
    grid = create_grid( traj = traj, 
                        out_path = os.path.join( path_to_sim, r"output_files" ),
                        wcdatfilename = "willard_chandler_grid.dat", 
                        wcpdbfilename = "willard_chandler_grid.pdb", 
                        write_pdb = True, 
                        n_procs = 20,
                        verbose = True,
                        want_rewrite = want_rewrite,
                        alpha = WC_DEFAULTS["alpha"],
                        contour = 16., # use 0.5 bulk contour
                        mesh = WC_DEFAULTS["mesh"],
                        )
    
    ## REMOVE AIR-WATER INTERFACE
    grid = grid[grid[:,2] < grid[:,2].mean()]
    ## Z_REF = avg(grid)
    z_ref = grid[:,2].mean()
    
    return z_ref

def load_md_traj( path_traj,
                  input_prefix,
                  make_whole = True,
                  add_bonds = True,
                  want_rewrite = False,
                  standard_names = False ):
    r'''
    ''' 
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv( wd = path_traj )
    
    ## LOAD FULL TRAJECTORY
    full_pdb, full_xtc = trjconv_conversion.generate_pdb_from_gro( input_prefix, make_whole = make_whole, rewrite = want_rewrite )
    traj = md.load( full_xtc,
                    top = full_pdb,
                    standard_names = standard_names )
    
    if add_bonds is True:
        ## ITP FILES IN WD/LOOKS FOR THOSE CREATED BY MDBuilder
        itp_files = []
        all_itp_files = find_itp_files( directory = path_traj, extension = 'itp' )
        for itp in all_itp_files:
            with open( itp ) as f:
                line = f.readline()
                if "MDBuilder" in line:
                    itp_files.append(itp)
        del all_itp_files
        
        ## EXTRACT ITP DATA
        itp_data = [ extract_itp( itp_file ) for itp_file in itp_files if itp_file != '' ]
        
        ## UPDATE TRAJ WITH ITP DATA/ADDS BONDS TO NON WATER ATOMS
        traj = add_bonds_to_traj( traj, itp_data )
    
    return traj

def traj_info( traj ):
    r'''
    '''
    print( "\n---  TRAJECTORY INFO ---" )
    print( "%-12s %6d" % ( "# Frames:", traj.n_frames ) )
    list_of_residues = set( [ residue.name for residue in traj.topology.residues ] )
    print( "%-12s %6d" % ( "# Residues:", len(list_of_residues) ) )
    print( "%-12s %6d" % ( "# Atoms:", len( list(traj.topology.atoms) ) ) )
    for residue in list_of_residues:
        n_residue = len([ ii for ii in traj.topology.residues if ii.name == residue ])
        print( "%-12s %6d" % ( "# " + residue + ":", n_residue ) )

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## USER SPECIFIED INPUTS
    z_cutoff     = 0.3       # nm, z distance cutoff, only includes water below this value
    r_cutoff     = 0.33      # nm, radius cutoff for small cavities
    use_com      = True      # use com of water, false oxygen atom used as ref
    n_procs      = 20        # number of processors to use in parallel
    periodic     = True      # use pbc in calculations
    residue_list = [ "SOL" ] # list of residues to include in calculations
    verbose      = True      # print out to cmd line
    print_freq   = 100       # print after N steps
    want_rewrite = False     # rewrite files

    ## GROMACS OUTPUT
    input_prefix = "sam_test" # sys.argv[1]
    
    ## MAIN DIRECTORY
    main_dir = r"R:\simulations\polar_sams\unbiased"
    
    ## SUB DIRECTORIES
    sub_directories = [ 
                        "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                       ]
    
    ## LIST OF ANALYSIS FUNCTIONS
    analysis_list = { 
#                      "hbond_triplets" : compute_hbond_triplets,
#                      "hbonds_average"             : compute_hbonds_average,
#                      "hbonds_total"               : compute_hbonds_total,
#                      "hbonds_distribution"        : compute_hbonds_distribution,
                      }

    ## LOOP THROUGH DIRECTORIES
    for sd in sub_directories:
        ## PREPARE DIRECTORY FOR ANALYSIS
        path_to_sim = check_server_path( os.path.join( main_dir, sd ) )
        
        ## GET Z_REF FROM WC GRID
        z_ref = compute_z_ref( path_wd = path_to_sim, 
                               input_prefix = input_prefix,
                               want_rewrite = want_rewrite )

        ## LOAD TRAJECTORY CONVERT TRAJECTORY BY MAKING WHOLE AND ADDING BONDS
        traj = load_md_traj( path_traj = path_to_sim,
                             input_prefix = input_prefix,
                             make_whole = True,
                             add_bonds = True,
                             want_rewrite = want_rewrite )
        traj = traj[:1]
        ## PRINT TRAJ INFO
        traj_info( traj )
        
        ## COMPUTE HBOND TYPES
        path_pkl = os.path.join( path_to_sim, r"output_files", r"sam_test_hbonds_average.pkl" )
        compute_num_hbonds( traj,
                            path_pkl,
                            rewrite = True,
                            z_ref = z_ref,
                            z_cutoff = 0.3,
                            n_procs = 1,
                            print_freq = 1,
                                    )
        
#        ## COMPUTE ANALYSIS IN ANALYSIS_LIST
#        for analysis_key, function_obj in analysis_list.items(): 
#            print( "\n--- executing %s function ---" % analysis_key )
#            path_pkl = os.path.join( path_to_sim, r"output_files", r"{:s}_{:s}.pkl".format(input_prefix, analysis_key) )
#            
#            ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
#            analysis_kwargs = { 'traj'         : traj,
#                                'path_pkl'     : path_pkl,
#                                'z_ref'        : z_ref,
#                                'z_cutoff'     : z_cutoff,
#                                'r_cutoff'     : r_cutoff,
#                                'use_com'      : use_com,
#                                'n_procs'      : n_procs,
#                                'periodic'     : periodic,
#                                'residue_list' : residue_list,
#                                'verbose'      : verbose,
#                                'print_freq'   : print_freq,
#                                'rewrite'      : want_rewrite }
#
#            if analysis_key == "triplet_angle_distribution":
#                analysis_kwargs["use_com"] = False
#
#            function_obj(**analysis_kwargs)
            