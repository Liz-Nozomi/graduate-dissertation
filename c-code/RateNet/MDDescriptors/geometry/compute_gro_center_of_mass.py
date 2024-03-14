# -*- coding: utf-8 -*-
"""
compute_gro_center_of_mass.py
The purpose of this script is to compute the center of mass of a specific 
residue. 

INPUTS:
    - gro file
    - residue (list)
OUTPUTS:
    - summary file with center of mass of each
    
Written by: Alex K. Chew (10/31/2019)

GOOD REFERENCES:
    - center of mass calc with gromacs
        https://sajeewasp.com/iteratively-calculate-center-of-mass-gromacs/
"""

import os
import sys
import numpy as np 
# import numpy.matlib as npm # For matrix algebra
from optparse import OptionParser # Used to allow commands within command line

## IMPORTING GRO TOOLS
import MDDescriptors.core.read_write_tools as read_write_tools

### FUNCTION TO ASSIGN MASSES
def assign_mass(s):
    if (s=="C"):
        return 12.0107
    elif (s=="H"):
        return 1.00794
    elif (s=="N"):
        return 14.0067
    elif (s=="O"):
        return 15.9994
    elif (s=="P"):
        return 30.973762
    elif (s=="S"):
        return 32.065
    elif (s=="Au"):
        return 196.96657
    else:
        print("Error! No element mass for: %s"%(s) )
        print("Make sure to include it in the assign_mass function!")
        sys.exit()

## COMPUTING CENTER OF MASS
def calc_com(masses,xcoords,ycoords,zcoords):
    '''
    This function computes the center of mass
    INPUTS:
        masses: [np.array]
            masses of each atom
        xcoords, ycoords, zcoords: [np.array]
            x, y, z coordinates of each
    OUTPUTS:
        center of mass as a shape of a tuple, size 3
    '''
    xproduct=0.0
    yproduct=0.0
    zproduct=0.0
    for i in range(len(masses)):
        xproduct=xproduct+masses[i]*xcoords[i]
        yproduct=yproduct+masses[i]*ycoords[i]
        zproduct=zproduct+masses[i]*zcoords[i]
    return (xproduct/sum(masses),yproduct/sum(masses),zproduct/sum(masses))

#%%
if __name__ == "__main__":
    
    ### TURNING TEST ON / OFF
    testing = check_testing() # False if you're running this script on command prompt!!!
    
    ### TESTING IS ON
    if testing is True:
    
        ## DEFINING GRO PATH
        path_gro=os.path.join( r"R:/scratch/nanoparticle_project/simulations/HYDROPHOBICITY_PROJECT_C11/EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1/NVT_grid_1",
                              "sam_prod.gro"
                              )
        
        ## DEFINING PATH TO PRINT
        path_summary = os.path.join( r"R:/scratch/nanoparticle_project/simulations/HYDROPHOBICITY_PROJECT_C11/EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1/NVT_grid_1",
                                      "com_output.summary"
                                      )
        
        ## DEFINING RESIDUE NAME LIST
        residue_list = [ "AUNP" ]
        
    else:
        ### DEFINING PARSER OPTIONSn
        # Adding options for command line input (e.g. --ligx, etc.)
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## INPUT GRO FILE
        parser.add_option("--igro", dest="path_gro", action="store", type="string", help="Path to gro file", default=".")        
        
        ## OUTPUT SUMMARY FILE
        parser.add_option("--osum", dest="path_summary", action="store", type="string", help="Path to summary file", default=".")        
        parser.add_option("-n", "--names", dest="ligand_names", action="callback", type="string", callback=get_ligand_args,
                  help="Name of ligand molecules to be loaded from ligands folder. For multiple ligands, separate each ligand name by comma (no whitespace)")
        
        
    
    ## IMPORTING
    gro_file = read_write_tools.extract_gro(path_gro)
    
    ## DEFINIING RESIDUE NAME
    residue_name = residue_list[0]
    
    ## CREATING DICTIONARY
    com_dict = {}
    
    ## LOOPING THROUGH EACH RESIDUE NAME
    for residue_name in residue_list:
    
        ## EXTRACTING ALL INDICES
        index_list = [ idx for idx, resname in enumerate(gro_file.ResidueName) 
                                                    if resname == residue_name]
        
        ## GETTING ATOM NAMES
        atomic_masses = [ assign_mass(gro_file.AtomName[idx]) for idx in index_list ]
        
        ## GETTING COORDINATES
        xCoords = np.array(gro_file.xCoord)[index_list]
        yCoords = np.array(gro_file.yCoord)[index_list]
        zCoords = np.array(gro_file.zCoord)[index_list]
        
        ## DEFINING CENTER OF MASS
        center_of_mass = calc_com(masses = atomic_masses,
                                  xcoords = xCoords,
                                  ycoords = yCoords,
                                  zcoords = zCoords,
                                  )
        ## STORING
        com_dict[residue_name] = center_of_mass
    
    ## PRINTING
    print("Writing summary file at: %s"%(path_summary) )
    ### FUNCTION TO PRINT OUT SUMMARY
    with open(path_summary, 'w') as summary_file:
        ## PRINTING EACH 
        for each_key in com_dict:
            summary_file.write("%s %.8f %.8f %.8f\n"%( each_key,
                                                     com_dict[each_key][0],
                                                     com_dict[each_key][1],
                                                     com_dict[each_key][2],
                                                     ) )
            # OUTPUTS: AUNP 3.39819 3.37668 3.37450

            