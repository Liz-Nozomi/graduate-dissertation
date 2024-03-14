# -*- coding: utf-8 -*-
"""
plot_density.py
The purpose of this code is to plot the density along an axis. It will read 
gmx density codes, then generate a plot out of it. 

Author: Alex K. Chew (01/07/2020)

"""
## IMPORTING IMPORT MODULES
import os
import sys # Used to import arguments
import numpy as np
import matplotlib

## IMPORT MATPLOTLIB
if sys.prefix == '/usr':
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    plt.rcParams['agg.path.chunksize'] = 10000 # Had to apply this to work on the server
else:
    import matplotlib.pyplot as plt

## CHECKING TOOLS
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

## IMPORTING FROM RDF TOOLS
from MDDescriptors.traj_tools.plot_rdf import get_xy_labels, get_legend_names, main

### FUNCTION TO GET THE DATA
def get_xvg_numerical_data( data ):
    '''
    The purpose of this function is to extract the RDF values from gmx 
    rdf function. 
    INPUTS:
        data: [list]
            list of the data file
    OUTPUTS
        Extract_data: [np.array, float]
            extracted data as float
    '''
    ## LOCATING FINAL INDICES
    Final_indices= [i for i, j in enumerate(data) if j.startswith('@') or j.startswith('#')][-1]
    
    ## EXTRACTING THE RDF DATA
    Extract_data = np.array([ x.split() for x in data[Final_indices+1:] ]).astype('float')
    
    return Extract_data

###################################################
### CLASS FUNCTION TO EXTRACT DATA FROM DENSITY ###
###################################################
class extract_gmx_density:
    '''
    The purpose of this function is to extract gmd density data
    INPUTS:
        data: [list]
            list of the data extracted
    OUTPUTS:
    '''
    def __init__(self, data):
        
        ## GETTING X AND Y LABELS
        self.x_label, self.y_label = get_xy_labels(data = data)
        
        ## DEFINING EXTRACTED DATA
        self.Extract_data = get_xvg_numerical_data(data)
        
        ## GETTING LEGEND
        self.combined_name_list = get_legend_names(data)
        
        return

#%% MAIN SCRIPT
if __name__=="__main__":
    
    # Testing parameters
    Testing = check_testing() # False if you are running this program on command line
    
    ## DEFAULT VARIABLES
    separated = True
    
    ## RUNNING TESTING
    if Testing == True:
        
        ## DEFINING FULL PATH
        path_to_directory = r"R:\scratch\nanoparticle_project\prep_files\lipid_bilayers\DOPC-300.00-196_196\gromacs\analysis"
        
        ## DEFINING RDF XVG
        xvg_list = ["head_group.xvg",
                    "lipid_membrane.xvg",
                    "water.xvg"
                    ]
        
        ## DEFINING NAME OF IMAGE
        output_image_name="density.png"
    

    ## RUNNING MAIN FUNCTION
    main(path_to_directory = path_to_directory, 
         rdf_xvg_list = xvg_list, 
         output_image_name = output_image_name,
         separated = separated,
         extraction_function = extract_gmx_density,
         want_line_at_y_1 = False,
         )


        