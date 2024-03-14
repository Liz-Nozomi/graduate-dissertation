# -*- coding: utf-8 -*-
"""
extract_lipid_bilayer_info.py
The purpose of this function is to extract lipid bilayer information.

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

## SCIPY TO FIND PEAKS
from scipy.signal import find_peaks

## IMPORTING FROM RDF TOOLS
from MDDescriptors.traj_tools.plot_rdf import main
from MDDescriptors.traj_tools.plot_density import extract_gmx_density

## CHECKING TOOLS
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

## IMPORTING READ XVG
from MDDescriptors.core.read_write_tools import read_xvg

## LIGAND BUILDER
from MDBuilder.builder.ligand_builder import get_ligand_args

## DEFINING FUNCTION TO EXTRACT LIPID MEMBRANE INFORMATION
def extract_lipid_membrane_info_from_density(path_to_directory,
                                             xvg_list,
                                             output_image_name,
                                             output_text_file,
                                             ):
    '''
    The purpose of this function is to extract lipid membrane information 
    given that gmx density was performed. 
    INPUTS:
        path_to_directory: [str]
            path to the directory of interest
        xvg_list: [list]
            list of xvg directories. Not that we assume head group xvg is 
            "head_group.xvg". This function looks for that file to compute 
            lipid membrane thickness
        output_image_name: [str]
            output image name, stored in path_to_directory
        output_text_file: [str]
            output text file name, stored in path_to_directory
            
    OUTPUTS:
        void. This function outputs images
    '''
    ## RUNNING MAIN FUNCTION
    fig, ax, rdf_dict = \
        main(path_to_directory = path_to_directory, 
             rdf_xvg_list = xvg_list, 
             output_image_name = output_image_name,
             separated = separated,
             extraction_function = extract_gmx_density,
             want_line_at_y_1 = False,
             store_image = False,
             )
    
    ## DEFINING FIGURE PATH
    fig_path = os.path.join(path_to_directory, output_image_name)
    fig_name, fig_extension = os.path.splitext(output_image_name)
    
    ## SAVING FIGURE
    fig.savefig(fig_path, format=fig_extension[1:], dpi=600)    
    
    ## GETTING HEAD NAME INDEX
    head_name_xvg = xvg_list[xvg_list.index("head_group.xvg")]
    
    ## EXTRACTING HEAD NAME XVG
    head_name_xvg_data = read_xvg(os.path.join(path_to_directory,
                                               head_name_xvg))
    
    ## GETTING HEAD NAME INDEX
    extracted_head_name = extract_gmx_density(data = head_name_xvg_data.xvg_lines).combined_name_list[0]
    
    ## EXTRACTING THE DATA
    data_head_name = rdf_dict[extracted_head_name]
    
    ## GETTING INDEX FOR PLOT
    head_name_plot_index = list(rdf_dict.keys()).index(extracted_head_name)
    
    ## FINDING PEAKS
    peaks, _ = find_peaks(data_head_name['y'])
    
    ## NOTE IF LENGTH IS GREATER THAN 2
    if len(peaks) > 2:
        print("Note, we find more than one peak for the membrane!")
        print("This should not cause major errors, just check the densities to ensure correctness")
        print("We will take the maximum and minimum peak")
        
    ## GETTING ALL PEAK INFORMATION
    peak_info = data_head_name['x'][peaks]
    
    ## MEMBRANE THICKNESS
    membrane_thickness = peak_info.max() - peak_info.min()
    
    ## GETTING APPROXIMATE Z VALUE
    membrane_center = np.mean((peak_info.max(),peak_info.min()))

    ## ADDING PEAKS TO PLOT
    ax[head_name_plot_index].plot(data_head_name['x'][peaks], data_head_name['y'][peaks], "x", color = 'k', label="Peaks")
    ## ADDING THE CENTER    
    for each_axis in ax:
        each_axis.axvline(x = membrane_center, color ='k', linestyle='--', label="Center")
        
    ## UPDATING LEGEND
    ax[head_name_plot_index].legend(loc = 'upper right')
    
    ## DEFINING TEXT
    plot_text="Thickness: %.2f nm\n Center: %.3f nm"%(membrane_thickness, membrane_center)
    
    ## ADDING TEXT TO FIGURE
    plt.text(0.0, 0.8, plot_text,
              ha='left', va='center',
              transform=ax[head_name_plot_index].transAxes)
    
    ## STORING IMAGE
    fig.savefig(fig_path, format=fig_extension[1:], dpi=600)    
    ## PRINTING
    print("Storing image in %s"%(fig_path))
    
    ## PRINTING THICKNESS
    print("-------------------------------------------------")
    print("Membrane center position: %.3f"%(membrane_center))
    print("Membrane thickness: %.2f nm"%(membrane_thickness))
    print("1/2 thickness: %.2f nm"%(membrane_thickness/2))
    
    ## DEFINING PATH TO TEXT
    path_to_text = os.path.join(path_to_directory,
                                output_text_file)
    
    ## ADDING TO TEXT FILE
    with open(path_to_text, 'w') as text_file:
        text_file.write("Membrane_center_position_nm %.5f\n"%( membrane_center ) )
        text_file.write("Membrane_thickness %.5f\n"%( membrane_thickness ) )
        text_file.write("Membrane_half_thickness %.5f\n"%( membrane_thickness/2 ) )
    
    ## PRINTING
    print("Writing text file to %s"%(path_to_text))
    
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
        
        ## DEFINING OUTPUT INFORMATION
        output_text_file="lipid_membrane_info.txt"
    
    else:
        # Importing parser to allow inputs
        from optparse import OptionParser # Used to allow commands within command line
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
    
        ## PATH INFORMATION
        parser.add_option('--path', dest = 'path_to_directory', help = 'Full input path to files', default = '.')

        ## XVG LIST
        parser.add_option("--xvg", dest="xvg_list", action="callback", type="string", callback=get_ligand_args,
                  help="Xvg files, separate each by comma (no whitespace)", default = None)
        
        ## IMAGE FILE INFORMATION
        parser.add_option('--image', dest = 'output_image_name', help = 'Output image that is stored into path', default = "density.png")
        
        ## TEXT FILE INFORMATION
        parser.add_option('--text', dest = 'output_text_file', help = 'Output text that is stored into path', default = "lipid_membrane_info.txt")        
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## DEFINING VARIABLES
        path_to_directory = options.path_to_directory
        xvg_list = options.xvg_list
        output_image_name = options.output_image_name
        output_text_file = options.output_text_file
        
    ## RUNNING MAIN FUNCTION
    extract_lipid_membrane_info_from_density(path_to_directory,
                                                 xvg_list,
                                                 output_image_name,
                                                 output_text_file,
                                                 )
    
    
    
    
        
    
    
    
    
    
    
    