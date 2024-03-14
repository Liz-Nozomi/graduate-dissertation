# -*- coding: utf-8 -*-
"""
plot_self_assembly_structure
The purpose of this function is to plot the self-assembly structure

USAGE: from plot_self_assembly_structure import plot_self_assembly_structure

Created on: 03/22/2018

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING MODULES
import matplotlib.pyplot as plt
# from MDDescriptors.core.plot_tools import save_fig_png # code to save figures
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new


### DEFINING GLOBAL PLOTTING VARIABLES
FONT_SIZE=16
FONT_NAME="Arial"    

### DEFINING LINE STYLE
LINE_STYLE={
        "linewidth": 1.4, # width of lines
        }

####################################################
### CLASS FUNCTION TO PLOT STRUCTURAL PROPERTIES ###
####################################################
class plot_self_assembly_structure:
    '''
    The purpose of this function is to plot the self assembly structure
    INPUTS:
        structure: class from self_assembly_structure
        fig: figure (will plot within this figure!) [ Default = None ]
        ax: axis for the figure [ Default = None ]
        line_type: type of the line
        line_color: color of the line
        line_label: Line label
    FUNCTIONS:
        create_plot_num_ligands_per_frame: Creates a plot of number of adsorbed ligands per time frame
        plot_num_ligand_per_frame: plots number of ligands per frame
    '''
    ### INITIALIZATION
    def __init__(self, structure, fig = None, ax = None, label=None, current_line_style={}):
        ## STORING STRUCTURE
        self.structure = structure
        ## STORING FIGURE DETAILS
        self.fig = fig
        self.ax = ax
        ## LINE DETAILS
        self.line_label = label
        self.current_line_style = current_line_style
        
        if fig is None or ax is None:
            ## CREATE FIGRUE
            self.fig, self.ax = self.create_plot_num_ligands_per_frame()
                    
        ## PLOTTING
        self.plot_num_ligand_per_frame()
        
        return
        
    ### FUNCTION TO CREATE SELF_ASSEMBLY_PLOT
    def create_plot_num_ligands_per_frame(self):
        '''
        The purpose of this function is to create a figure of number of ligand per frame
        INPUTS:
            self: class property
        OUTPUTS:
            fig, ax: figure and axis for plot
        '''
        ## CREATING PLOT
        fig = plt.figure() 
        ax = fig.add_subplot(111)
    
        ## DRAWING LABELS
        ax.set_xlabel('Simulation time (ns)',fontname=FONT_NAME,fontsize = FONT_SIZE)
        ax.set_ylabel('Number of adsorbed thiols',fontname=FONT_NAME,fontsize = FONT_SIZE)
        
        return fig, ax
    
    ### FUNCTION TO PLOT SELF-ASSEMBLY
    def plot_num_ligand_per_frame(self):
        '''
        The purpose of this function is to plot number of ligand per frame
        INPUTS:
            self: class property
        OUTPUTS:
            number of ligands adsorbed per simulation time
        '''            
        ## PLOTTING
        self.ax.plot(self.structure.frames_ns, self.structure.num_gold_sulfur_bonding_per_frame,
                     label = self.line_label,**self.current_line_style ) # , **LINE_STYLE}
        
#######################################################
### CLASS FUNCTION TO EXTRACT STRUCTURAL PROPERTIES ###
#######################################################
class extract_self_assembly_structure:
    '''
    The purpose of this class is to extract self-assembly structure
    INPUTS:
        structure: class from self_assembly_structure
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info:
            num_ligands_per_frame: number of ligands (y) per frame (x)
            ligand_density_area_angs_per_ligand: ligand density in Angstroms^2/ligand
            final_number_adsorbed_ligands: final number of adsorbed ligands
    '''
    ### INITIALIZATION
    def __init__(self, structure, pickle_name):
        ## STORING STRUCTURE
        self.structure = structure
        
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
    
        ## FINDING SPECIFIC INFORMATION ABOUT SYSTEM
        self.find_num_ligand_per_frame()
        
        ## FINDING LIGAND DENSITY
        self.find_ligand_density()
    
    ### FUNCTION TO FIND NUMBER OF LIGANDS PER FRAME
    def find_num_ligand_per_frame(self, title='num_ligands_per_frame'):
        '''
        The purpose of this function is to find the number of ligands per frame:
            num_ligands_per_frame: number of ligands (y) per frame (x)
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## DEFINING INPUTS
        x = self.structure.frames_ns
        y = self.structure.num_gold_sulfur_bonding_per_frame
        ## FINDING LABELS
        labels = [ 'Simulation time (ns)', 'Number of adsorbed thiols' ]
        ## STORING THE DATA
        self.csv_info = csv_info_add(self.csv_info, title, [x,y], labels )
        return
        
    ### FUNCTION TO FIND LIGAND DENSITY
    def find_ligand_density(self):
        '''
        The purpose of this function is to find the ligand density:
            ligand_density_area_angs_per_ligand: ligand density in Angstroms^2/ligand
            final_number_adsorbed_ligands: final number of adsorbed ligands
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'ligand_density_area_angs_per_ligand', data = [self.structure.area_angs_per_ligand] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'final_number_adsorbed_ligands', data = [self.structure.num_gold_sulfur_bonding_per_frame[-1]] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'final_diameter(nm)', data = [self.structure.gold_diameter] )
        
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    structure_plot = plot_self_assembly_structure(structure)