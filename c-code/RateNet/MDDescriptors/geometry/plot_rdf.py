# -*- coding: utf-8 -*-
"""
plot_rdf.py
The purpose of this script is to take the rdf functions from rdf.py and plot them
    
USAGE: from plot_rdf import plot_rdf

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""

### DEFINING GLOBAL PLOTTING VARIABLES
FONT_SIZE=16
FONT_NAME="Arial"    

### DEFINING COLOR LIST
COLOR_LIST=['k','b','r','g','m','y','k','w']

### DEFINING LINE STYLE
LINE_STYLE={
            "linewidth": 1.4, # width of lines
            }

### DEFINING SAVING STYLE
DPI_LEVEL=600

### CHANGING STYLE
#    import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = [3.0, 3.0]
#    mpl.rcParams['figure.dpi'] = 300
#    mpl.rcParams['savefig.dpi'] = 600
#    mpl.rcParams['font.size'] = 12
#    mpl.rcParams['legend.fontsize'] = 'small'
#    mpl.rcParams['figure.titlesize'] = 'medium'

### IMPORTING MODULES
import matplotlib.pyplot as plt
from MDDescriptors.core.plot_tools import save_fig_png # code to save figures


################################################################
### CLASS FUNCTION TO PLOTTING RADIAL DISTRIBUTION FUNCTIONS ###
################################################################
class plot_rdf:
    '''
    The purpose of this class is to take the calc_rdf class and plot it accordingly
    INPUTS:
        rdf: rdf class from calc_rdf
    OUTPUTS:
        
        
    FUNCTIONS:
        plot_rdf_solute_solvent: plots rdf between solute and solvent
        plot_rdf_solute_oxygen_to_solvent: plots rdf between solute-water and solvent
    '''
    ### INITIALIZATION
    def __init__(self, rdf, ):
        ## STORING INPUT
        self.rdf = rdf
        ## PLOTTING RDFS OF SOLUTE TO SOLVENT
        # self.plot_rdf_solute_solvent()
        return
    
    ### FUNCTION TO CREATE RDF PLOT
    def create_rdf_plot(self):
        '''
        The purpose of this function is to generate a figure for you to add your RDFs.
        Inputs:
            fontSize: Size of font for x and y labels
            fontName: Name of the font
        Output:
            fig: Figure to print
            ax: Axes to plot on
        '''
        ## CREATING PLOT
        fig = plt.figure() 
        ax = fig.add_subplot(111)
    
        ## DRAWING LABELS
        ax.set_xlabel('r (nm)',fontname=FONT_NAME,fontsize = FONT_SIZE)
        ax.set_ylabel('Radial Distribution Function',fontname=FONT_NAME,fontsize = FONT_SIZE)
        
        # Drawing ideal gas line
        ax.axhline(y=1, linewidth=1, color='black', linestyle='--')
        
        return fig, ax
    
    ### FUNCTION TO PLOT RDF OF SOLUTE AND SOLVENT
    def plot_rdf_solute_solvent(self, save_fig=False):
        '''
        The purpose of this script is to plot radial distribution function for each solvent
        INPUTS:
            self: class property
            save_fig: True if you want to save all the figures [Default: False]
        OUTPUTS:
            multiple figures 
        '''
        ## CREATING RDF PLOT
        fig, ax = self.create_rdf_plot()
        
        ## LOOPING THROUGH EACH SOLVENT
        for each_solvent in range(len(self.rdf.solvent_name)):
            ## PLOTTING RDF
            ax.plot(self.rdf.rdf_r[each_solvent], self.rdf.rdf_g_r[each_solvent], '-', color = COLOR_LIST[each_solvent],
                    label= "%s --- %s"%(self.rdf.solute_name, self.rdf.solvent_name[each_solvent]),
                    **LINE_STYLE)
            ax.legend()
            
        ## SAVING IF NECESSARY
        label = "RDF_%s_%s"%(self.rdf.solute_name, '_'.join(self.rdf.solvent_name))
        save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)

        return
    
    ### FUNCTION TO PLOT RDF OF SOLUTE-OXY TO SOLVENT
    def plot_rdf_solute_oxygen_to_solvent(self, save_fig=False):
        '''
        The purpose of this function is to plot the solute oxygen to solvent radial distribution functions
        INPUTS:
            self: class property
            save_fig: True if you want to save all the figures [Default: False]
        OUTPUTS:
            rdf vs r for each oxygen
        '''
        ## CHECKING IF THE RDF DOES HAVE OXYGEN DETAILS
        if self.rdf.want_oxy_rdf is True:
            ## LOOPING THROUGH EACH ATOM NAME
            for oxy_index, each_oxygen in enumerate(self.rdf.rdf_oxy_names):
                ## CREATING RDF PLOT
                fig, ax = self.create_rdf_plot()
                ## EXTRACTING DATA SINCE IT IS [COSOLVENT] THEN [OXYGEN]
                current_oxy_r =[ self.rdf.rdf_oxy_r[solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                current_oxy_g_r =[ self.rdf.rdf_oxy_g_r[solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                ## LOOPING THROUGH EACH SOLVENT
                for each_solvent in range(len(self.rdf.solvent_name)):
                    ## PLOTTING RDF
                    ax.plot(current_oxy_r[each_solvent], current_oxy_g_r[each_solvent], '-', color = COLOR_LIST[each_solvent],
                            label= "%s-%s --- %s"%(self.rdf.solute_name, each_oxygen, self.rdf.solvent_name[each_solvent]),
                            **LINE_STYLE)
                    ax.legend()
                    
                ## SAVING IF NECESSARY
                label = "RDF_%s_%s_%s"%(self.rdf.solute_name, each_oxygen,'_'.join(self.rdf.solvent_name))
                save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
        else:
            print("There is no oxygen rdf information, want_oxy_rdf is set to: %s"%(self.rdf.want_oxy_rdf))
        
        
    
#%% MAIN SCRIPT
if __name__ == "__main__":
    ## PLOTTING RDF
    plotting_rdf = plot_rdf(rdf = rdf)
