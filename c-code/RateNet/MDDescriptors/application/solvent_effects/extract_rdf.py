# -*- coding: utf-8 -*-
"""
extract_rdf.py
The purpose of this script is to analyze the data from the multi_traj_analysis_tool.py for RDFs. This script also contains code to plot the rdfs

AUTHOR(S)
    Alex K. Chew (alexkchew@gmail.com)
    
FUNCTIONS:
    calc_cumulative_dist_function: calculates the cumulative distribution function given a cutoff
    find_first_solvation_shell_rdf: finds the first solvation shell of an RDF
    
"""
### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, load_multi_traj_pickles, find_multi_traj_pickle
### IMPORTING PLOTTING FUNCTIONS
# from MDDescriptors.geometry.plot_rdf import plot_rdf
## IMPORTING DECODER FUNCTIONS
from MDDescriptors.core.decoder import decode_name
## PLOTTING FUNCTIONS
import matplotlib.pyplot as plt
from MDDescriptors.core.plot_tools import save_fig_png # code to save figures
## DEFAULT PLOTTING STYLES
from MDDescriptors.global_vars.plotting_global_vars import FONT_SIZE, FONT_NAME, COLOR_LIST, LINE_STYLE, DPI_LEVEL
from MDDescriptors.core.plot_tools import get_cmap
## SYSTEM TOOLS
import sys
## MATH TOOLS
import numpy as np
## CSV TOOLS
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add
import MDDescriptors.core.calc_tools as calc_tools
## IMPORTING SIGNALLING TOOLS USED FOR FITTING THE DATA
from scipy import signal

### FUNCTION TO CREATE RDF PLOT
def create_rdf_plot(fontname = FONT_NAME, fontsize = FONT_SIZE):
    '''
    The purpose of this function is to generate a figure for you to add your RDFs.
    INPUTS:
        fontsize: [float]
            Size of font for x and y labels
        fontname: [float]
            Name of the font
    OUTPUTS:
        fig: [object]
            Figure to print
        ax: [object]
            Axes to plot on
    ## TO USE: ax.plot(x, y)
    '''
    ## CREATING PLOT
    fig = plt.figure() 
    ax = fig.add_subplot(111)

    ## DRAWING LABELS
    ax.set_xlabel('r (nm)',fontname=fontname,fontsize = fontsize)
    ax.set_ylabel('Radial Distribution Function',fontname=fontname,fontsize = fontsize)
    
    # Drawing ideal gas line
    ax.axhline(y=1, linewidth=1, color='black', linestyle='--')
    
    return fig, ax


### FUNCTION TO FIND THE CUTOFF RADIUS
def calc_cutoff_radius(g_r, r, tolerance = 0.015):
    '''
    The purpose of this function is to get the cutoff radius based off a radial distribution function.
    INPUTS:
        g_r: [np.array] 
            g(r) for radial distribution function
        r: [np.array] 
            r for RDF, in nms
        tolerance: [float, default=0.015]
            tolerance for when to find the equilibrium point
    OUTPUTS:
        equil_radius: [float] radius in nm of the cutoff
    '''
    ## GETTING EQUILIBRIUM POINT
    equil_index = calc_tools.find_equilibrium_point(g_r,tolerance)
    equil_radius = r[equil_index]
    return equil_radius

### FUNCTION TO FIND THE COORDINATION NUMBER GIVEN A CUTOFF
def calc_cumulative_dist_function( density, g_r, r, bin_width, r_cutoff = None, verbose = False):
    '''
    The purpose of this function is to take your radial distribution function and get the coordination number at a given cutoff. 
    Cumulative distribution function is defined as:

        integral_0^inf rho * g(r) * 4 * pi * r^2 dr
        where:
            rho is the number density of the system
            g(r) is the radial distribution function
            r is the radius vector
    INPUTS:
        density: [float] Number density of your solvent (number/nm^3) (Number of atoms / ensemble volume)
        g_r: [np.array] radial distribution function
        r: [np.array] radius vector for g_r
        bin_width: [float] Bin width for your RDFs
        verbose: [logical, default=False]
            True if you want output printed
    OUTPUTS:
        cdf: [float] Coordination number, or number of atoms within a cutoff r
    '''
    ## FINDING INDEX OF THE CUTOFF
    if r_cutoff is None: # No cutoff, assuming entire radius vector
        index_within_cutoff = np.where( r <= np.max(r) )
    else:
        index_within_cutoff = np.where( r <= r_cutoff )
    ## PRINTING
    if verbose is True:
        print("CALCULATING CDF FOR CUTOFF RADIUS: %.2f"%(r[index_within_cutoff][-1])   )
    ## USING CUTOFF
    g_r_within_cutoff = g_r[index_within_cutoff]
    r_within_cutoff = r[index_within_cutoff]
    ## DEFINING CDF
    CDF_Integral_Function=density * g_r_within_cutoff * 4 * np.pi * r_within_cutoff * r_within_cutoff
    ## INTEGRATING TO GET THE CDF
    cdf= np.trapz(CDF_Integral_Function,r_within_cutoff,dx=bin_width)
    if verbose is True:
        print("CDF found: %.3f"%(cdf) )
    return cdf

### FUNCTION TO FIND THE INDEX OF THE FIRST SOLVATION SHELL
def find_first_solvation_shell_rdf( g_r, r ):
    '''
    The purpose of this function is to find the first solvation shell for a radial distribution function
    INPUTS:
        g_r: [np.array] radial distribution function
        r: [np.array] radius vector for g_r
    OUTPUTS:
        first_solv_shell: [dict] contains all information about the first solvation shell
            'max':
                'g_r': [float] value of g(r) at the first maximum
                'r': [float] value of r at the first maximum
                'index': [int] index value of maximum
            'min':
                'g_r': [float] value of g(r) at the first minimum
                'r': [float] value of r at the first minimum
                'index': [int] index value of minimum
    '''
    ## START BY LOOPING AND FINDING THE FIRST MAXIMUM
    current_index = 0 ## INITIAL INDEX
    current_g_r_value = g_r[current_index]
    current_r_value = r[current_index]
    
    ## ADDING ONE TO INDEX
    current_index += 1
    
    ## USING WHILE LOOP TO FIND THE FIRST MAXIMUM
    while g_r[current_index] >= current_g_r_value:
        ## ADDING TO THE LOOP
        current_g_r_value = g_r[current_index]
        current_r_value = r[current_index]
        current_index += 1
    
    ## CORRECTING FOR CURRENT INDEX (Going too far)
    current_index -= 1
    
    ## STORING TO DATABASE 
    first_solv_shell = {'max':{ 'r'     : current_r_value,
                                'g_r'   : current_g_r_value,
                                'index' :  current_index,
                                } }
    
    ## AFTER FINDING THE FIRST MAXIMUM, WE NEED TO FIND THE MINIMUM
    while g_r[current_index] <= current_g_r_value:
        ## ADDING TO THE LOOP
        current_g_r_value = g_r[current_index]
        current_r_value = r[current_index]
        current_index += 1
        
    ## CORRECTING FOR CURRENT INDEX
    current_index -= 1
    
    ## STORING TO DATABASE 
    first_solv_shell['min'] = { 'r'     : current_r_value,
                                'g_r'   : current_g_r_value,
                                'index' :  current_index,
                                }
    
    return first_solv_shell

### FUNCTION TO FIND THE INDEX OF THE FIRST SOLVATION SHELL
def find_first_solvation_shell_rdf_absolute_minimum( g_r, r ):
    '''
    The purpose of this function is to find the first solvation shell for a radial distribution function
    INPUTS:
        g_r: [np.array] radial distribution function
        r: [np.array] radius vector for g_r
    OUTPUTS:
        first_solv_shell: [dict] contains all information about the first solvation shell
            'max':
                'g_r': [float] value of g(r) at the first maximum
                'r': [float] value of r at the first maximum
                'index': [int] index value of maximum
            'min':
                'g_r': [float] value of g(r) at the first minimum
                'r': [float] value of r at the first minimum
                'index': [int] index value of minimum
    '''
    ## START BY LOOPING AND FINDING THE FIRST MAXIMUM
    current_index = 0 ## INITIAL INDEX
    current_g_r_value = g_r[current_index]
    current_r_value = r[current_index]
    
    ## ADDING ONE TO INDEX
    current_index += 1
    
    ## USING WHILE LOOP TO FIND THE FIRST MAXIMUM
    while g_r[current_index] >= current_g_r_value:
        ## ADDING TO THE LOOP
        current_g_r_value = g_r[current_index]
        current_r_value = r[current_index]
        current_index += 1
    
    ## CORRECTING FOR CURRENT INDEX (Going too far)
    current_index -= 1
    
    ## STORING TO DATABASE 
    first_solv_shell = {'max':{ 'r'     : current_r_value,
                                'g_r'   : current_g_r_value,
                                'index' :  current_index,
                                } }
    
    ## AFTER FINDING THE FIRST MAXIMUM, WE NEED TO FIND THE MINIMUM
    ## HERE, WE WILL SIMPLY USE THE NUMPY FUNCTION TO GET THE ARGUMENTS
    ## DEFINING NEW G_R
    g_r_after_max = g_r[first_solv_shell['max']['index']:]
    
    ## FINDING INDEX
    local_minima_g_r_index = signal.argrelextrema( g_r_after_max , np.less )
    
    ## USING SCIPY TO FIND LOCAL MINIMA
    # local_minima_g_r = g_r_after_max[ local_minima_g_r_index ][0] # First index
    ## FINDING RADIUS
    local_minima_r = r[first_solv_shell['max']['index']:][local_minima_g_r_index][0] # First index
    # r_after_max = r[first_solv_shell['max']['index']:]
    # min_index = np.where( g_r == np.min(g_r_after_max))
    min_index = np.where( r == local_minima_r)[0][0] ## ESCAPING TUPLE AND LIST
    ## STORING TO DATABASE 
    first_solv_shell['min'] = { 'r'     : r[min_index],
                                'g_r'   : g_r[min_index],
                                'index' :  min_index,
                                }
    
    return first_solv_shell

### FUNCTION TO FIND THE INDEX OF THE FIRST SOLVATION SHELL
def find_first_solvation_shell_rdf_argrelextrama( g_r, r, upper_order=3 ):
    '''
    The purpose of this function is to find the first solvation shell for a radial distribution function
    INPUTS:
        g_r: [np.array] radial distribution function
        r: [np.array] radius vector for g_r
    OUTPUTS:
        first_solv_shell: [dict] contains all information about the first solvation shell
            'max':
                'g_r': [float] value of g(r) at the first maximum
                'r': [float] value of r at the first maximum
                'index': [int] index value of maximum
            'min':
                'g_r': [float] value of g(r) at the first minimum
                'r': [float] value of r at the first minimum
                'index': [int] index value of minimum
    '''
    ## FINDING INDEX
    local_max_g_r_index = signal.argrelextrema( g_r , np.greater , order = upper_order ) # 
    
    ## FINDING RADIUS
    local_max_r_index = r[local_max_g_r_index][0]
    
    ## FINDING MAX INDEX
    max_index = np.where( r == local_max_r_index)[0][0] ## ESCAPING TUPLE AND LIST
        
    ## STORING TO DATABASE 
    first_solv_shell = {'max':{ 'r'     : r[max_index],
                                'g_r'   : g_r[max_index],
                                'index' :  max_index,
                                } }
    
    ## AFTER FINDING THE FIRST MAXIMUM, WE NEED TO FIND THE MINIMUM
    ## HERE, WE WILL SIMPLY USE THE NUMPY FUNCTION TO GET THE ARGUMENTS
    ## DEFINING NEW G_R
    g_r_after_max = g_r[first_solv_shell['max']['index']:]
    
    ## FINDING INDEX
    local_minima_g_r_index = signal.argrelextrema( g_r_after_max , np.less )
    
    ## USING SCIPY TO FIND LOCAL MINIMA
    # local_minima_g_r = g_r_after_max[ local_minima_g_r_index ][0] # First index
    ## FINDING RADIUS
    local_minima_r = r[first_solv_shell['max']['index']:][local_minima_g_r_index][0] # First index
    # r_after_max = r[first_solv_shell['max']['index']:]
    # min_index = np.where( g_r == np.min(g_r_after_max))
    min_index = np.where( r == local_minima_r)[0][0] ## ESCAPING TUPLE AND LIST
    ## STORING TO DATABASE 
    first_solv_shell['min'] = { 'r'     : r[min_index],
                                'g_r'   : g_r[min_index],
                                'index' :  min_index,
                                }
    
    return first_solv_shell


### FUNCTION TO CALCULATE THE FIRST COORDINATION NUMBER
def calc_first_coord_num(density, g_r, r, bin_width):
    '''
    The purpose of this function is to calculate the first coordination number based on cumulative distribution function
        Cumulative distribution function is defined as:
    
            integral_0^inf rho * g(r) * 4 * pi * r^2 dr
            where:
                rho is the number density of the system
                g(r) is the radial distribution function
                r is the radius vector
    NOTE: This function will use the first coordination number to calculate the cdf
    INPUTS:
        density: [float] 
            Number density of your solvent (number/nm^3) (Number of atoms / ensemble volume)
        g_r: [np.array] 
            radial distribution function
        r: [np.array] 
            radius vector for g_r
        bin_width: [float]
            bin width used to find the radius vector
    OUTPUTS:
        cdf: [float]
            cumulative distribution function
        first_solv_shell: [float]
            first solvation shell information
    '''
    ## FINDING SOLVATION SHELL INFORMATION
    # first_solv_shell = find_first_solvation_shell_rdf( g_r =  g_r, r= r)
    # first_solv_shell = find_first_solvation_shell_rdf_absolute_minimum( g_r =  g_r, r= r)
    first_solv_shell = find_first_solvation_shell_rdf_argrelextrama( g_r =  g_r, r= r)
    ## DEFINING THE MINIMUM
    min_first_solvation_shell_r = first_solv_shell['min']['r']
    
    ## CALCULATING CDF
    cdf = calc_cumulative_dist_function( density, 
                                        g_r = g_r, 
                                        r = r, 
                                        bin_width = bin_width, 
                                        r_cutoff = min_first_solvation_shell_r)
    
    return cdf, first_solv_shell

### FUNCTION TO CALCULATE CUMULATIVE ISTRIBUTION FUNCTION USING FILTER
def calc_first_coord_num_filtered( density, g_r, r, bin_width, savgol_filter_params={'window_length':3, 'polyorder': 1, 'mode': 'nearest'}):
    '''
    The purpose of this function is to calculate the coordination number after running filtering
    INPUTS:
        density: [float] 
            Number density of your solvent (number/nm^3) (Number of atoms / ensemble volume)
        g_r: [np.array] 
            radial distribution function
        r: [np.array] 
            radius vector for g_r
        bin_width: [float]
            bin width used to find the radius vector
        savgol_filter_params: [dict]
            savgol filtering parameters
    OUTPUTS:
        cdf_filter: [float]
            cumulative distribution function that uses the filter
        min_first_solvation_shell_r_filter: [float]
            minimum for the first solvation of r that uses the filter
        g_r_filtered: [np.array]
            g_r after it has been filtered
    '''
    ## RUNNING FILTERING
    g_r_filtered = signal.savgol_filter(x=g_r, **savgol_filter_params)
    
    ## RUNNING FIRST COORDINATION NUMBER  CALCULATIONS
    cdf_filter, first_solv_shell = calc_first_coord_num( density = density,
                                                             g_r = g_r_filtered, 
                                                             r = r, 
                                                             bin_width = bin_width,
                                                            )
    return cdf_filter, first_solv_shell, g_r_filtered


################################################################
### CLASS FUNCTION TO PLOTTING RADIAL DISTRIBUTION FUNCTIONS ###
################################################################
class extract_rdf:
    '''
    The purpose of this class is to take the calc_rdf class and plot it accordingly
    INPUTS:
        rdf: rdf class from calc_rdf
        pickle_name: [str] directory name of the pickle
    OUTPUTS:
        
        
    FUNCTIONS:
        plot_rdf_solute_solvent: plots rdf between solute and solvent
        plot_rdf_solute_oxygen_to_solvent: plots rdf between solute-water and solvent
    '''
    ### INITIALIZATION
    def __init__(self, rdf, pickle_name, decoder_type = 'solvent_effects'):
        ## STORING INPUT
        self.rdf = rdf
        self.pickle_name = pickle_name
        self.decoded_name = decode_name(pickle_name, decoder_type )
        
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        
        ## EXTRACTION OF MOLE FRACTION INFORMATION
        self.extract_composition_of_water_vs_r()
        
        ## PLOTTING RDF
        self.plot_rdf_solute_solvent(save_fig = False, want_plot = False)
        
        ## CDF first solvation
        # self.extract_cdf_first_solvation()
#        ## PLOTTING SOLUTE SOLVENT
#        self.plot_rdf_solute_solvent_first_solvation(save_fig=False)
#        # self.plot_rdf_solute_solvent_frames(save_fig=True) # save_fig=True
         
#        want_plot = False
#        ## RDF FOR OXYGENS
#        # self.plot_rdf_solute_oxygen_to_solvent_largest_cutoff( save_fig = False, want_plot=want_plot)
#        self.plot_rdf_solute_oxygen_to_solvent(save_fig = True, want_first_solvation_shell = True, want_plot=want_plot)
#        ## PLOTTING RDFS OF SOLUTE TO SOLVENT
        
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
        
    ### FUNCTION TO PLOT RDF PER FRAME
    def plot_rdf_solute_solvent_frames(self, save_fig=False ):
        '''
        The purpose of this script is to plot the RDF for multiple solute and solvents
        INPUTS:
            self:
                class property
            save_fig: [logical, default=False]
                True if you want to save the final iamge
        OUTPUTS:
            fig, ax
        '''
        ## GENERATING COLOR MAP
        color_list=get_cmap(len(self.rdf.frames) )
        ## LOOPING THROUGH EACH SOLUTE
        for each_solute in range(self.rdf.num_solutes):
            ## DEFINING SOLUTE NAME
            solute_name=self.rdf.solute_name[each_solute]
            ## LOOPING THROUGH EACH SOLVENT
            for each_solvent in range(self.rdf.num_solvents):
                ## CREATING RDF PLOT
                fig, ax = self.create_rdf_plot()
                ## LOOPING THROUGH EACH FRAME
                for frame_idx, each_frame in enumerate(self.rdf.frames):
                    ## GETTING R AND GR
                    r = self.rdf.rdf_frames_r[each_solute][each_solvent][frame_idx][1]
                    g_r = self.rdf.rdf_frames_g_r[each_solute][each_solvent][frame_idx][1]
                    ## DEFINING COLOR
                    current_color= color_list(frame_idx)
                    ## CHANGING LAST FRAME COLOR TO BLACK
                    if frame_idx == len(self.rdf.frames)-1:
                        current_color = 'k'
                    ## PLOTTING RDF
                    ax.plot(r, g_r, '-', color = current_color,
                            label= "%s-%s (frame: %d)"%(solute_name, self.rdf.solvent_name[each_solvent], each_frame),
                            **LINE_STYLE) # color = COLOR_LIST[frame_idx],
                ## CREATING LEGEND
                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                # ax.legend(loc='best') # 'lower right'
        
                ## SAVING IF NECESSARY
                label = "Frame_RDF_%s-%s_%s"%(self.pickle_name, solute_name, self.rdf.solvent_name[each_solvent] )
                save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
        
        return

    
    ### FUNCTION TO PLOT RDF OF SOLUTE AND SOLVENT
    def plot_rdf_solute_solvent(self, save_fig=False, want_plot=True, want_radius_cutoff = False):
        '''
        The purpose of this script is to plot radial distribution function for each solvent
        INPUTS:
            self: class property
            save_fig: True if you want to save all the figures [Default: False]
            want_plot: [logical] True if you want to see a plot
            want_radius_cutoff: [logical, default=False]
                True if you want cutoff radius based on the maxima of the water
        OUTPUTS:
            multiple figures 
        '''
        ## LOOPING THROUGH EACH SOLUTE
        for each_solute in range(self.rdf.num_solutes):
            ## CREATING RDF PLOT
            fig, ax = self.create_rdf_plot()
            ## DEFINING SOLUTE NAME
            solute_name=self.rdf.solute_name[each_solute]
            ## LOOPING THROUGH EACH SOLVENT
            for each_solvent in range(len(self.rdf.solvent_name)):
                
                ## DEFINING R AND G_R
                r=self.rdf.rdf_r[each_solute][each_solvent]
                g_r=self.rdf.rdf_g_r[each_solute][each_solvent]
                
                if want_plot is True:
                    ## PLOTTING RDF
                    ax.plot(r, g_r, '-', color = COLOR_LIST[each_solvent],
                            label= "%s --- %s"%(solute_name, self.rdf.solvent_name[each_solvent]),
                            **LINE_STYLE)
                    ax.legend()
                
                ## STORING INFORMATION
                self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + self.rdf.solvent_name[each_solvent], 
                                             data = [ 
                                                     r,
                                                     g_r,
                                                     ],
                                             labels=['r(nm)', 'g_r']) # 'r(nm)',
                
                
                ## CALCULATING CUTOFF RADIUS
                if want_radius_cutoff == True and self.rdf.solvent_name[each_solvent] == 'HOH':
                    ## CALCULATING EQUIL RADIUS
                    equil_radius = calc_cutoff_radius( g_r = self.rdf.rdf_g_r[each_solute][each_solvent],
                                                       r   = self.rdf.rdf_r[each_solute][each_solvent],
                                                      )
                    ## STORING VARIABLE
                    self.equil_radius = equil_radius
                    ## ADDING TO PLOT
                    if want_plot is True:
                        ax.axvline( x = equil_radius, linestyle='--', color='k', linewidth=2, label='Cutoff radius' )
                    
                ## ADDING LEGEND
                if want_plot is True:
                    ax.legend()
                
            ## SAVING IF NECESSARY
            label = "RDF_%s_%s"%(solute_name, '_'.join(self.rdf.solvent_name))
            save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
        
        return fig, ax
    
    ### FUNCTION TO EXTRACT MOLE FRACTION CODE
    def extract_composition_of_water_vs_r( self, ):
        '''
        The purpose of this function is to extract the composition of water vs. r
        INPUTS:
            self: [object]
                class property
        OUTPUTS:
            
        '''
        ## RUNNING ONLY IF COMPOSITION IS AVAIALABLE
        if self.rdf.want_composition_with_r is True:
            ## DEFINING DESIRED INPUTS
            rdf_water_r = self.rdf.comp_r_vec
            rdf_water_comp = self.rdf.comp_water_mass_frac_by_r[0] # Escaping double array
        
            ## ADDING TO CSV INFO
            self.csv_info = csv_info_add(self.csv_info, data_title = 'water_comp_vs_r', 
                                         data = [ rdf_water_r, rdf_water_comp ], labels=['r(nm)', 'mass_water_vs_r'] )
        
        return
    
    ### FUNCTION TO EXTRACT THE CDFS PER FRAME BASIS
    def extract_cdf_first_solvation(self,savgol_filter_params={'window_length':3, 'polyorder': 1, 'mode': 'nearest'}):
        '''
        The purpose of this function is to run through each of the RDFs and store the cumulative distribution function. If your RDF has multiple frames, we will average across and take the standard deviation.
        INPUTS:
            self: [object]
                class property
        OUTPUTS:
            CDF values
        
        
        '''
        ## LOOPING THROUGH EACH SOLUTE
        for each_solute in range(self.rdf.num_solutes):
            ## LOOPING THROUGH EACH SOLVENT
            for each_solvent in range(len(self.rdf.solvent_name)):
                ## PRINTING
                print("WORKING ON SOLVENT: %s"%(self.rdf.solvent_name[each_solvent]))
                ## LOOPING THROUGH EACH ATOM NAME
                for oxy_index, each_oxygen in enumerate(self.rdf.rdf_oxy_names[each_solute]):
                    ## EXTRACTING DATA SINCE IT IS [COSOLVENT] THEN [OXYGEN]
                    current_oxy_r =[ self.rdf.rdf_oxy_r[each_solute][solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                    current_oxy_g_r =[ self.rdf.rdf_oxy_g_r[each_solute][solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                    ## EXTRACTING DATA FOR SPLITTING R BASIS
                    current_oxy_split_r = [  [self.rdf.rdf_split_data['OXY_RDF']['r'][each_solute][solvent_index][each_block][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))] 
                                            for each_block in range(self.rdf.split_rdf) ]
                    current_oxy_split_g_r = [  [self.rdf.rdf_split_data['OXY_RDF']['g_r'][each_solute][solvent_index][each_block][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))] 
                                            for each_block in range(self.rdf.split_rdf) ]
                    
                        
                    ## CALCULATING SOLVENT DENSITY
                    solvent_density = self.rdf.total_solvent[each_solvent] / self.rdf.volume
                    ## CALCULATING CDF OF THE ENTIRE THING
                    cdf_filter, first_solvation_shell_filtered, g_r_filtered = calc_first_coord_num_filtered( density   = solvent_density,
                                                                                                                  g_r       = current_oxy_g_r[each_solvent],
                                                                                                                  r         = current_oxy_r[each_solvent],
                                                                                                                  bin_width = self.rdf.bin_width,
                                                                                                                  savgol_filter_params = savgol_filter_params
                                                                                                                 )
                    ## STORING THE CDF OF ENTIRE THING
                    ## STORING INTO CSV
                    if self.rdf.solvent_name[each_solvent] == 'HOH':
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_HOH_' + each_oxygen, 
                                                     data = [cdf_filter] )
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'r_min_HOH_' + each_oxygen, 
                                                     data = [first_solvation_shell_filtered['min']['r']] )
                    else: # Cosolvent
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_COS_' + each_oxygen, 
                                                     data = [cdf_filter] )
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'r_min_COS_' + each_oxygen, 
                                                     data = [first_solvation_shell_filtered['min']['r']] )
                    
                    ## DEFINING EMPTY CDF BLOCK
                    cdfs_block = []
                        
                    ## LOOPING THROUGH THE BLOCKS AND CALCULATING FILTER
                    for each_block in range(self.rdf.split_rdf):
                        ## DEFINING CURRENT R AND G_R
                        current_block_r = current_oxy_split_r[each_block][each_solvent]
                        current_block_g_r = current_oxy_split_g_r[each_block][each_solvent]
                        
                        ## COMPUTING CDF
                        cdf_block_filter, first_solvation_shell_r_block_filter, g_r_block_filtered = calc_first_coord_num_filtered( density   = solvent_density,
                                                                                                                                          g_r       = current_block_g_r,
                                                                                                                                          r         = current_block_r,
                                                                                                                                          bin_width = self.rdf.bin_width,
                                                                                                                                          savgol_filter_params = savgol_filter_params
                                                                                                                                         )
                        ## STORING VALUES
                        cdfs_block.append(cdf_block_filter)
                    
                    ## FINDING AVERAGE AND STD
                    block_avg_cdf = np.mean( cdfs_block )
                    block_std_cdf = np.std( cdfs_block )
                    
                    ## STORING CDF
                    ## STORING INTO CSV
                    if self.rdf.solvent_name[each_solvent] == 'HOH':
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'block_avg_cumul_HOH_' + each_oxygen, 
                                                     data = [block_avg_cdf] )
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'block_std_cumul_HOH_' + each_oxygen, 
                                                     data = [block_std_cdf] )
                    else: # Cosolvent
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'block_avg_cumul_COS_' + each_oxygen, 
                                                     data = [block_avg_cdf] )
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'block_std_cumul_COS_' + each_oxygen, 
                                                     data = [block_std_cdf] )
                        
            return
                            
                        

    ### FUNCTION TO PLOT RDF OF SOLUTE AND SOLVENT WITH FIRST SOLVATION SHELL
    def plot_rdf_solute_solvent_first_solvation(self, save_fig=False, want_plot=True):
        '''
        The purpose of this script is to plot radial distribution function for each solvent and plot the first solvation shell
        INPUTS:
            self: class property
            save_fig: True if you want to save all the figures [Default: False]
            want_plot: [logical] True if you want to see a plot
        OUTPUTS:
            multiple figures 
        '''
        ## LOOPING THROUGH EACH SOLUTE
        for each_solute in range(self.rdf.num_solutes):
            ## CREATING FIGURE FOR RDF SOLUTE / SOLVENT
            if want_plot is True:
                fig, ax = self.plot_rdf_solute_solvent(save_fig = False)
            
            ## NOW, FINDING EQUILIBRIUM POINTS OF EACH PLOT AND DRAWING A LINE FOR THEM
            ## LOOPING THROUGH EACH SOLVENT
            for each_solvent in range(len(self.rdf.solvent_name)):
                ## FINDING SOLVATION SHELL INFORMATION
                first_solv_shell = find_first_solvation_shell_rdf( g_r =  self.rdf.rdf_g_r[each_solute][each_solvent],
                                                                   r= self.rdf.rdf_r[each_solute][each_solvent])
                if want_plot is True:
                    ## PLOTTING THE MAXIMA
                    ax.plot( first_solv_shell['max']['r'], first_solv_shell['max']['g_r'],
                            marker = 'o', label = 'First_solv_max', color = COLOR_LIST[each_solvent], **LINE_STYLE)
                    ## PLOTTING THE MINIMA
                    ax.plot( first_solv_shell['min']['r'], first_solv_shell['min']['g_r'],
                            marker = 'x', label = 'First_solv_min', color = COLOR_LIST[each_solvent], **LINE_STYLE)
                    ## PLOTTING LEGEND
                    ax.legend()
                
            ## SAVING IF NECESSARY
            label = "RDF_First_Solvation_%s_%s"%(self.rdf.solute_name[each_solute], '_'.join(self.rdf.solvent_name))
            if want_plot is True:
                save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)

        return
    
    ### FUNCTION TO PLOT THE RDF AND USE THE LARGEST CUTOFF
    def plot_rdf_solute_oxygen_to_solvent_largest_cutoff(self, save_fig=False, want_plot=True):
        '''
        The purpose of this function is to plot the oxygen to solvent with the largest cutoff possible.
        Algorithm:
            - calculate RDF cutoffs for each solvent system
            - find the largest of the cutoff
            - use that cutoff to get the cumulative distribution function
        INPUTS:
            self: [object]
                class property
            want_plot: [logical, default=True]
                True if you want the plot
            save_fig: [logical, default=False]
                True if you want to save all the figures
        OUTPUTS:
            - extracted values for RDF with largest cutoff
            - plot that shows the new cutoff
        '''
        ## CHECKING IF THE RDF DOES HAVE OXYGEN DETAILS
        if self.rdf.want_oxy_rdf is True:
            ## LOOPING THROUGH EACH SOLUTE
            for each_solute in range(self.rdf.num_solutes):
                ## LOOPING THROUGH EACH ATOM NAME
                for oxy_index, each_oxygen in enumerate(self.rdf.rdf_oxy_names[each_solute]):
                    ## CREATING RDF PLOT
                    if want_plot is True:
                        fig, ax = self.create_rdf_plot()
                    ## EXTRACTING DATA SINCE IT IS [COSOLVENT] THEN [OXYGEN]
                    current_oxy_r =[ self.rdf.rdf_oxy_r[each_solute][solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                    current_oxy_g_r =[ self.rdf.rdf_oxy_g_r[each_solute][solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                    ## CREATING STORAGE SPACE FOR SOLVATION SHELL
                    solvation_shell_storage = []
                    ## LOOPING THROUGH EACH SOLVENT
                    for each_solvent in range(len(self.rdf.solvent_name)):
                        print("FINDING SOLVATION SHELL ON COSOLVENT: %s"%(self.rdf.solvent_name[each_solvent]))
                        ## FINDING SOLVATION SHELL INFORMATION
                        first_solv_shell = find_first_solvation_shell_rdf( g_r =  current_oxy_g_r[each_solvent],
                                                                       r= current_oxy_r[each_solvent])
                        ## APPENDING FIRST SOLVATION SHELL
                        solvation_shell_storage.append(first_solv_shell)
                        
                    ## NOW, FINDING MAXIMUM OF THE SOLVATION SHELLS
                    solvation_shell_r_min = [ each_solvation['min']['r'] for each_solvation in solvation_shell_storage]
                    max_r_btwn_solvation_shells = np.max(solvation_shell_r_min)
                    
                    ## STORING MAXIMUM
                    self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_largeR_r_value' + each_oxygen, 
                                                 data = [max_r_btwn_solvation_shells] )
                    
                    ## STORAGE SPACE FOR CUMULATIVE DIST
                    cumul_dist_storage = []
                    
                    ## LOOPING THROUGH EACH SOLVENT NOW WITH FULL INFORMATION
                    for each_solvent in range(len(self.rdf.solvent_name)):
                        ## INTEGRATING AND GETTING THE CUMULATIVE DISTRIBUTION FUNCTION
                        cumulative_dist = calc_cumulative_dist_function( density = self.rdf.total_solvent[each_solvent] / self.rdf.volume,
                                                                         g_r = current_oxy_g_r[each_solvent],
                                                                         r = current_oxy_r[each_solvent],
                                                                         bin_width = self.rdf.bin_width,
                                                                         r_cutoff = max_r_btwn_solvation_shells,
                                                                        )
                        ## STORING CUMULATIVE DIST
                        cumul_dist_storage.append(cumulative_dist)
                        ## STORING INTO CSV
                        if self.rdf.solvent_name[each_solvent] == 'HOH':
                            self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_largeR_HOH_' + each_oxygen, 
                                                         data = [cumulative_dist] )
                        else: # Cosolvent
                            self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_largeR_COS_' + each_oxygen, 
                                                         data = [cumulative_dist] )
                        
                        ## PLOTTING
                        if want_plot is True:
                            ## PLOTTING RDF
                            ax.plot(current_oxy_r[each_solvent], current_oxy_g_r[each_solvent], '-', color = COLOR_LIST[each_solvent],
                                    label= "%s-%s --- %s"%(self.rdf.solute_name, each_oxygen, self.rdf.solvent_name[each_solvent]),
                                    **LINE_STYLE)
                    if want_plot is True:
                        ## DRAWING LINE FOR MAXIMUM R
                        ax.axvline( x= max_r_btwn_solvation_shells, linestyle="--", linewidth=2, label = 'Largest_solv_min_r', color = 'k')
                        ## CREATING LEGEND
                        ax.legend()
                    
                    ## FINDING MOLE FRACTIONS
                    mole_fractions = np.array(cumul_dist_storage) / np.sum(cumul_dist_storage)
                    
                    ## STORING MOLE FRACTION
                    for idx, each_solvent in enumerate(self.rdf.solvent_name):
                        if each_solvent == 'HOH':
                            self.csv_info = csv_info_add(self.csv_info, data_title = 'x_HOH_' + each_oxygen, 
                                                         data = [mole_fractions[idx]] )
                        else:
                            self.csv_info = csv_info_add(self.csv_info, data_title = 'x_COS_' + each_oxygen, 
                                                         data = [mole_fractions[idx]] )
                            
                    
                    ## SAVING IF NECESSARY
                    label = "RDFLargerR_%s_%d_%s_%s"%(self.rdf.solute_name[each_solute], self.decoded_name['mass_frac_water'], each_oxygen,'_'.join(self.rdf.solvent_name))
                    if want_plot is True:
                        save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
        return
    
    
    ### FUNCTION TO PLOT RDF FOR EACH OXYGEN TO ALL SOLVENTS
    def plot_rdf_solute_oxygen_to_specific_solvent_atoms(self, 
                                                         want_solvent_list = None
                                                         ):
        '''
        The purpose of this function is to plot the solute to oxygen by plotting each solute to 
        the corresponding atoms 
        INPUTS:
            self: [obj]
                self property
            desired_solvent_list: [list]
                list of the solvents that you want. If None, then we will look for all solvents
                
        '''
        ## CHECKING IF TRUE
        if self.rdf.want_oxy_rdf_all_solvent is True:
            ## LOOPING THROUGH EACH SOLUTE
            for each_solute in range(self.rdf.num_solutes):
                ## LOOPING THROUGH EACH SOLVENT
                for solvent_index in range(len(self.rdf.solvent_name)):  
                    ## DEFINING SOLVENT ATOMIC LIST
                    desired_solvent_list = list(self.rdf.rdf_oxy_all_solvents[each_solute][solvent_index].keys())
                    if want_solvent_list is not None:
                        within_list = np.all(np.isin(want_solvent_list, desired_solvent_list))
                        if within_list == True:
                            desired_solvent_list = want_solvent_list[:]
                            print("fixing desired list")
                            print(desired_solvent_list)
                    else:   
                        within_list = True
                    print(want_solvent_list)
                    print(within_list)
                    print(desired_solvent_list)
                    ## PLOTTING IF SOLVENT IS TRUE
                    if within_list == True:
                        ## LOOPING THROUGH EACH OXYGEN INDEX
                        for oxy_index, each_oxygen in enumerate(self.rdf.rdf_oxy_names[each_solute]):
                            ## GENERATING A PLOT
                            fig, ax = self.create_rdf_plot()
                            ## EXTRACTING RDF DETAILS
                            current_oxy_r =[ self.rdf.rdf_oxy_all_solvents[each_solute][solvent_index][each_solvent_combination][0][oxy_index] 
                                                        for each_solvent_combination in desired_solvent_list]
                            current_oxy_g_r =[ self.rdf.rdf_oxy_all_solvents[each_solute][solvent_index][each_solvent_combination][1][oxy_index] 
                                                        for each_solvent_combination in desired_solvent_list]
                            self.current_oxy_r = current_oxy_r
                            self.current_oxy_g_r = current_oxy_g_r
                            print(current_oxy_r)
                            ## LOOPING THROUGH EACH
                            for each_rdf, key in enumerate(desired_solvent_list):
                                ## PLOTTING RDF
                                ax.plot(current_oxy_r[each_rdf], 
                                        current_oxy_g_r[each_rdf], '-', 
                                        # color = COLOR_LIST[each_solvent],
                                        label= "%s-%s--%s"%(self.rdf.solute_name[each_solute], each_oxygen, key),
                                        **LINE_STYLE)
                            ## PLOTTING LEGEND
                            ax.legend()
        
        return
    
    ### FUNCTION TO PLOT RDF OF SOLUTE-OXY TO SOLVENT
    def plot_rdf_solute_oxygen_to_solvent(self, save_fig=False, want_first_solvation_shell=False, want_plot=True):
        '''
        The purpose of this function is to plot the solute oxygen to solvent radial distribution functions
        INPUTS:
            self: class property
            save_fig: True if you want to save all the figures [Default: False]
            want_first_solvation_shell: True if you want first solvation shell to be shown
        OUTPUTS:
            rdf vs r for each oxygen
            figs, axs: figures and axis as lists
        '''
        ## CHECKING IF THE RDF DOES HAVE OXYGEN DETAILS
        if self.rdf.want_oxy_rdf is True:
            ## LOOPING THROUGH EACH SOLUTE
            for each_solute in range(self.rdf.num_solutes):
                ## LOOPING THROUGH EACH ATOM NAME
                for oxy_index, each_oxygen in enumerate(self.rdf.rdf_oxy_names[each_solute]):
                    ## CREATING RDF PLOT
                    if want_plot is True:
                        fig, ax = self.create_rdf_plot()
                    ## EXTRACTING DATA SINCE IT IS [COSOLVENT] THEN [OXYGEN]
                    current_oxy_r =[ self.rdf.rdf_oxy_r[each_solute][solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                    current_oxy_g_r =[ self.rdf.rdf_oxy_g_r[each_solute][solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                    ## LOOPING THROUGH EACH SOLVENT
                    for each_solvent in range(len(self.rdf.solvent_name)):
                        print("WORKING ON SOLVENT: %s"%(self.rdf.solvent_name[each_solvent]))
                        if want_plot is True:
                            ## PLOTTING RDF
                            ax.plot(current_oxy_r[each_solvent], current_oxy_g_r[each_solvent], '-', color = COLOR_LIST[each_solvent],
                                    label= "%s-%s --- %s"%(self.rdf.solute_name[each_solute], each_oxygen, self.rdf.solvent_name[each_solvent]),
                                    **LINE_STYLE)
                            
                        ## FINDING SOLVATION SHELL INFORMATION
                        first_solv_shell = find_first_solvation_shell_rdf( g_r =  current_oxy_g_r[each_solvent],
                                                                       r= current_oxy_r[each_solvent])                                        
                            
                        ## STORING INTO CSV
                        if self.rdf.solvent_name[each_solvent] == 'HOH':
                            self.csv_info = csv_info_add(self.csv_info, data_title = 'RDF_HOH_' + each_oxygen, 
                                                         data = [ current_oxy_r[each_solvent], current_oxy_g_r[each_solvent] ], labels=['r(nm)', 'g_r'] )
                        else: # Cosolvent
                            self.csv_info = csv_info_add(self.csv_info, data_title = 'RDF_COS_' + each_oxygen, 
                                                         data = [ current_oxy_r[each_solvent], current_oxy_g_r[each_solvent] ], labels=['r(nm)', 'g_r'] )
                            
                        ## SEEING IF YOU WANT FIRST SOLVATION SHELL
                        if want_first_solvation_shell is True:
                            ## FINDING SOLVATION SHELL INFORMATION
                            first_solv_shell = find_first_solvation_shell_rdf( g_r =  current_oxy_g_r[each_solvent],
                                                                           r= current_oxy_r[each_solvent])
                            ## INTEGRATING AND GETTING THE CUMULATIVE DISTRIBUTION FUNCTION
                            cumulative_dist = calc_cumulative_dist_function( density = self.rdf.total_solvent[each_solvent] / self.rdf.volume,
                                                                             g_r = current_oxy_g_r[each_solvent],
                                                                             r = current_oxy_r[each_solvent],
                                                                             bin_width = self.rdf.bin_width,
                                                                             r_cutoff = first_solv_shell['min']['r'],
                                                                            )
                            ## STORING INTO CSV
                            if self.rdf.solvent_name[each_solvent] == 'HOH':
                                self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_HOH_' + each_oxygen, 
                                                             data = [cumulative_dist] )
                                self.csv_info = csv_info_add(self.csv_info, data_title = 'r_min_HOH_' + each_oxygen, 
                                                             data = [first_solv_shell['min']['r']] )
                                self.csv_info = csv_info_add(self.csv_info, data_title = 'g_r_max_HOH_' + each_oxygen, 
                                                             data = [first_solv_shell['max']['g_r']] )
                            else: # Cosolvent
                                self.csv_info = csv_info_add(self.csv_info, data_title = 'cumul_COS_' + each_oxygen, 
                                                             data = [cumulative_dist] )
                                self.csv_info = csv_info_add(self.csv_info, data_title = 'r_min_COS_' + each_oxygen, 
                                                             data = [first_solv_shell['min']['r']] )
                                self.csv_info = csv_info_add(self.csv_info, data_title = 'g_r_max_COS_' + each_oxygen, 
                                                             data = [first_solv_shell['max']['g_r']] )
                            
                            if want_plot is True:
                                ## PLOTTING THE MAXIMA
                                ax.plot( first_solv_shell['max']['r'], first_solv_shell['max']['g_r'],
                                        marker = 'o', label = 'First_solv_max', color = COLOR_LIST[each_solvent], **LINE_STYLE)
                                ## PLOTTING THE MINIMA
                                ax.plot( first_solv_shell['min']['r'], first_solv_shell['min']['g_r'],
                                        marker = 'x', label = 'First_solv_min', color = COLOR_LIST[each_solvent], **LINE_STYLE)                     
                        ## CREATING LEGEND
                        if want_plot is True:
                            ax.legend()
                    ## SAVING IF NECESSARY
                    label = "RDF_%s_%d_%s_%s"%(self.rdf.solute_name[each_solute], self.decoded_name['mass_frac_water'], each_oxygen,'_'.join(self.rdf.solvent_name))
                    if want_plot is True:
                        save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
            else:
                print("There is no oxygen rdf information, want_oxy_rdf is set to: %s"%(self.rdf.want_oxy_rdf))
            
        return 


#%% MAIN SCRIPT
if __name__ == "__main__":

    from MDDescriptors.application.solvent_effects.rdf import calc_rdf
    ## DEFINING CLASS
    Descriptor_class = calc_rdf
    
    ## DEFINING DATE
    Date='181010-PDO_DIO_DMSO'
    Date='181010-PDO_MULTIPLE_SOLVENTS'
    Date='181018-CLL'
    Date='181114-bin_0.02'
    Date='190108'
    Date='190726'
    # Date='181114-bin_0.02_test'
    # Date='181115' # 0.01
    # Date='181109'
    # Date='181012-Archive_10ns'
    # Date='180622'
    
    ## DEFINING DESIRED DIRECTORY
    # Pickle_loading_file=r"mdRun_433.15_6_nm_ACE_50_WtPercWater_spce_dmso"
    # Pickle_loading_file=r"mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_tetramethylenesulfoxide"
    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane"
    Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_5_WtPercWater_spce_dioxane"
    Pickle_loading_file=r"Mostlikely_433.15_6_nm_PDO_10_WtPercWater_spce_dmso"
    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane"
    # Pickle_loading_file=r"mdRun_300.00_6_nm_HYD_10_WtPercWater_spce_dmso"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_100_WtPercWater_spce_Pure"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_GVL_L"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_GVL_L"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dioxane"
#    
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dmso"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dmso"
#    
    # Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_tetrahydrofuran"
    # Pickle_loading_file=r"mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_tetrahydrofuran"
#    # Pickle_loading_file=r"mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dmso"
    
    '''
    mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_NN-dimethylacetamide
    mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_NN-dimethylformamide
    mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_urea
    '''
    # Pickle_loading_file=r"mdRun_433.15_6_nm_PRO_100_WtPercWater_spce_Pure"
    # Pickle_loading_file="mdRun_433.15_6_nm_PDO_50_WtPercWater_spce_dmso"
    # Pickle_loading_file="mdRun_433.15_6_nm_PDO_75_WtPercWater_spce_dmso"
    # Pickle_loading_file="mdRun_433.15_6_nm_PDO_25_WtPercWater_spce_dmso"
    # Pickle_loading_file="mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dmso"
    
    #### SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ### EXTRACTING THE RDF
    rdf = multi_traj_results
    
    
    
    
    
    
    
    #%%
    import matplotlib.pyplot as plt
    plt.close('all')
    # plt.close('all')
    rdf_plot = extract_rdf(rdf,Pickle_loading_file)
    
    ## DEFINING DESIRED SOLVENT LIST
    want_solvent_list = [
            'dmso_S1',
            'dmso_O1',
            ]
    
    rdf_plot.plot_rdf_solute_oxygen_to_specific_solvent_atoms( want_solvent_list = want_solvent_list)
    
    #%%
    test = ['dmso_O1']
    test2 = ['dmso_O1', 'dmso_O2', 'dmso_O3']
    logical = np.all(np.isin(test, test2))
    
    
    #%%
    rdf_plot.plot_rdf_solute_solvent( want_plot = True, want_radius_cutoff= True )
    # print("%.5f"%(rdf_plot.equil_radius))
    # rdf_plot.plot_rdf_solute_solvent_first_solvation(True)
    
#    rdf_plot.plot_rdf_solute_oxygen_to_solvent(want_first_solvation_shell = True,
#                                               want_plot = True,
#                                               save_fig = False)
#    
    #%%
    

    ## CLOSING ALL PLOTS
    # plt.close('all')
    

    ## SOLVENT INDEX
    solvent_index = 1
    block_index = 0
    oxy_index = 1
    
    
    ## DEFINING DENSITY
    density = rdf.total_solvent[solvent_index] / rdf.volume
    
    ## DEFINING SAVGOL FILTERING PARAMETERS
    savgol_filter_params={'window_length':3, 'polyorder': 1, 'mode': 'nearest'}
    
    ## DEFINING RDF DATA
#    rdf_data_g_r = rdf.rdf_oxy_g_r[0][solvent_index][0]
#    rdf_data_r = rdf.rdf_oxy_r[0][solvent_index][0]
    # [SOLUTE][SOLVENT][BLOCK_INDEX][ATOM_TYPE]
    rdf_data_g_r = rdf.rdf_split_data['OXY_RDF']['g_r'][0][solvent_index][block_index][oxy_index]
    rdf_data_r = rdf.rdf_split_data['OXY_RDF']['r'][0][solvent_index][block_index][oxy_index]
#    
    ## USING FILTERING
    # rdf_filtered = signal.savgol_filter(x=rdf_data_g_r, **savgol_filter_params)
    
    ## FINDING EXTREMA
    # local and maximums: https://datascience.stackexchange.com/questions/27031/how-to-get-spike-values-from-a-value-sequence?rq=1
    # test_x_max = signal.argrelextrema( rdf_filtered , np.greater  )
    # test_x_min = signal.argrelextrema( rdf_filtered , np.less  )
    
    ## CALCULATING CDF OF EACH
    cdf, first_solv_shell = calc_first_coord_num( density = density,
                                                             g_r = rdf_data_g_r, 
                                                             r = rdf_data_r, 
                                                             bin_width = rdf.bin_width,
                                                            )
#    ## FILTERED
#    cdf_filter, min_first_solvation_shell_r_filter = calc_first_coord_num( density = density,
#                                                             g_r = rdf_filtered, 
#                                                             r = rdf_data_r, 
#                                                             bin_width = rdf.bin_width,
#                                                            )
    
    ## FILTERED
    cdf_filter, first_solv_shell_filtered, rdf_filtered= calc_first_coord_num_filtered( density = density,
                                                             g_r = rdf_data_g_r, 
                                                             r = rdf_data_r, 
                                                             bin_width = rdf.bin_width,
                                                             savgol_filter_params = savgol_filter_params,
                                                            )
    
    
    ## MAKING PLOT
    fig, ax = create_rdf_plot()
    
    ## ADDING TO PLOT
    ax.plot( rdf_data_r, rdf_data_g_r, color='black', linestyle='-', linewidth=2, label="Actual Data, CDF=%.3f"%(cdf) )
    ax.plot( rdf_data_r, rdf_filtered, color='blue', linestyle='-', linewidth=2, label = "Savgol Filter, CDF=%.3f"%(cdf_filter)  )
    
    ## PLOTTING CUTOFF
    ax.axvline( x= first_solv_shell['min']['r'], color="black", linestyle='-.', linewidth=2, label = "Actual cutoff (%.2f nm)"%( first_solv_shell['min']['r']) )
    ax.axvline( x= first_solv_shell_filtered['min']['r'], color="blue", linestyle='-.', linewidth=2, label = "Savgol cutoff (%.2f nm)"%(first_solv_shell_filtered['min']['r'])  )
    
    ## PLOTTING MAXIMA
    ax.axvline( x= first_solv_shell['max']['r'], color="black", linestyle='--', linewidth=2, label = "Actual Maxima (%.2f nm)"%( first_solv_shell['max']['r']) )
    ax.axvline( x= first_solv_shell_filtered['max']['r'], color="blue", linestyle='--', linewidth=2, label = "Savgol Maxima (%.2f nm)"%(first_solv_shell_filtered['max']['r'])  )
    
    ## ADDING LEGEND
    ax.legend()
    
    
    
    
    
        
    
    
    
    #%%
    

    
    #%%

    
    ##### MULTI TRAJ ANALYSIS
    # traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date, 
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, # ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'],
                    export_class = extract_rdf,
                    export_text_file_name = 'extract_rdf',
                    )    



    
    
    #%%
    ##### MULTI TRAJ ANALYSIS
    list_of_pickles = find_multi_traj_pickle(Date, Descriptor_class)

    rdfs=[]
    
    for Pickle_loading_file in list_of_pickles:
        ### EXTRACTING THE DATA
        multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
        ### EXTRACTING THE RDF
        rdf = multi_traj_results
        ## PLOTTING RDF
        # rdf_plot = extract_rdf(rdf,Pickle_loading_file)
        # rdf_plot.plot_rdf_solute_oxygen_to_solvent(True,
          #                                          want_first_solvation_shell = True)
    
        ## STORING IN RDFS
        rdfs.append(rdf)
    
    
    #%%
    
    
    #########################################################################
    ### CLASS FUNCTION TO PLOTTING MULTIPLE RADIAL DISTRIBUTION FUNCTIONS ###
    #########################################################################
    class multi_plot_rdf:
        '''
        The purpose of this class is to take multiple class_rdf classes and plot it accordingly
        INPUTS:
            rdfs: list of calc_rdf classes
            names: list of names associated with each rdf
            decode_type: string denoting way to decode the names
            rdf_range: [tuple, 3, default=None]
                range of your RDF in y-axis, e.g. (0, 50, 5) <-- indicates minimum is zero, maximum is 50, and increment of 5
        OUTPUTS:
            ## INPUTS
                self.rdfs: rdfs
                self.names: Names of the directories
                self.decode_type: decoding type for the directories
                
            ## DECODING
                self.names_decoded: decoded names
                self.unique_solute_names: unique solute names
            
            
        FUNCTIONS:
            find_unique: finds unique decoders
            convert_water_to_cosolvent_mass_frac: converts water mass fraction to cosolvent
            
        ACTIVE FUNCTIONS:
            plot_rdf_solute_solvent_multiple_mass_frac: Plots rdf solute to solvent for multiple mass fractions
            plot_rdf_solute_oxy_to_solvent_multiple_mass_frac: plots rdf of solute's oxygen to solvent for multiple mass fractions
        '''
        ### INITIALIZATION
        def __init__(self, rdfs, names, rdf_range=None, decode_type='solvent_effects'):
            ## DEFINING ORGANIZATION LEVELS
            self.organization_levels = [ 'solute_residue_name', 'cosolvent_name', 'mass_frac_water' ]
            
            ## STORING INPUTS
            self.rdfs = rdfs
            self.names = names
            self.rdf_range=rdf_range
            self.decode_type = decode_type
            
            ## DECODING NAMES
            self.names_decoded = [decode_name(name=name,decode_type=decode_type) for name in self.names]
            
            ## FINDING UNIQUE SOLUTE NAMES
            self.unique_solute_names = self.find_unique('solute_residue_name')
            
            ## PLOTTING RDF FOR DIFFERENT MASS FRACTIONS
            # self.plot_rdf_solute_solvent_multiple_mass_frac()

        ### FUNCTION TO FIND ALL UNIQUE RESIDUES
        def find_unique(self,decoding_name):
            '''
            The purpose of this function is to find all unique solutes
            INPUTS:
                self: class property
                decoding_name: decoding name
                    e.g. 'solute_residue_name', etc.
            OUTPUTS:
                unique_names: unique names
            '''
            unique_names = list(set([each_decoded_name[decoding_name] for each_decoded_name in self.names_decoded]))
            return unique_names
            
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
        
        ### FUNCTION TO CONVERT MASS FRACTION FROM WATER TO COSOLVENT
        @staticmethod
        def convert_water_to_cosolvent_mass_frac(mass_frac_water_perc):
            '''
            The purpose of this script is to convert mass fraction from water to cosolvent
            INPUTS:
                mass_frac_water_perc: mass fraction of water (as a percent, e.g. 10)
            OUTPUTS:
                mass_frac_cosolvent: mass fraction of cosolvent (e.g. 0.90)
            '''
            return (100 - mass_frac_water_perc)/float(100)
        
        ### FUNCTION TO PLOT FOR DIFFERENT MASS FRACTIONS
        def plot_rdf_solute_solvent_multiple_mass_frac(self, save_fig=False):
            '''
            The purpose of this function is to plot the solute to solvent for multiple mass fractions
            INPUTS:
                self: class object
                save_fig: True if you want to save all the figures
            OUTPUTS:
                plot of RDF vs distance for different mass fractions of solvents
            '''
            ## LOOPING THROUGH EACH SOLUTE
            for solute_idx, each_solute in enumerate(self.unique_solute_names):
                ## LOOPING THROUGH EACH COSOLVENT
                for each_solvent in self.find_unique('cosolvent_name'):
                    ## EXCLUDING IF PURE CASE
                    if each_solvent != 'Pure':
                        ## FINDING ALL INDICES THAT HAVE THIS SOLUTE AND SOLVENT
                        mass_frac_indices = [index for index, name_decoded in enumerate(self.names_decoded) \
                                             if name_decoded['solute_residue_name']==each_solute and name_decoded['cosolvent_name'] ==each_solvent]
                        ## FINDING ALL MASS FRACTIONS
                        water_mass_frac_values = [ self.names_decoded[index]['mass_frac_water'] for index in mass_frac_indices]
                        ## SORT BY THE SMALLEST MASS FRACTION OF WATER
                        water_mass_frac_values, mass_frac_indices = (list(t) for t in zip(*sorted(zip(water_mass_frac_values, mass_frac_indices))))
                        ## GETTING MASS FRACTION OF COSOLVENT
                        cosolvent_mass_frac_values = [ self.convert_water_to_cosolvent_mass_frac(each_mass_perc) for each_mass_perc in water_mass_frac_values ]                                                
                        
                        ## RDF -- SOLUTE - SOLVENT
                        for solvent_index,each_solvent_name in enumerate(rdfs[mass_frac_indices[0]].solvent_name):
                            ## CREATING RDF PLOT
                            fig, ax = self.create_rdf_plot()
                            ## SETTING THE TITLE
                            ax.set_title("%s --- %s"%(each_solute, each_solvent_name))
                            ## LOOPING THROUGH EACH MASS FRACTION AND PLOTTING
                            for each_mass_frac in range(len(mass_frac_indices)):
                                ## GETTING DATA INDEX
                                data_index = mass_frac_indices[each_mass_frac]
                                ## GETTING G_R AND R
                                g_r = self.rdfs[data_index].rdf_g_r[0][solvent_index]
                                r   = self.rdfs[data_index].rdf_r[0][solvent_index]
                                ## PLOTTING G_R VS R
                                ax.plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac],
                                                label= "m_org: %.2f"%(cosolvent_mass_frac_values[each_mass_frac]),
                                                **LINE_STYLE)
                                
                                
                            ## ADDING PLOT IF 100% WATER EXISTS
                            pure_water_index = [index for index, name_decoded in enumerate(self.names_decoded) \
                                                 if name_decoded['solute_residue_name']==each_solute and \
                                                 name_decoded['cosolvent_name'] == 'Pure' and \
                                                 name_decoded['mass_frac_water'] == 100
                                                 ]
                            if len(pure_water_index) !=0 and each_solvent_name == 'HOH':
                                ## GETTING G_R AND R
                                g_r = self.rdfs[pure_water_index[0]].rdf_g_r[0][0]
                                r   = self.rdfs[pure_water_index[0]].rdf_r[0][0]
                                ## PLOTTING G_R VS R
                                ax.plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac+1],
                                                label= "m_org: %.2f"%(0),
                                                **LINE_STYLE)
                            ## CREATING LEGEND
                            ax.legend()
                            ## PLACING LEGEND OUTSIDE
                            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            ## UPDATING RANGE IF NECESSARY
                            if self.rdf_range != None:
                                ## DEFINING Y-TICK RANGE
                                y_range = np.arange( self.rdf_range[0], self.rdf_range[1] + self.rdf_range[2], self.rdf_range[2] )
                                ## UPDATING PLOT
                                ax.set_yticks( y_range ) 

                            ## LABELING FIGURE
                            label = "RDF_mass_frac_%s_%s_%s"%(each_solute, each_solvent,each_solvent_name)
                            ## SAVING FIGURE
                            save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
            return
            
        ### FUNCTION TO PLOT OXYGENS
        def plot_rdf_solute_oxy_to_solvent_multiple_mass_frac(self, save_fig=False, clear_plot = True):
            '''
            The purpose of this function is to plot the solute to solvent for multiple mass fractions
            INPUTS:
                self: class object
                save_fig: True if you want to save all the figures
                clear_plot: [logical, default=True]
                    True if you want to clear all plots to prevent overflow of plots
            OUTPUTS:
                plot of RDF vs distance for different mass fractions of solvents
            '''
            ## LOOPING THROUGH EACH SOLUTE
            for each_solute in self.unique_solute_names:
                ## LOOPING THROUGH EACH COSOLVENT
                for each_solvent in self.find_unique('cosolvent_name'):
                    ## EXCLUDING IF PURE CASE
                    if each_solvent != 'Pure':
                        ## CLEARING PLOTS
                        if clear_plot is True:
                            plt.close('all')    
                        
                        ## FINDING ALL INDICES THAT HAVE THIS SOLUTE AND SOLVENT
                        mass_frac_indices = [index for index, name_decoded in enumerate(self.names_decoded) \
                                             if name_decoded['solute_residue_name']==each_solute and name_decoded['cosolvent_name'] == each_solvent]
                        ## FINDING ALL MASS FRACTIONS
                        water_mass_frac_values = [ self.names_decoded[index]['mass_frac_water'] for index in mass_frac_indices]
                        ## SORT BY THE SMALLEST MASS FRACTION OF WATER
                        water_mass_frac_values, mass_frac_indices = (list(t) for t in zip(*sorted(zip(water_mass_frac_values, mass_frac_indices))))
                        ## GETTING MASS FRACTION OF COSOLVENT
                        cosolvent_mass_frac_values = [ self.convert_water_to_cosolvent_mass_frac(each_mass_perc) for each_mass_perc in water_mass_frac_values ]                                                
                        ## CREATING FIGURE AND AXIS
                        figs_axs = [ [[self.create_rdf_plot()][0] for index in range(len(self.rdfs[mass_frac_indices[0]].solvent_name))]  # Vary by solvent name
                                        for atomname in range(len(self.rdfs[mass_frac_indices[0]].rdf_oxy_names[0]  )) ] # Vary by atom solute name
                        
                        ### LOOPING OVER EACH ATOM NAME
                        for atom_index, atomname in enumerate(self.rdfs[mass_frac_indices[0]].rdf_oxy_names[0]):
                            
                            ## LOOPING OVER EACH SOLVENT
                            for solvent_index,each_solvent_name in enumerate(self.rdfs[mass_frac_indices[0]].solvent_name):

                                ## SETTING THE TITLE
                                figs_axs[atom_index][solvent_index][1].set_title("%s-%s --- %s"%(each_solute,atomname, each_solvent_name))
                                ## LOOPING THROUGH EACH MASS FRACTION AND PLOTTING
                                for each_mass_frac in range(len(mass_frac_indices)):
                                    ## GETTING DATA INDEX
                                    data_index = mass_frac_indices[each_mass_frac]
                                    ## GETTING G_R AND R
                                    g_r = self.rdfs[data_index].rdf_oxy_g_r[0][solvent_index][atom_index]
                                    r   = self.rdfs[data_index].rdf_oxy_r[0][solvent_index][atom_index]
                                
                                    ## PLOTTING G_R VS R
                                    figs_axs[atom_index][solvent_index][1].plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac],
                                                    label= "m_org: %.2f"%(cosolvent_mass_frac_values[each_mass_frac]),
                                                    **LINE_STYLE)
                                ## ADDING PLOT IF 100% WATER EXISTS
                                pure_water_index = [index for index, name_decoded in enumerate(self.names_decoded) \
                                                     if name_decoded['solute_residue_name']==each_solute and \
                                                     name_decoded['cosolvent_name'] == 'Pure' and \
                                                     name_decoded['mass_frac_water'] == 100
                                                     ]
                                if len(pure_water_index) !=0 and each_solvent_name == 'HOH':
                                    ## GETTING G_R AND R
                                    g_r = self.rdfs[pure_water_index[0]].rdf_oxy_g_r[0][0][atom_index]
                                    r   = self.rdfs[pure_water_index[0]].rdf_oxy_r[0][0][atom_index]
                                    ## PLOTTING G_R VS R
                                    figs_axs[atom_index][solvent_index][1].plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac+1],
                                                    label= "m_org: %.2f"%(0),
                                                    **LINE_STYLE)
                                ## CREATING LEGEND
                                figs_axs[atom_index][solvent_index][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                # # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                ## LABELING FIGURE
                                figs_axs[atom_index][solvent_index][1].label_ = "RDF_mass_frac_%s_%s_%s_%s"%(each_solute, each_solvent,each_solvent_name,atomname)
                                ## SAVING FIGURE
                                # save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
                                self.figs_axs = figs_axs[:]
                        ## SAVING FIGURE IF NECESSARY
                        [ [save_fig_png(fig = figs_axs[atom_index][solvent_index][0],
                                         label=figs_axs[atom_index][solvent_index][1].label_, 
                                         save_fig=save_fig)] 
                                    for solvent_index in range(len(self.rdfs[mass_frac_indices[0]].solvent_name)) # Vary by solvent name
                                    for atom_index in range(len(self.rdfs[mass_frac_indices[0]].rdf_oxy_names[0])) ] # Vary by atom solute name
            return
        
    #%%
    ## CLOSING ALL FIGURES
    plt.close('all')    
            
    multi_rdf = multi_plot_rdf(rdfs = rdfs,
                               names = list_of_pickles,
                               rdf_range= (0, 2.4, 0.2 ),
                               decode_type = 'solvent_effects',
                               )
    
    # multi_rdf.plot_rdf_solute_solvent_multiple_mass_frac(save_fig = True)
    


    ## CLOSING ALL FIGURES
    plt.close('all')    

    ## PLOTTING
    multi_rdf.plot_rdf_solute_oxy_to_solvent_multiple_mass_frac(True)
    
    
    