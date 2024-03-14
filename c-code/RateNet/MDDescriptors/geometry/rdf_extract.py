# -*- coding: utf-8 -*-
"""
rdf_extract.py
The purpose of this script is to analyze the data from the multi_traj_analysis_tool.py for RDFs. This script also contains code to plot the rdfs



Author(s):
    Alex K. Chew (alexkchew@gmail.com)
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
## SYSTEM TOOLS
import sys
## MATH TOOLS
import numpy as np
## CSV TOOLS
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add

### FUNCTION TO FIND THE COORDINATION NUMBER GIVEN A CUTOFF
def calc_cumulative_dist_function( density, g_r, r, bin_width, r_cutoff = None):
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
    OUTPUTS:
        cdf: [float] Coordination number, or number of atoms within a cutoff r
    '''
    ## FINDING INDEX OF THE CUTOFF
    if r_cutoff is None: # No cutoff, assuming entire radius vector
        index_within_cutoff = np.where( r <= np.max(r) )
    else:
        index_within_cutoff = np.where( r <= r_cutoff )
    ## PRINTING
    print("CALCULATING CDF FOR CUTOFF RADIUS: %.2f"%(r[index_within_cutoff][-1])   )
    ## USING CUTOFF
    g_r_within_cutoff = g_r[index_within_cutoff]
    r_within_cutoff = r[index_within_cutoff]
    ## DEFINING CDF
    CDF_Integral_Function=density * g_r_within_cutoff * 4 * np.pi * r_within_cutoff * r_within_cutoff
    ## INTEGRATING TO GET THE CDF
    cdf= np.trapz(CDF_Integral_Function,r_within_cutoff,dx=bin_width)
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

################################################################
### CLASS FUNCTION TO PLOTTING RADIAL DISTRIBUTION FUNCTIONS ###
################################################################
class plot_rdf:
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
        self.decoded_name = decode_name(pickle_name, decoder_type )
        
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        
        ## RDF FOR OXYGENS
        self.plot_rdf_solute_oxygen_to_solvent(want_first_solvation_shell = True, want_plot=False)
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
    def plot_rdf_solute_solvent(self, save_fig=False, want_plot=True):
        '''
        The purpose of this script is to plot radial distribution function for each solvent
        INPUTS:
            self: class property
            save_fig: True if you want to save all the figures [Default: False]
            want_plot: [logical] True if you want to see a plot
        OUTPUTS:
            multiple figures 
        '''
        ## CREATING RDF PLOT
        fig, ax = self.create_rdf_plot()
        
        ## LOOPING THROUGH EACH SOLVENT
        for each_solvent in range(len(self.rdf.solvent_name)):
            
            if want_plot is True:
                ## PLOTTING RDF
                ax.plot(self.rdf.rdf_r[each_solvent], self.rdf.rdf_g_r[each_solvent], '-', color = COLOR_LIST[each_solvent],
                        label= "%s --- %s"%(self.rdf.solute_name, self.rdf.solvent_name[each_solvent]),
                        **LINE_STYLE)
                ax.legend()
            
        ## SAVING IF NECESSARY
        label = "RDF_%s_%s"%(self.rdf.solute_name, '_'.join(self.rdf.solvent_name))
        save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
        
        return fig, ax
    
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
        ## CREATING FIGURE FOR RDF SOLUTE / SOLVENT
        if want_plot is True:
            fig, ax = self.plot_rdf_solute_solvent(save_fig = False)
        
        ## NOW, FINDING EQUILIBRIUM POINTS OF EACH PLOT AND DRAWING A LINE FOR THEM
        ## LOOPING THROUGH EACH SOLVENT
        for each_solvent in range(len(self.rdf.solvent_name)):
            ## FINDING SOLVATION SHELL INFORMATION
            first_solv_shell = find_first_solvation_shell_rdf( g_r =  self.rdf.rdf_g_r[each_solvent],
                                                               r= self.rdf.rdf_r[each_solvent])
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
        label = "RDF_First_Solvation_%s_%s"%(self.rdf.solute_name, '_'.join(self.rdf.solvent_name))
        if want_plot is True:
            save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)

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
            ## LOOPING THROUGH EACH ATOM NAME
            for oxy_index, each_oxygen in enumerate(self.rdf.rdf_oxy_names):
                ## CREATING RDF PLOT
                if want_plot is True:
                    fig, ax = self.create_rdf_plot()
                ## EXTRACTING DATA SINCE IT IS [COSOLVENT] THEN [OXYGEN]
                current_oxy_r =[ self.rdf.rdf_oxy_r[solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                current_oxy_g_r =[ self.rdf.rdf_oxy_g_r[solvent_index][oxy_index] for solvent_index in range(len(self.rdf.solvent_name))]
                ## LOOPING THROUGH EACH SOLVENT
                for each_solvent in range(len(self.rdf.solvent_name)):
                    print("WORKING ON SOLVENT: %s"%(self.rdf.solvent_name[each_solvent]))
                    if want_plot is True:
                        ## PLOTTING RDF
                        ax.plot(current_oxy_r[each_solvent], current_oxy_g_r[each_solvent], '-', color = COLOR_LIST[each_solvent],
                                label= "%s-%s --- %s"%(self.rdf.solute_name, each_oxygen, self.rdf.solvent_name[each_solvent]),
                                **LINE_STYLE)
                        
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
                label = "RDF_%s_%d_%s_%s"%(self.rdf.solute_name, self.decoded_name['mass_frac_water'], each_oxygen,'_'.join(self.rdf.solvent_name))
                if want_plot is True:
                    save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
        else:
            print("There is no oxygen rdf information, want_oxy_rdf is set to: %s"%(self.rdf.want_oxy_rdf))
            
        return 


#%% MAIN SCRIPT
if __name__ == "__main__":

    from MDDescriptors.geometry.rdf import calc_rdf
    ## DEFINING CLASS
    Descriptor_class = calc_rdf
    
    ## DEFINING DATE
    Date='181001'
    # Date='180622'
    #%%
    ## DEFINING DESIRED DIRECTORY
    # Pickle_loading_file=r"mdRun_433.15_6_nm_ACE_50_WtPercWater_spce_dmso"
    Pickle_loading_file=r"mdRun_300.00_6_nm_HYD_100_WtPercWater_spce_Pure"
    
    '''
    mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_NN-dimethylacetamide
    mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_NN-dimethylformamide
    mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_urea
    '''
    
    #### SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ### EXTRACTING THE RDF
    rdf = multi_traj_results
    
    rdf_plot = plot_rdf(rdf,Pickle_loading_file)
    
    # rdf_plot.plot_rdf_solute_oxygen_to_solvent(True)
    
    # rdf_plot.plot_rdf_solute_solvent_first_solvation(True)
    
    rdf_plot.plot_rdf_solute_oxygen_to_solvent(want_first_solvation_shell = True,
                                               want_plot = True,
                                               save_fig = True)
    
    
    #%%

    
    ##### MULTI TRAJ ANALYSIS
    # traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date, 
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, # ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'],
                    export_class = plot_rdf,
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
        rdf_plot = plot_rdf(rdf,Pickle_loading_file)
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
        def __init__(self, rdfs, names, decode_type='solvent_effects'):
            ## DEFINING ORGANIZATION LEVELS
            self.organization_levels = [ 'solute_residue_name', 'cosolvent_name', 'mass_frac_water' ]
            
            ## STORING INPUTS
            self.rdfs = rdfs
            self.names = names
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
            for each_solute in self.unique_solute_names:
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
                                g_r = self.rdfs[data_index].rdf_g_r[solvent_index]
                                r   = self.rdfs[data_index].rdf_r[solvent_index]
                                ## PLOTTING G_R VS R
                                ax.plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac],
                                                label= "Cosolvent mass frac: %.2f"%(cosolvent_mass_frac_values[each_mass_frac]),
                                                **LINE_STYLE)
                            ## ADDING PLOT IF 100% WATER EXISTS
                            pure_water_index = [index for index, name_decoded in enumerate(self.names_decoded) \
                                                 if name_decoded['solute_residue_name']==each_solute and \
                                                 name_decoded['cosolvent_name'] == 'Pure' and \
                                                 name_decoded['mass_frac_water'] == 100
                                                 ]
                            if len(pure_water_index) !=0 and each_solvent_name == 'HOH':
                                ## GETTING G_R AND R
                                g_r = self.rdfs[pure_water_index[0]].rdf_g_r[0]
                                r   = self.rdfs[pure_water_index[0]].rdf_r[0]
                                ## PLOTTING G_R VS R
                                ax.plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac+1],
                                                label= "Cosolvent mass frac: %.2f"%(0),
                                                **LINE_STYLE)
                            ## CREATING LEGEND
                            ax.legend()
                            ## LABELING FIGURE
                            label = "RDF_mass_frac_%s_%s_%s"%(each_solute, each_solvent,each_solvent_name)
                            ## SAVING FIGURE
                            save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
            return
            
        ### FUNCTION TO PLOT OXYGENS
        def plot_rdf_solute_oxy_to_solvent_multiple_mass_frac(self, save_fig=False):
            '''
            The purpose of this function is to plot the solute to solvent for multiple mass fractions
            INPUTS:
                self: class object
                save_fig: True if you want to save all the figures
            OUTPUTS:
                plot of RDF vs distance for different mass fractions of solvents
            '''
            ## LOOPING THROUGH EACH SOLUTE
            for each_solute in self.unique_solute_names:
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
                        ## CREATING FIGURE AND AXIS
                        figs_axs = [ [[self.create_rdf_plot()][0] for index in range(len(rdfs[mass_frac_indices[0]].solvent_name))]  # Vary by solvent name
                                        for atomname in range(len(rdfs[mass_frac_indices[0]].rdf_oxy_names)) ] # Vary by atom solute name
                        ### LOOPING OVER EACH ATOM NAME
                        for atom_index, atomname in enumerate(rdfs[mass_frac_indices[0]].rdf_oxy_names):
                            
                            ## LOOPING OVER EACH SOLVENT
                            for solvent_index,each_solvent_name in enumerate(rdfs[mass_frac_indices[0]].solvent_name):
                                ## SETTING THE TITLE
                                figs_axs[atom_index][solvent_index][1].set_title("%s-%s --- %s"%(each_solute,atomname, each_solvent_name))
                                ## LOOPING THROUGH EACH MASS FRACTION AND PLOTTING
                                for each_mass_frac in range(len(mass_frac_indices)):
                                    ## GETTING DATA INDEX
                                    data_index = mass_frac_indices[each_mass_frac]
                                    ## GETTING G_R AND R
                                    g_r = self.rdfs[data_index].rdf_oxy_g_r[solvent_index][atom_index]
                                    r   = self.rdfs[data_index].rdf_oxy_r[solvent_index][atom_index]
                                
                                    ## PLOTTING G_R VS R
                                    figs_axs[atom_index][solvent_index][1].plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac],
                                                    label= "Cosolvent mass frac: %.2f"%(cosolvent_mass_frac_values[each_mass_frac]),
                                                    **LINE_STYLE)
                                ## ADDING PLOT IF 100% WATER EXISTS
                                pure_water_index = [index for index, name_decoded in enumerate(self.names_decoded) \
                                                     if name_decoded['solute_residue_name']==each_solute and \
                                                     name_decoded['cosolvent_name'] == 'Pure' and \
                                                     name_decoded['mass_frac_water'] == 100
                                                     ]
                                if len(pure_water_index) !=0 and each_solvent_name == 'HOH':
                                    ## GETTING G_R AND R
                                    g_r = self.rdfs[pure_water_index[0]].rdf_oxy_g_r[0][atom_index]
                                    r   = self.rdfs[pure_water_index[0]].rdf_oxy_r[0][atom_index]
                                    ## PLOTTING G_R VS R
                                    figs_axs[atom_index][solvent_index][1].plot(r, g_r, '-', color = COLOR_LIST[each_mass_frac+1],
                                                    label= "Cosolvent mass frac: %.2f"%(0),
                                                    **LINE_STYLE)
                                ## CREATING LEGEND
                                figs_axs[atom_index][solvent_index][1].legend()
                                ## LABELING FIGURE
                                figs_axs[atom_index][solvent_index][1].label_ = "RDF_mass_frac_%s_%s_%s_%s"%(each_solute, each_solvent,each_solvent_name,atomname)
                                ## SAVING FIGURE
                                # save_fig_png(fig, label, save_fig, dpi=DPI_LEVEL)
                                self.figs_axs = figs_axs[:]
                        ## SAVING FIGURE IF NECESSARY
                        [ [save_fig_png(fig = figs_axs[atom_index][solvent_index][0],
                                         label=figs_axs[atom_index][solvent_index][1].label_, 
                                         save_fig=save_fig)] 
                                    for solvent_index in range(len(rdfs[mass_frac_indices[0]].solvent_name)) # Vary by solvent name
                                    for atom_index in range(len(rdfs[mass_frac_indices[0]].rdf_oxy_names)) ] # Vary by atom solute name
            return
    ## CLOSING ALL FIGURES
    plt.close('all')    
            
    multi_rdf = multi_plot_rdf(rdfs = rdfs,
                               names = list_of_pickles,
                               decode_type = 'solvent_effects',
                               )
    
    multi_rdf.plot_rdf_solute_solvent_multiple_mass_frac()
    
    
    

    #%%

    ## PLOTTING
    multi_rdf.plot_rdf_solute_oxy_to_solvent_multiple_mass_frac(True)
    
    
    