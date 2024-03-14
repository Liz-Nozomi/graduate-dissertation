# -*- coding: utf-8 -*-
"""
extract_self_assembly_structure_multi_traj_analysis.py
The purpose of this script is to analyze the data from the multi_traj_analysis_tool.py

Created on: 03/25/2018

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""
### GENERAL MODULES
import matplotlib.pyplot as plt

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add, csv_dict
### IMPORTING FUNCTION TO GET THE DECODED TYPE
from MDDescriptors.core.decoder import decode_name

### IMPORTING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot


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
        ax.set_xlabel('Simulation time (ns)', **LABELS )
        ax.set_ylabel('Number of adsorbed thiols', **LABELS)
        
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
    def __init__(self, structure, pickle_name, decoder_type='self_assembly_np'):
        ## STORING STRUCTURE
        self.structure = structure
        
        ## STORING INFORMATION FOR CSV
        self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type)
    
        ## FINDING SPECIFIC INFORMATION ABOUT SYSTEM
        self.find_num_ligand_per_frame()
        
        ## FINDING LIGAND DENSITY
        self.find_ligand_density()
        
        ## ADDING GENERAL BOX STUFF
        self.csv_dict.add( data_title = 'box_length', data = [self.structure.traj_info.ens_length] ) 
        
        ## ADDING FACET TO SURFACE AREA
        self.find_facet_to_surface_ratio()
        
        ## STORING CSV INFO
        self.csv_info = self.csv_dict.csv_info
    
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
        self.csv_dict.add( data_title = title,  data = [x, y], labels = labels)
        # self.csv_info = csv_info_add(self.csv_info, title, [x,y], labels )
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
        ## FINAL DIAMETER
        self.csv_dict.add( data_title = 'final_diameter(nm)', data = [self.structure.gold_diameter] )
        
        ## TOTAL NUMBER OF GOLD ATOMS
        self.csv_dict.add( data_title = 'total_gold_atoms', data = [self.structure.total_gold] )
        
        ## TOTAL NUMBER OF EXCESS LIGANDS
        self.csv_dict.add( data_title = 'total_excess_ligands', data = [self.structure.total_sulfur] )
        
        ## FINAL NUMBER OF ADSORBED LIGANDS
        self.csv_dict.add( data_title = 'final_number_adsorbed_ligands', data = [self.structure.num_gold_sulfur_bonding_per_frame[-1]] )
        
        ## SURFACE AREA
        self.csv_dict.add( data_title =  'surface_area_spherical(nm2)', data = [self.structure.surface_area_spherical ] )
        self.csv_dict.add( data_title =  'surface_area_hull(nm2)', data = [self.structure.surface_area_hull ] )
        
        ## ANGSTROMS^2 PER LIGAND
        self.csv_dict.add( data_title = 'area_angs_per_ligand_spherical(A2/lig)', data = [self.structure.area_angs_per_ligand_spherical] )
        self.csv_dict.add( data_title = 'area_angs_per_ligand_hull(A2/lig)', data = [self.structure.area_angs_per_ligand_hull] )
        
        ## LIGAND PER NM2
        self.csv_dict.add( data_title = 'ligand_per_area_spherical(lig/nm2)', data = [self.structure.ligand_per_area_spherical] )
        self.csv_dict.add( data_title = 'ligand_per_area_hull(lig/nm2)', data = [self.structure.ligand_per_area_hull] )        
        
        return
    
    ### FUNCTION TO FIND FACET TO SURFACE RATIO
    def find_facet_to_surface_ratio(self):
        ''' This script finds the total facet atom and divides it by the total surface atoms '''
        
        ## FINDING GOLD SURFACE AND FACET INDICES
        gold_surface_indices = self.structure.gold_gold_coordination['gold_surface_indices']
        gold_facet_indices = self.structure.gold_gold_surface_facet_edge_dict['gold_surface_facet_index']
        
        ## FINDING RATIO FACET TO SURFACE
        facet_to_surface_atom = len(gold_facet_indices)/len(gold_surface_indices)
        
        ## INCLUDING TO CSV INFO
        self.csv_dict.add( data_title = 'facet_to_surface_gold_atoms', data = [ facet_to_surface_atom ] ) 
        
        return 

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    from MDDescriptors.application.nanoparticle.self_assembly_structure import self_assembly_structure
    ## DEFINING CLASS
    Descriptor_class = self_assembly_structure
    
    ## DEFINING DATE
    Date='180920' # '180814-FINAL'
    
    ## DEFINING DESIRED DIRECTORY
    # Pickle_loading_file=r"mdRun_433.15_6_nm_ACE_50_WtPercWater_spce_dmso"
    Pickle_loading_file=r"Planar_10x10_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_1"
    
    ## IMPORTING PLOTTING TOOLS
    # from MDDescriptors.application.nanoparticle.plot_self_assembly_structure import plot_self_assembly_structure, extract_self_assembly_structure
    
    #%%
    #### SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )    

    #%%
    
    ### PLOTTING
    plot_self_assembly_structure(multi_traj_results)
    
    ### EXTRACTING CSV
    extracted_structure = extract_self_assembly_structure(multi_traj_results, Pickle_loading_file)
    
    #%%
    
    from MDDescriptors.core.csv_tools import csv_info_export
    
    exported_csv = csv_info_export(extracted_structure.csv_info, ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'])
    
    #%%
    
    #%%
    ##### MULTI TRAJ ANALYSIS
    traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    
    
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    from MDDescriptors.core.csv_tools import multi_csv_export
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, # ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'],
                    export_class = extract_self_assembly_structure,
                    export_text_file_name = 'extract_self_assembly_structure',
                    )
    #%%
    # df_1.to_csv('test.csv')
    [ each.to_csv(str(index)+'.csv') for index, each in enumerate(dfs_2)]
    # dfs_2.to_csv('test2.csv')
    
    
    #%%
    name = decode_name(list_of_pickles[0], 'self_assembly_np')
    
    
    
    LINE_STYLE=['-','--', ':', '-.']
    LINE_COLOR=['k', 'b', 'r', 'g', 'm', 'c']
    
    ### FUNCTION TO FIND THE CURRENT LINE STYLE
    def find_line_styles(list_of_pickles_decoded, vary_lines_dict):
        '''
        The purpose of this function is to give each of the pickle a line style based on the decoding
        INPUTS:
            list_of_pickles_decoded: list of pickles that were decoded. Basically, it is a list with dictionary items, e.g.:
                [{'diameter': 2.0, ...}, ... ]
            vary_lines_dict: dictionary that links to the keys within the list of pickles. Then, following that, we will have a list of things to vary by (e.g. line_color, etc.)
                e.g.: { 'diameter': ['line_color', 'line_style' ]}
        OUTPUTS:
            list corresponding to the line styles
        '''
        ## FINDING TOTAL LIST
        total_list=len(list_of_pickles_decoded)
        ## START BY CREATING EMPTY OF DICTIONARIES
        line_styles = [{}] *total_list
        ## CREATING DICTIONARY FOR UNIQUE IDS
        unique_id={}
        ## LOOPING THROUGH EACH LIST TO GENERATE UNIQUE ID'S
        for each_item in range(total_list):
            ## NOW, CHECKING IF THE DICTIONARY HAS THE ITEMS WE CARE ABOUT
            keys_within_dicts = [ current_keys for current_keys in list_of_pickles_decoded[each_item].keys() if current_keys in vary_lines_dict.keys() ]
            ## LOOPING THROUGH EACH KEY
            for each_key in keys_within_dicts:
                ## FINDING CORRESPONDING VALUE TO THE DICTIONARY
                dict_value = list_of_pickles_decoded[each_item][each_key]
                ## SEE IF THIS KEY IS WITHIN UNIQUE ID
                if each_key not in unique_id.keys():
                    ## DEFINING CODE
                    unique_code = 0
                    ## CREATE A KEY
                    unique_id[each_key] = [[dict_value, unique_code]] # Dictionary value and its corresponding code
                ## IF DICTIONARY KEY EXISTS, THEN LET'S SEE IF THE DICTIONARY VALUE IS ALREADY INSIDE
                else:
                    ## SEEING IF THE VALUE IS WITHIN THE IDS
                    current_values =[each_key_value[0] for each_key_value in unique_id[each_key] ] 
                    current_keys = [each_key_value[1] for each_key_value in unique_id[each_key] ] 
                    if dict_value not in current_values:
                        ## ADD THE VALUE, WITH A NEW KEY VALUE
                        unique_id[each_key].append([dict_value, current_keys[-1]+1 ])
        
        ## GETTING KEYS FROM UNIQUE ID
        unique_id_keys = unique_id.keys()
        
        ## LOOPING THROUGH EACH ITEM AND MATCHING THE KEYS FROM UNIQUE_ID
        for each_item in range(total_list):
            ## DEFINING DECODED
            decoded_name=list_of_pickles_decoded[each_item]            
            ## DEFINING LINE STYLES
            item_line_styles = {}         
            ## LOOPING THROUGH EACH UNIQUE ID KEY
            for each_unique_id_key in unique_id_keys:
                ## FINDING TYPES TO VARY
                vary_types = vary_lines_dict[each_unique_id_key]
                print(vary_types)
                ## VARYING TYPES
                for each_vary_type in vary_types:   
                    ## FINDING THE INDEX WITHIN UNIQUE KEYS THAT WE ARE CURRENTLY USING (e.g. 2 nm diameter, etc.)
                    each_value = [each_list[0] for each_list in unique_id[each_unique_id_key]]
                    # print(each_value)
                    index_of_key = each_value.index( decoded_name[each_unique_id_key] )
                    ## FINDING INDEX NOW OF THAT KEY
                    index_unique_key = unique_id[each_unique_id_key][index_of_key][1]
                    ## ADDING TO THE DICTIONARY
                    item_line_styles = find_line_details(item_line_styles, index_unique_key, each_vary_type )
                
                ## STORING
                line_styles[each_item] = item_line_styles.copy()
        
        return line_styles
    
    ### FUNCTION TO FIND THE LINE DETAILS BASED ON THE INDEX, ETC.
    def find_line_details(line_dict, index, vary_type, LINE_STYLE = LINE_STYLE, LINE_COLOR = LINE_COLOR):
        '''
        The purpose of this function is to take the index, and the type, then output to a dictionary
        INPUTS:
            line_dict: dictionary for the current line
            index: index of the type
            vary_type: types to vary, e.g.
                line_color: vary by line color
                line_style: vary by line style
            LINE_STYLE: List of all possible line styles
            LINE_COLOR: List of all possible line colors
        OUTPUTS:
            line_dict: Updated line dictionary
        '''
        if vary_type == 'line_color':
            try:
                line_dict['color']=LINE_COLOR[index]
            except Exception:
                pass
        elif vary_type == 'line_style':
            try:
                line_dict['linestyle']=LINE_STYLE[index]
            except Exception:
                pass
        else:
            print("Error, no line type of %s is found, are you sure it is added?"%(vary_type))

        return line_dict

    ### FUNCTION TO DEFINE A LABEL
    def define_plot_label(pickle_decoded_name, desired_labels):
        '''
        The purpose of this function is to output a string label for a plot
        INPUTS:
            pickle_decoded_name: Name of the pickle that is decoded
                [{'diameter': 2.0, ...}, ... ]
            desired_labels: list of labels you want that should match the list of pickles
        OUTPUTS:
            label: string that you can output into a plot
        '''
        ## LOOPING THROUGH EACH DESIRED LABELS
        if len(desired_labels) > 0:
            ## CREATING BLANK LABEL
            label =''
            for index, each_desired_labels in enumerate(desired_labels):
                ## GETTING OUTPUT
                try:
                    current_label_addition = str( pickle_decoded_name[each_desired_labels] )
                except:
                    current_label_addition = '' # If no label is added
                ## ADDING TO CURRENT LABEL
                label = label + current_label_addition
                ## ADDING SPACE IF NECESSARY
                if index != len(desired_labels)-1 and len(label) != 0:
                    if label[-1] != ' ':
                        label += ' '            
        ## NO DESIRED LABELS, THUS NO 
        else:
            label = None
        
        return label
        

    
    #######################################################################
    ### CLASS FUNCTION TO PLOT MULTIPLE TRAJECTORIES INTO A SINGLE PLOT ###
    #######################################################################
    class multi_traj_plot:
        '''
        The purpose of this class is to take a plotting class, such as plot_rdf, etc. and combine those plots into one single plot. The idea here is that we will take the inputs of those plots and feed their output ax/fig to the next plot. We will update the color accordingly!
        INPUTS:
            traj_results: list of results as a list given by "load_multi_traj_pickles"
            list_of_pickles: list of opickles from the "load_multi_traj_pickles"
            plot_class: plotting class/function
                NOTE: We assume that the plotting function has a fig and ax associated with it! 
            vary_lines_dict: Dictionary to be able to  vary line information
            decoder_type: way to decode your directory name
            desired_labels: labels that you want outputted onto the plot
        OUTPUTS:
            
        ALGORITHM:
            - Start by plotting and using a for loop for each plot
            - Create a label for the plot
            - Loop until all pickles are complete
        '''
        ### INITIALIZATION
        def __init__(self, traj_results, list_of_pickles, plot_class, vary_lines_dict, decoder_type=None, desired_labels = None ):
            ## DECODING ALL THE PICKLES
            self.list_of_pickles_decoded= [ decode_name(each_pickle, decoder_type)  for each_pickle in list_of_pickles ]
            ## FINDING ALL LINE DETAILS
            self.line_styles = find_line_styles(self.list_of_pickles_decoded, vary_lines_dict)
            
            ## DEFINING FIG AND AX
            fig = None; ax = None; label_storage = []
            
            ## LOOPING THROUGH EACH PICKLE AND USING PLOTTING FUNCTION
            for index, each_pickle in enumerate(list_of_pickles):
                ## PRINTING
                print("PLOTTING FOR %s"%(each_pickle))
                ## DEFINING CURRENT LINE STYLE
                current_line_style = self.line_styles[index]
                ## DEFINING A LABEL
                label = define_plot_label(self.list_of_pickles_decoded[index], desired_labels)
                ## TURNING OFF LABELS THAT ARE BASIC REPLICATES
                if label in label_storage:
                    label = None
                else:
                    label_storage.append(label)
                ## PLOTTING
                plot_results = plot_class(traj_results[index], fig = fig, ax = ax, label=label, current_line_style = current_line_style)
                ## SAVING THE FIG AND AX
                fig = plot_results.fig
                ax = plot_results.ax
            
            ## STORING FIGURE AND AX
            self.fig = fig
            self.ax = ax
            
            return
    
    
    ### DEFINING DICTIONARY TO VARY LINE COLORS
    vary_lines_dict = {
                    'diameter': ['line_color'],# , 'line_style'
                     'shape': ['line_style']
                           }    
    
    multi_plot = multi_traj_plot( traj_results, list_of_pickles,                # List of pickle names, essentially list of identification modes
                                 plot_class =  plot_self_assembly_structure,    # Plotting class that simply plots and outputs
                                 vary_lines_dict = vary_lines_dict,             # Which lines do you want to vary? (i.e. diameter, etc.)
                                 decoder_type = 'self_assembly_np',             # Decoding type for the names of the pickles
                                 desired_labels = ['diameter','shape'] ,        # Desired labels that you want to be shown on the plot (empty if none!)
                                 )
    # Put a legend to the right of the current axis
    multi_plot.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    multi_plot.fig.savefig('selfassembly.png', bbox_inches='tight')
    
    #%%

    

    line_styles, unique_id = find_line_styles(multi_plot.list_of_pickles_decoded, vary_lines_dict)
    
        
    
    
    
    
    
    #%%
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
    
    ### IMPORTING MODULES
    import matplotlib.pyplot as plt
    from MDDescriptors.core.plot_tools import save_fig_png # code to save figures
    
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
            ax.set_xlabel('r (nm)',**LABELS)
            ax.set_ylabel('Radial Distribution Function',**LABELS)
            
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
                        for solvent_index,each_solvent_name in enumerate(self.rdfs[mass_frac_indices[0]].solvent_name):
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


    #%%
    ## PLOTTING
    multi_rdf.plot_rdf_solute_oxy_to_solvent_multiple_mass_frac(True)
    
    
    