k#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraction_tools.py
The purpose of this function is to generate extraction tools that could be used 
post MDDescriptor simulations. 

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

## IMPORTING CORE MODULES
import glob
import sys
import os
import pandas as pd
import numpy as np
## MATPLOTLIB
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
## SKLEARN
from sklearn.preprocessing import MinMaxScaler
## COMMAND TO STANDARDIZE
from sklearn.preprocessing import StandardScaler
## PCA
from sklearn.decomposition import PCA

## IMPORTING LINEAR MODEL
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

## GETTING PEARSON'S R
from scipy.stats import pearsonr

## IMPORTING STATISTICS MODULE
from scipy import stats

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import DEFAULT_IMAGE_OUTPUT

## IMPORTING CUSTOM MODULES
import MDDescriptors.core.pickle_tools as pickle_tools
## PLOT TOOLS
import MDDescriptors.core.plot_tools as plot_tools

## ADDING DECODER FUNCTIONS
import MDDescriptors.core.decoder as decoder

## ADDING PATH FUNCTIONS 
import MDDescriptors.core.check_tools as check_tools
## ADDING PRINT FUNCTIONS
import MDDescriptors.core.print_tools as print_tools

## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.application.nanoparticle.extract_gmx_hbond import analyze_gmx_hbondcbar_kw
from MDDescriptors.application.nanoparticle.extract_gmx_principal import analyze_gmx_principal
from MDDescriptors.application.nanoparticle.extract_gmx_rmsf import analyze_gmx_rmsf
from MDDescriptors.application.nanoparticle.extract_gmx_gyrate import analyze_gmx_gyrate
## SASA
from MDDescriptors.application.nanoparticle.nanoparticle_sasa import nanoparticle_sasa

## IMPORTING CM 2 INCH
from MDDescriptors.core.plot_tools import cm2inch


## DEFINING DESIRED COLOR MAP
CMAP = 'cool'
# 'hot'
# 'jet'

## DEFINING CSV DICT
#PATH_CSV = r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\NP_membrane_binding_descriptors_manuscript\Excel\csv_files"
PATH_CSV = r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_membrane_binding_descriptors_manuscript/Excel/csv_files"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\NP_membrane_binding_descriptors_manuscript\Excel\csv_files"

## DEFINING IMAGE OUTPUT PATH
OUTPUT_IMAGE_PATH = r"/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Research_Presentations/20200921-Dow_interview/presentation/Images/np_project/images_from_descriptors"
# "/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200720/images"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\NP_membrane_binding_descriptors_manuscript\Figures\Figure_X_Predictions"

## DEFINING FIGURE EXTENSION
FIG_EXTENSION = 'png'

## SETTING DEFAUTLS
plot_tools.set_mpl_defaults()

### FUNCTION TO EXTRACT NP SASA
def extract_nanoparticle_sasa_value(sasa):
    '''
    This function extracts information from nanoparticle sasa
    '''
    return np.mean(sasa.sasa['all_atom_ligands'])

### CLASS ATTRIBUTES THAT ARE NECESSARY
DESCRIPTOR_EXTRACTION_DICT = {
        'analyze_gmx_hbond' : ['avg_h_bond'],
        'analyze_gmx_principal': ['moi_extract', 'eccentricity_avg'],
        'analyze_gmx_rmsf': ['avg_rmsf'],
        'analyze_gmx_gyrate': ['avg_radius_gyrate'],
        'nanoparticle_sasa': extract_nanoparticle_sasa_value,
        }

# results.sasa['all_atom_ligands']
### FUNCTION TO GET ATTRIBUTE OF A LIST
def get_class_attribute( class_object, attribute_list ):
    '''
    The purpose of this function is to get the attribute based on a list
    INPUTS:
        class_object: [obj]
            class object
    attribute_list: [list]
        list of attribute, assuming that your class object has deep layers
    OUTPUTS:
        current_obj: results
    '''
    ## DEFINING OBJECT
    current_obj = class_object
    ## LOOPING
    for each_attribute in attribute_list:
        current_obj = current_obj.__getattribute__(each_attribute)
    
    return current_obj

### FUNCTION TO MOVE COLUMNS TO END
def dataframe_move_cols_to_end(df,
                               cols_at_end):
    '''
    The purpose of this function is to move columns to the end.
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe
        cols_at_end: [list]
            list of columns at the end
    OUTPUTS:
       df : [pd.dataframe]
            pandas dataframe with cols at end
    '''
    df = df[[c for c in df if c not in cols_at_end] 
                + [c for c in cols_at_end if c in df]]
    return df

### CLASS TO EXTRACT FROM OTHER CLASSES
class class_extraction:
    '''
    The purpose of this function is to extract from classes
    INPUTS:
        descriptor_extraction_dict: [dict]
            dictionary to extract information from descriptors
            ## TODO
            Update this class
            run descriptor extraction
            generate heat map
    '''
    ## INITIALIZING
    def __init__(self,
                 descriptor_extraction_dict = DESCRIPTOR_EXTRACTION_DICT):
        
        ## STORING
        self.descriptor_extraction_dict = descriptor_extraction_dict
        
        return
    
    ## FUNCTION TO EXTRACT VALUE
    def extract_class(self, class_obj):
        '''
        The purpose of this function is to extract a single value from a class 
        given a dictionary
        INPUTS:
            class_obj: [obj]
                class object
            func: [func]
                function to run your class object
        OUTPUTS:
            value: [float]
                value coming out of the class
        '''
        ## FINDING ATTRIBUTE YOU WOULD LIKE
        attribute = self.descriptor_extraction_dict[class_obj.__class__.__name__]
        
        ## SEEING IF YOU HAVE A FUNCTION
        if type(attribute) == list:
            ## FINDING VALUE
            value = get_class_attribute( class_object = class_obj,
                                         attribute_list = attribute)
        elif callable(attribute) is True:
            ## FINDING VALUE
            value = attribute( class_obj )
        
        return value


### FUNCTION TO ORDER DF
def order_df(df,
             ordered_classes,
             col_name = 'solute',
             ):
    '''
    This function orders a dataframe based on an input list
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe
        col_name: [str]
            column name
        ordered_classes: [list]
            ordered classes
    OUTPUTS:
        ordered_df: [pd.dataframe]
            ordered pandas dataframe based on your input list. Note that 
            this code only outputs the information given as a list
    '''
    ## CREATING EMPTY DF LIST
    df_list = []

    for i in ordered_classes:
       df_list.append(df[df[col_name]==i])
    
    ordered_df = pd.concat(df_list)
    return ordered_df
        

### FUNCTION TO MATCH ROWS AND COPY OVER
def reorganize_rows_by_column_matching(df,
                                       column_identifier = 'identifier',
                                       match_prefix = 'switch',
                                       match_column_name = 'cosolvent',
                                       column_matching = [ 'diameter', 'ligand', 'shape', 'temperature', 'trial']):
    '''
    The purpose of this function is to take a dataframe and reorganized based on
    some column matching criteria. Example:
        Original dataframe
                                                       identifier  ...  principal
            0   EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_T...  ...   0.101491
            1   EAM_300.00_K_2_nmDIAM_ROT005_CHARMM36jul2017_T...  ...   0.057734
            2   EAM_300.00_K_2_nmDIAM_ROT006_CHARMM36jul2017_T...  ...   0.128770
            3   EAM_300.00_K_2_nmDIAM_ROT010_CHARMM36jul2017_T...  ...   0.141012
            4   EAM_300.00_K_2_nmDIAM_ROT011_CHARMM36jul2017_T...  ...   0.104375
            5   EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_T...  ...   0.046140
            6   EAM_300.00_K_2_nmDIAM_ROT014_CHARMM36jul2017_T...  ...   0.053450
            7   switch_solvents-50000-dmso-EAM_300.00_K_2_nmDI...  ...   0.096695
        New dataframe
                                                      identifier  ...  dmso_hbond
            0  EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_T...  ...         NaN
            1  EAM_300.00_K_2_nmDIAM_ROT005_CHARMM36jul2017_T...  ...         NaN
            2  EAM_300.00_K_2_nmDIAM_ROT006_CHARMM36jul2017_T...  ...         NaN
            3  EAM_300.00_K_2_nmDIAM_ROT010_CHARMM36jul2017_T...  ...         NaN
            4  EAM_300.00_K_2_nmDIAM_ROT011_CHARMM36jul2017_T...  ...         NaN
            5  EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_T...  ...         NaN
            6  EAM_300.00_K_2_nmDIAM_ROT014_CHARMM36jul2017_T...  ...         NaN
            
        Notice that the number of identifiers decreased and we have included the results as columns.
    INPUTS:
        df: [pd.dataframe]
            panda dataframe of interest
        column_identifier: [str]
            string that matches the identifier
        match_prefix: [str]
            some matching prefix, in this example it is switch
        match_column_name: [str]
            some column name within dataframe that is used to distinguish the details. This is also used 
            to modify the names of the items.
        column_matching: [list]
            list of column matching criterias to ensure that you are matching the correct rows.
    OUTPUTS:
        descriptor_copy: [pd.dataframe]
            dataframe with the updated columns and rows that were matched. Columns are renamed according 
            to 'column_matching' variable.
    '''    
    ## MATCHING INDEX
    matched_index = descriptors_df[column_identifier].str.contains(match_prefix)
    
    ## KEEPING NON MATCHING INDEXES
    descriptor_copy = descriptors_df[~matched_index].copy()
        
    ## LOOPING THROUGH EACH DATAFRAME
    for idx, each_row in descriptors_df[matched_index].iterrows():
        ## DEFINING MATCHING COLUMNS
        matching_columns = each_row[column_matching]
        ## FINDING INDEX MATCHED
        matched_index = descriptor_copy.index[(descriptor_copy[column_matching] == matching_columns).all(axis=1)==True].tolist()
        ## GETTING ALL THE VALUES
        values = each_row[descriptors_dict.keys()]
        ## RENAMING EACH ROW
        values_dict = values.to_dict()
        values_dict_rename = { each_row[match_column_name] + '_' + each_key: values_dict[each_key] for each_key in values_dict}
        ## APPENDING TO LIST
        for each_key in values_dict_rename:
            descriptor_copy.at[matched_index[0],each_key] = values_dict_rename[each_key]
    return descriptor_copy

    
### FUNCTION TO DROP ANY COLUMNS THAT ARE MISSING
def pd_drop_na_columns(df):
    ''' This function drops any missing n/a columns'''
    return df.dropna(axis='columns')


### FUNCTION TO CHECK AXIS LIMITS
def check_ax_limits(ax, fig_limits = None):
    '''
    The purpose of this function is to check axis limits
    INPUTS:
        ax: [obj]
            axis object
        fig_limits: [dict]
            dicitonary of fig limits, e.g. :
                {'x': [-0.5, 7.5],
                 'y': [-0.5, 7.5]}
            This means x ranges from -0.5, 7.5
    OUTPUTS:
        ax: [obj]
            updated axis object
    '''
    
    ## CHECKING LIMITS
    if fig_limits is not None:
        ## SETTING X LIMITS
        if 'x' in fig_limits.keys():
            ax.set_xlim( fig_limits['x'][0], fig_limits['x'][1] )
            
            ## SETTING X TICKS
            ax.set_xticks( np.arange( fig_limits['x'][0], 
                                      fig_limits['x'][1] + fig_limits['x'][2], 
                                      step=fig_limits['x'][2])  )
        if 'y' in fig_limits.keys():
            ax.set_ylim( fig_limits['y'][0], fig_limits['y'][1] )
            
            ## SETTING Y TICKS
            ax.set_yticks( np.arange( fig_limits['y'][0], 
                                      fig_limits['y'][1] + fig_limits['y'][2], 
                                      step=fig_limits['y'][2])  )
    return ax

### FUNCTION TO PLOT SASA OVER TIME
def plot_sasa_over_time(results,
                        fig_size_cm = (8.3, 8.3),
                        fig_limits = None,
                        fig = None,
                        ax = None,
                        ):
    '''
    The purpose of this function is to plot the sasa over time.
    INPUTS:
        results: [obj]
            results object from nanoparticle_sasa class
        fig_size_cm: [tuple]
            figure size in cm
        fig_limits: [dict]
            dicitonary of fig limits, e.g. :
                {'x': [-0.5, 7.5],
                 'y': [-0.5, 7.5]}
            This means x ranges from -0.5, 7.5
    OUTPUTS:
        fig, ax: figure and axis
    '''
    
    ## DEFINING SASA VALUE
    sasa_values = results.sasa['all_atom_ligands']
    time_step = np.arange(0, len(sasa_values))
    
    ## CREATING FIGURE
    ## CONVERTING TO INCHES
    if fig is None or ax is None:
        figsize=cm2inch( *fig_size_cm )
        ## PLOTTING
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1) 
    
    ## GETTING AVERAGE VALUE
    avg_result = np.mean(sasa_values)
    
    ## PLOTTING
    ax.plot(time_step, sasa_values, '-', color='k')
    
    ## CHECKING LIMITS
    ax = check_ax_limits(ax, fig_limits)
    
    ## PLOTTING AVERAGE VALUE
    ax.axhline(y = avg_result, linestyle = '--', color='r', label="Avg")
    
    ## ADDING AXIS LEGEND
    ax.set_xlabel("Frame")
    ax.set_ylabel("SASA (nm$^2$)")
    
    ## ADDING LEGEND
    ax.legend()
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax


### FUNCTION TO RELABEL
def add_label_to_dict(df,
                      name_labels,
                      column_name = 'ligand',
                      label_name = 'label'):
    '''
    The purpose of this function is to label the dictionary by adding a new 
    column.
    INPUTS:
        df: [dataframe]
            pandas dataframe
        name_labels: [dict]
            dictionary containing the labels you want to include, e.g.
                ligand_name_labels = {
                        'ROT001' : 'C1',
                        'ROT010' : 'C2', 
                        'ROT011' : 'C4',
                        'ROT005' : 'C6',
                        'ROT014' : 'C8',
                        'ROT012' : 'C10',
                        'ROT006' : 'CYC',
                        }
        column_name: [str]
            column name of reference
        label_name: [str]
            label name
    OUTPUTS:
        df: [dataframe]
            Updated dataframe with a new column that is referenced
        
    '''
    ## LOOPING THROUGH LIGAND NAMES
    for idx, name in enumerate(name_labels):
        df.loc[df[column_name] == name,label_name] = name_labels[name]
    return df

### FUNCTION TO PLOT HEAT MAP
def plot_heat_map(descriptors_df_norm,
                  y_labels,
                  descriptor_keys,
                  cmap = 'bwr',
                  descriptor_actual_values = None,
                  figsize = None,
                  title = "Normalized descriptors for ROT ligands",
                  want_color_bar = False,
                  want_text = True,
                  cbarlabel = "",
                  cbar_kw = {},
                  cbar_y_tick_labels = None,
                  text_font_size = 8,
                  want_minor = True,
                  ):
    '''
    The purpose of this function is to plot the heat maps.
    Resource: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    INPUTS:
        descriptors_df_norm: [df]
            dataframe containing data
        y_labels: [list]
            list of y labels
        descriptor_keys: [list]
            list of descriptor keys
        cmap: [str]
            color map desired
        descriptor_actual_values: [df]
            actual values for descriptors if you want to print it out
        want_color_bar: [logical]
            True if you want color bar
        cbarlabel: [str]
            label for the colorbar
        cbar_kw: [dict]
            dictionary inputs for the color  bar
        text_font_size: [float]
            font size of text
        cbar_y_tick_labels: [list]
            list of y tick labels 
        want_text: [logical]
            True if you want text of each value
        want_minor: [logical]
            True if you want minor 
    OUTPUTS:
        fig, ax
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ## CREATING FIGURE AND AXIS
    fig, ax = plt.subplots(figsize = figsize)
    
    ## DEFINING NORMALIZED
    descriptor_normalized_values = np.array(descriptors_df_norm[descriptor_keys])
    
    ## PLOTTING
    im = ax.imshow(descriptor_normalized_values,
                   cmap = cmap)
    
    ## GETTING LIGAND NAMES
    y_labels = y_labels
    
    ## GETTING DESCRIPTOR NAMES
    x_labels = descriptor_keys
    
    ## SHOWING X AND Y TICKS
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    
    ## SEETTING X AND Y LABELS    
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
        
    ## SEEING IF THERE IS ACTUAL VALUES
    if descriptor_actual_values is None:
        descriptor_actual_values_fig = descriptor_normalized_values
    else:
        descriptor_actual_values_fig = np.array(descriptor_actual_values[descriptor_keys])
    
    if want_text is True:
        ## ADDING TEXT ANNOATIONS
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                ## DEFINING  VALUE
                value = descriptor_actual_values_fig[i, j]
                if value > 100:
                    ## DEFINING TEXT TO ADD
                    text_to_add = "%.0f"%(value)
                else:
                    text_to_add = "%.2f"%(value)
                ## ADDING TEXT
                ax.text(j, i, text_to_add,
                        ha="center", va="center", color="k", fontsize = text_font_size)
            
        
    ## ADDING COLOR BAR
    
    ## MAKES COLOR BAR SAME SIZE
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, ax=ax, cax = cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    if cbar_y_tick_labels is not None:
        cbar.ax.set_yticklabels(cbar_y_tick_labels)
        
    # Turn spines off and create white grid.
#    for edge, spine in ax.spines.items():
#        spine.set_visible(False)

    ## ROTATING X AXIS
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
    
    ## GETTING TICKS
    ax.set_xticks(np.arange(len(x_labels))-.5, minor=want_minor)
    ax.set_yticks(np.arange(len(y_labels))-.5, minor=want_minor)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
#    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
#    ax.tick_params(which="minor", bottom=False, left=False)
    
    ## SETTING TITLE
    if title is not None:
        ax.set_title(title)        
    ## TIGHT LAYOUT
    fig.tight_layout()
    ## SHOWING FIGURE
    plt.show()
    return fig, ax

### FUNCTION TO LOAD ALL DESCRIPTORS
def load_descriptors(descriptors_dict,
                     pickle_path,
                     interest_dict,
                     decode_type = 'nanoparticle',
                     ):
    '''
    The purpose of this function is to load all the descriptors.
    INPUTS:
        descriptors_dict: [dict]
            dictionary that contains the analysis molecular descriptors
        pickle_path: [str]
            pickle path
        interest_dict: [dict]
            dictionary of keys that you are interesed in. These should match 
            the decoder method
        decode_type: [str]
            way to decode the information
    OUTPUTS:
        descriptors_df: [df]
            dataframe of extracted descriptors for each file , e.g.
                                                           identifier  ...  cosolvent
                0   EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_T...  ...        NaN
                1   EAM_300.00_K_2_nmDIAM_ROT002_CHARMM36jul2017_T...  ...        NaN
    '''
    ## CREATING EMPTY DICTIONARY
    descriptors_df = pd.DataFrame(columns = ['identifier'] + list(descriptors_dict.keys())  )

    ## LOOPING THROUGH DESCRIPTOR DICT
    for each_descriptor in descriptors_dict:
        ## DEFINING DICT NAME
        descriptor_name = descriptors_dict[each_descriptor].__name__
        ## FINDING ALL FILES
        file_list = glob.glob(os.path.join(pickle_path, descriptor_name + '/*') )
        ## GETTING BASENAME
        file_list_basename = [ os.path.basename(each_file) for each_file in file_list]
        ## PRINTING
        if verbose is True:            
            print("--- Working on descriptor: %s ---"%(descriptor_name) )
        
        ## LOOPING THROUGH EACH DESCRIPTOR
        for each_file in file_list_basename:
            ## DEFINING WANT STORAGE
            want_stored = False
            ## DECODING NAME
            name = decoder.decode_name(name = each_file,
                                       decode_type = decode_type)
            ## SEEING IF INTEREST IN SPECIFIC VALUES
            if bool(interest_dict) is True:
                ## FINDING KEYS
                name_values = { each_key : name[each_key] for each_key in interest_dict }
                ## DEFINING TRUTH ARRAY
                truth_array = [True if name_values[each_key] in interest_dict[each_key] else False for each_key in interest_dict ]
                want_stored = np.all(truth_array)
                
            else:
                ## STORING IF NO INTEREST SPECIFIED
                want_stored = True
                
            #######################
            ### RUNNING STORAGE ###
            #######################
            if want_stored == False:
                ## PRINTING
                print("   ***Skipping: %s"%(each_file))
            else:
                ## PRINTING
                print("   Storing: %s"%(each_file))
                
                ## LOADING THE PICKLE
                results = pickle_tools.load_pickle_from_descriptor(pickle_dir = pickle_path,
                                                                   descriptor_class = descriptors_dict[each_descriptor],
                                                                   pickle_name = each_file,
                                                                   verbose = False
                                                                   )
                ## GETTING THE RESULTS
                value = extraction.extract_class(class_obj = results)
                
                ## ADDING UNIQUE IDENTIFIER
                name['identifier'] = each_file
                
                ## ADDING TO NAME
                name[each_descriptor] = value
                ## CONVERTING TO DATAFRAME
                ## SEEING IF DATABASE ALREADY HAS UNIQUE IDENTIFIER
                if (descriptors_df['identifier'] == each_file).any():
                    ## PRINTING
                    # print("adding to idx and column")
                    ## FIND INDEX, ADD COLUMN
                    idx = descriptors_df.index[descriptors_df['identifier'] == each_file].tolist()[0]
                    ## ADDING VLAUE
                    descriptors_df.loc[idx, each_descriptor] = value
                else:                
                    ## APPENDING TO DATAFRAME
                    descriptors_df = descriptors_df.append(name, ignore_index = True)
    return descriptors_df

### FUNCTION TO RUN PRINCIPAL COMPONENT ANALYSIS
def run_principal_component_analysis(descriptors_df_reorg,
                                     descriptor_keys,
                                     n_components = 2,):
    '''
    The purpose of this function is to run PCA given a dataframe object.
    First, this code will standardize all descriptors by transforming them 
    to arrays. Then, the PCA code will run across all descriptors
    ## SEE REF: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    ## REF SHOWS VARIABLES ON THE PLOT: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
    INPUTS:
        descriptors_df_reorg: [df]
            dataframe containing all the data
        descriptor_keys: [list]
            list of descriptors
        n_components: [int]
            number of components to analyze
    OUTPUTS:
        finalDf: [df]
            dataframe after principal component analysis
                     pc1       pc2     label
            0   1.622689 -0.508636        C1
            1   1.292417  0.260971        C2
            2   0.187438 -0.686356        C4
        pca: [obj]
            pca output 
        pca_columns: [list]
            list of pca columns
    '''
    ## RESETTING INDEX
    descriptors_df_reorg = descriptors_df_reorg.reset_index(drop=True)
    
    ## PERFORMING PCA ANALYSIS
    descriptor_array = descriptors_df_reorg.loc[:, descriptor_keys].values
    
    # Standardizing the features
    descriptor_array_standardized = StandardScaler().fit_transform(descriptor_array)
    
    ## SEPARATING TARGETS
    # Separating out the target
    # y = descriptors_df_reorg.loc[:,['label']].values
    
    # np.array(descriptors_df_reorg[descriptor_keys])
    ## DEFINING PCA
    pca = PCA(n_components=n_components)
    
    ## GETTING PRINCIPAL COMPONENTS
    principalComponents = pca.fit_transform(descriptor_array_standardized)
    
    ## COLUMNS
    pca_columns = ['PC%d'%(each_component) for each_component in np.arange(1, n_components + 1 )]
    
    ## CREATING PRINCIPAL DATAFRAME
    principalDf = pd.DataFrame(data = principalComponents,
                               columns = pca_columns )
    
    ## GETTING FINAL DATAFRAME
    finalDf = pd.concat([principalDf, descriptors_df_reorg[['label']]], axis = 1)
    
    return finalDf, pca, pca_columns

### FUNCTION ADD BIAS
def add_ones_to_last_col(array):
    '''
    The purpose of this function is to add an array of ones to last column.
    INPUTS:
        array: [np.array]
            array you would like to add columns of ones to
    OUTPUTS:
        array_with_bias: [np.array]
            updated arrays with a column of 1's        
    '''
    ## APPENDING NEW ARRAY WITH BIAS TERM
    array_with_bias =  np.append( array, np.ones(len(array))[:, np.newaxis], axis=1 )
    return array_with_bias
    
## COMPUTING BEST FIT SLOPE
def best_fit_slope_and_intercept(xs,ys):
    # Reference: https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/?completed=/how-to-program-best-fit-line-slope-machine-learning-tutorial/
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    b = np.mean(ys) - m*np.mean(xs)
    
    return m, b

## COMPUTING ROOT MEAN SQUARED ERROR
def compute_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

## FUNCTION TO CALCUALTE MSE, R2, EVS, MAE
def metrics(y_fit,y_act, want_dict = False):
    '''
    The purpose of this function is to compute metrics given predicted and actual
    values.
    INPUTS:
        y_fit: [np.array]
            fitted values as an array
        y_act: [np.array]
            actual values
        want_dict: [logical, default=False]
            True if you want the output to be a dictionary instead of a tuple
    OUTPUTS:
        mae: [float]
            mean averaged error
        rmse: [float]
            root mean squared errors
        evs: [float]
            explained variance score
        r2: [float]
            R squared for linear fit
        slope: [float]
            best-fit slope between predicted and actual values
        pearson_r_value: [float]
            pearsons r
        
    '''
    ## EXPLAINED VARIANCE SCORE
    evs = explained_variance_score(y_act, y_fit)
    ## MEAN AVERAGE ERROR
    mae = mean_absolute_error(y_act, y_fit)
    ## ROOT MEAN SQUARED ERROR
    rmse = compute_rmse(predictions=y_fit, targets = y_act) # mean_squared_error(y_act, y_fit)
    ## SLOPE AND INTERCEPT
    slope, b = best_fit_slope_and_intercept( xs = y_act, ys = y_fit)
    ## R-SQUARED VALUE
    r2 = r2_score(y_act, y_fit)
    ## PEARSON R
    pearson_r = pearsonr( x = y_act, y = y_fit )[0]
    if want_dict is False:
        return mae, rmse, evs, r2, slope, pearson_r
    else:
        ## CREATING DICTIONARY
        output_dict = {
                'evs': evs,
                'mae': mae,
                'rmse': rmse,
                'slope': slope,
                'r2': r2,
                'pearson_r': pearson_r
                }
        ## OUTPUTTING
        return output_dict

    def compute_regression_accuracy( y_pred, y_true ):
        '''
        The purpose of this code is to compute the regression accuracy. 
        INPUTS:
            y_pred: [np.array]
                predictions of y values
            y_true: [np.array]
                true y-values
        OUTPUTS:
            accuracy_dict: [dict]
                accuracy in the form of a dictionary
        '''
        ## COMPUTING ACCURACY (MEAN AVERAGE ERROR, ROOT MEAN SQUARED ERROR, etc.)
        mae, rmse, evs, r2, slope = metrics(y_fit = y_pred, 
                                    y_act = y_true)
        ## CREATING DICTIONARY
        accuracy_dict = {
                            'mae': mae,
                            'rmse': rmse,
                            'evs': evs,
                            'r2': r2,
                            'slope': slope
                            }
        
        return accuracy_dict

### PLOTTING SCATTER PLOT
def plot_scatter_parity(exp_values,
                        pred_values,
                        title = None,
                        labels = None,
                        xlabel = None,
                        ylabel = None,
                        scatter_dict = { 'marker': '.',
                                         'color': 'k',
                                         's': 200},
                        fig = None,
                        ax = None,
                        fig_size_cm=(8.3, 8.3),
                        fig_limits = None,
                        want_statistical_text_box = True,
                        ):
    '''
    The purpose of this function is to plot the scatter plot of a parity 
    given experiments and predicted values.
    INPUTS:
        exp_values: [np.array]
            experimental values
        pred_values: [np.array]
            predicted values
        labels: [list]
            list of labels for each point
        xlabel: [str]
            labels for the x-axis
        ylabel: [str]
            labels for the y-axis
        fig: [obj]
            input figure. If not None, this function will create a figure.
        ax: [obj]
            axis for figure. 
        fig_size_cm: [tuple, 2]
            figure size in cm
        fig_limits: [dict]
            dicitonary of fig limits, e.g. :
                {'x': [-0.5, 7.5],
                 'y': [-0.5, 7.5]}
            This means x ranges from -0.5, 7.5
        want_statistical_text_box: [logical]
            True if you want statistical box
    OUTPUTS:
        fig, ax: [obj]
            figure and axis for the plot
    '''
    ## CREATING IMAGES
    if fig is None or ax is None:
        
        ## CONVERTING TO INCHES
        figsize=cm2inch( *fig_size_cm )
        ## CREATING FIGURE
        
        ## PLOTTING
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1) 

        ## SETTING LABELS
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    ## PLOTTING
    ax.scatter(exp_values, pred_values, **scatter_dict)
    
    ## ADDING LABELS
    if labels is not None:
        ## LOOPING THROUGH EACH VALUE AND PRINTING LABEL
        for idx, each_label in enumerate(labels):
            x = exp_values[idx]
            y = pred_values[idx]
            ax.annotate(each_label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
            
    ## CHECKING LIMITS
    ax = check_ax_limits(ax, fig_limits)

    ## DRAWING Y = X LINE
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, '--', color = 'k', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ## SETTING X AND Y LIMITS
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ## ADDING TITLE
    if title is not None:
        ax.set_title(title, fontsize=10)

    ## ADDING STATISTICAL BOX
    if want_statistical_text_box == True:
        ## COMPUTING ACCURACY DICT
        accuracy_dict = metrics(y_fit = pred_values,
                                y_act = exp_values, 
                                want_dict = True)
        
        ## CREATING BOX TEXT
        box_text = "%s: %.2f\n%s: %.2f"%( "Slope", accuracy_dict['slope'],
                                          "RMSE", accuracy_dict['rmse']) 
        ## ADDING TEXT BOX
        text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=1)
        ax.add_artist(text_box)
        
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO GET LINEAR REGRESSION
def generate_linear_regression_with_bias(X_array, y_array):
    '''
    The purpose of this function is to generate a linear regression with bias. 
    We place a bias of 1 to normalize all the data.
    REF: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    INPUTS:
        X_array: [np.array]
            data values with N x M shape
        y_array: [np.array]
            data that you want to fit to. This should have N shape
    OUTPUTS:
        regr: [obj]
            LinearRegression object
        y_pred: [np.array]
            predicted y values from linear regression
    '''
    ## ADDING BIAS (column of 1's)
    X_array_with_bias = add_ones_to_last_col(X_array)
    
    ## Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept = False) # Since we are adding bias
    
    ## FITTING
    regr.fit(X_array_with_bias, y_array)
    
    ## PREDICTING WITH TRAINING SET
    y_pred = regr.predict(X_array_with_bias)
    return regr, y_pred

### FUNCTION TO PRINT COEFFICIENTS AND SO ON
def print_linear_regression_accuracy(regr, y_array, y_pred):
    ''' 
    This function prints linear regression accuracy.
    INPUTS:
        regr: [obj]
            LinearRegression object
        y_array: [np.array]
            data that you want to fit to
        y_pred: [np.array]
            predicted y values from linear regression
    OUTPUTS:
        no output
    '''

    ## COEFFICIENTS
    print('Coefficients: \n', regr.coef_)
    ## MEAN SQUARED ERROR
    print("Mean squared error: %.2f"
          % mean_squared_error(y_array, y_pred))
    ## EXPLAINED VARIANCE SCORE: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_array, y_pred))
    
    return

### FUNCTION TO PLOT PRINCIPAL COMPONENT VALUES
def plot_importance_pc_values( features_list,
                               abs_eigenvalue,
                               pc_name,
                               variance_value = None,
                               fig_size_cm=(8.3, 8.3),
                               y_limits = None,
                               title = None):
    '''
    The purpose of this function is plot the absolute eigen values versus 
    the molecules descriptors.
    INPUTS:
        features_list: [np.arrary]
            features list as x-axis
        abs_eigenvalue: [np.array]
            eigen values, used as y-axis
        pc_name: [str]
            name of the principal component, used as title
        variance_value: [float]
            variance value coresponding to the PC. If None, nothing happens. 
            Otherwise, insert variance value as a percentage within the 
            title.
        fig_size_cm: [tuple, 2]
            figure size in cm
        y_limits: [tuple]
            tuple y limits
        title: [str]
            title name
    OUTPUTS:
        fig, ax: figure and axis
    '''
    ## CONVERTING TO INCHES
    figsize=cm2inch( *fig_size_cm )
    
    ## CREATING FIGURE
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1) 
    ## PLOTTING
    ax.bar(features_list, abs_eigenvalue, align='center', color='k', alpha=1)
    
    ## ROTATING X AXIS
    plt.setp(ax.get_xticklabels(), 
             rotation=45, 
             ha="right",
             rotation_mode="anchor") #  fontsize = AXIS_FONTSIZE

    ## SETTING X AXIS LABEL
    ax.set_ylabel('Absolute eigenvalues') # , fontsize = 10
    
    ## SETTING THE LIMITS
    if y_limits is not None:
        ## SETTING Y LIMS
        ax.set_ylim((y_limits[0], y_limits[1]))
        ## SETTING TICKS
        ax.set_yticks( np.arange(y_limits[0], y_limits[1] + y_limits[2], y_limits[2]) )
    
    ## SETTING TITLE
    if variance_value is not None:
        title = pc_name + " (" + "%.1f"%(variance_value*100) + "%)"
    else:
        title = pc_name
    ## UPDATING TITLE
    ax.set_title(title)
    
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig,ax

### FUNCTION TO GET MOST IMPORTANT FEATURES
def pca_get_most_important_features(pca, descriptor_keys):
    '''
    The purpose of this function is to get the most important features
    INPUTS:
        pca: [obj]
            principal component object
        descriptor_keys: [list]
            list of descriptor keys
    OUTPUTS:
        importance_features_dict: [dict]
            dictonary of the most important features that is aranged in 
            descending order.
    '''
    ## DEFINING KEYS
    feature_names = np.array(descriptor_keys[:])
    
    ## DEFINING LIST TO STORE DATAFRAMES
    importance_features_dict = {}
    
    ## LOOPING THROUGH EACH OF THE COMPONENTS
    for pc_idx, pc_importance in enumerate(pca.components_):
        ## ABSOLUTE VALUES
        abs_pc_importance = np.abs(pc_importance)
        ## ORDERING
        order_index = np.argsort( abs_pc_importance )[::-1] # descending
        ## GETTING DICT
        importance_dict = {
                'features': feature_names[order_index],
                'abs. eigenvalues': abs_pc_importance[order_index],
                }
        
        ## GETTING DICTIONARY
        # df_imp = pd.DataFrame(importance_dict)
        
        ## STORING
        importance_features_dict['PC%d'%(pc_idx+1)] = importance_dict
        
    return importance_features_dict

### FUNCTION TO PLOT PRINCIPAL COMPONENT ANALYSIS
def plot_pca(df,
             fig_size_cm=(8.3, 8.3),
             axis_limits = None,
             want_grid = True,
             ):
    '''
    The purpose of this function is to plot the principal component analysis 
    given a dataframe
    INPUTS:
        df: [pd.dataframe]
            dataframe  containing the principal components, e.g.
                    principal component 1  principal component 2     label
                0                1.622689              -0.508636        C1
                1                1.292417               0.260971        C2
                2                0.187438              -0.686356        C4
                3               -0.495350               0.469113        C6
                4               -0.788387               1.755708        C8
                5               -1.434050               2.794774       C10
                6               -0.880937              -1.342210       CYC
        targets: [list]
            list of desired PCA labels you want to print
        fig_size_cm: [tuple]
            figure size in cm
        axis_limits: [tuple, shape = 3]
            axis limits in start, end, increment. If None, we will plot without 
            setting the limits
    OUTPUTS:
        fig, ax: [obj]
            figure and axis objects
    '''
    ## IMPORTING COLOR MATPLOTLIB
    import matplotlib.cm as cm
    ## CONVERTING TO INCHES
    figsize=cm2inch( *fig_size_cm )
    ## CREATING FIGURE
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1) 
    
    ## ADDING LABELS
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    
    ## DEFINING COLORS
    colors=cm.rainbow(np.linspace(0,1,len(targets)))
    
    ## LOOPING THROUGH TARGETS AND COLORS
    for target, color in zip(targets,colors):
        indicesToKeep = df['label'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1'], 
                   df.loc[indicesToKeep, 'PC2'],
                   c = color,
                   s = 50,)
        
    ## ADDING TEXT
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['label'] == target
        x = float(df.loc[indicesToKeep, 'PC1'])
        y = float(df.loc[indicesToKeep, 'PC2'])
        
        ## ADDING TEXT
        ax.annotate(target, 
                   (x,y),
                   horizontalalignment='center',
                   textcoords="offset points",
                   xytext=(0,5),
                   transform=ax.transAxes,
                   fontsize = 8)
    
    ## SETTING THE LIMITS
    if axis_limits is not None:
        ## SETTING X AND Y LIMITS
        ax.set_xlim((axis_limits[0], axis_limits[1]))
        ax.set_ylim((axis_limits[0], axis_limits[1]))
        
        ## SETTING TICKS
        ax.set_xticks( np.arange(*axis_limits) )
        ax.set_yticks( np.arange(*axis_limits) )
    
    # ax.legend(targets,  fontsize=8)
    # Put a legend to the right of the current axis
    # ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    if want_grid is True:
        ax.grid(linewidth=0.5)
    # tightlayout
    fig.tight_layout()
    return fig, ax

### FUNCTION TO PLOT BAR PLOT
def plot_bar_predicted_results(x_labels,
                               y_pred,
                               x_train_labels = None,
                               fig_size_cm = (8.3, 8.3),
                               ylabel = None,
                               want_sorted = False,
                               ):
    '''
    The purpose of this function is to plot the predicted results.
    Note that this plots a y = 0 line so it is not strange to have a cutoff
    INPUTS:
        x_labels: [list]
            list of x labels
        y_pred: [np.array]
            predicted y values
        x_train_labels: [list]
            list of x labels that you trained on. The algorithm will 
            look over and see which labels are trained, then change the 
            color to red. 
        fig_size_cm: [tuple]
            figure size in cm
        ylabel: [str]
            y label
        want_sorted: [logical]
            True if you want the bar plot sorted according to largest to smallest 
            value. 
    OUTPUTS:
        fig, ax: [obj]
            figure and axis for the plot
    '''
    ## SORTING
    if want_sorted is True:
        ## GETTING ARGUMENTS
        index = np.argsort(y_pred)
        
        ## SORING LABELS AND PREDICTIONS
        x_labels = np.array(x_labels)[index]
        y_pred = np.array(y_pred)[index]
    
    ## CREATING FIGURE
    ## CONVERTING TO INCHES
    figsize=cm2inch( *fig_size_cm )
    ## PLOTTING
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1) 
    
    ## PLOTTING
    bar_plot = ax.bar(x_labels, 
                       y_pred, 
                       align='center', 
                       color='k',
                       alpha=1)
    
    ## CHANGING COLOR
    if x_train_labels is not None:
        ## GETTING LABELS
        index_to_change = [ idx for idx, each_x in enumerate(x_labels) if each_x in x_train_labels]
        for each_index in index_to_change:
            bar_plot[each_index].set_color('r')

    ## SETTING AXIS
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    ## ROTATING X AXIS
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
    
    ## DRAWING LINE AT Y = 0
    ax.axhline(y=0, linestyle='-', color='k', linewidth = 1)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## DRAWING X AXIS
    ax.xaxis.grid(True)
    
    return fig, ax

### FUNCTION TO GET EQUATION
def find_equation_linear_regression(regr,
                                    pca_columns,
                                    fmt="%.2f",
                                    output_string="Output"
                                    ):
    '''
    The purpose of this function is to find the linear regression equation
    INPUTS:
        regr: [obj]
            regression object
        pca_columns: [list]
            list of pca columns
        fmt: [string]
            format of coefficients
    OUTPUTS:
        full_str: [str]
            string of whole equation
    '''
    
    ## GETTING COEFFICIENTS
    coefficients = [ fmt %(each_coeff) for each_coeff in regr.coef_]
    
    ## DEFINING OUTPUTS
    full_str = output_string + ' = '
    
    ## ADDING FOR EACH COEFFICIENT
    for idx, each_coeff in enumerate(coefficients):
        full_str+= "(" + each_coeff + ")"
        if idx != len(coefficients)-1:
            full_str += "%s"%(pca_columns[idx])
            full_str += " + "
    
    return full_str

### FUNCTION TO REORDER COLUMSN
def reorder_descriptor_cols(df,
                            descriptor_order,
                            columns_not_to_normalize = [],):
    '''
    The purpose of this script is to organize the descriptors so they are not 
    always randomly dispersed. 
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe
        descriptor_order: [list]
            list of the desired escriptor order
        columns_not_to_normalize: [list]
            list of columns not to normalize
    OUTPUTS:
        df_reorg: [pd.dataframe]
            reorganized dataframe

    '''
    ## GETTING CURRENT ORDER
    current_order = df.columns.tolist()
    
    ## GETTING DESCRIPTOR LIST
    descriptor_list = np.array([each_col for each_col in current_order if each_col not in columns_not_to_normalize ])
    
    ## GETTING NEW LIST
    new_descriptor_list = np.array([])
    
    ## LOOPING THROUGH ORDER LIST
    for each_descriptor in descriptor_order:
        ## FINDING INDEXES
        index =[ idx for idx, each_col in enumerate(descriptor_list) if each_descriptor in each_col ] 
        ## APPENDING
        new_descriptor_list = np.append(new_descriptor_list, descriptor_list[index])
    
    ## ADDING ALL OTHER DESCRIPTORS THAT ARE NOT LISTED
    new_descriptor_list = np.append(new_descriptor_list , descriptor_list[~np.isin(descriptor_list, new_descriptor_list)])

    ## APPENDING NOT NORMALIZED
    new_descriptor_list = np.append(columns_not_to_normalize, new_descriptor_list)
    
    ## GETTING NEW ORDER
    df_reorg = df[new_descriptor_list]
    
    return df_reorg

### FUNCTION TO GET R^2 MATRIX
def compute_r_sq_matrix(descriptors_array):
    '''
    The purpose of this function is to compute the R^2 matrix. This is 
    a good way to see if any descriptors are correlated. R^2 values 
    greater than 0.99 may indicate that the descriptor is redundant.
    NOTE:
        This matrix should be symmetric! We compute for all possible 
        combinations, which may be slightly inefficient. R^2 values 
        are computed fairly quickly, so we are not too worried about it. 
    INPUTS:
        descriptors_array: [np.array, shape=(num_data_pts, num_descriptors)]
            normalized descriptor array between 0 and 1. 
    OUTPUTS:
        r2_matrix: [np.array, shape = (num_descriptors, num_descriptors)]
            R^2 matrix containing the correlation coefficient. These values 
            range from 0 and 1. The diagonal of this matrix is always one. 
            e.g.
            array([[1.        , 0.74517887, 0.10295332, 0.69814782, 0.04745124,
                    0.35423126, 0.14450266, 0.01545312, 0.01013645, 0.44356253],
    '''
    ## GETTING TOTAL NUMBER OF DESCRIPTORS
    num_descriptors = descriptors_array.shape[1]
    
    ## DESIGNING NUMPY ARRAY TO CAPTURE MAP
    r2_matrix = np.zeros( (num_descriptors, num_descriptors) )        
    
    ## LOOPING THROUGH I AND J OF DESCRIPTOR SPACE
    for i in range(num_descriptors):
        descriptor_i = descriptors_array[:,i]
        for j in range(num_descriptors): # range(i,num_descriptors)
            ## GETTING DESCRIPTOR VALUES
            descriptor_j = descriptors_array[:,j]
            ## GETTING SLOPE AND SO ON
            slope, intercept, r_value, p_value, std_err = stats.linregress(descriptor_i,descriptor_j)
            ## GETTING R SQUARED
            r2 = r_value**2
            # r2_score(descriptor_i, descriptor_j)
            ## STORING
            r2_matrix[i,j] = r2
    return r2_matrix

### FUNCTION TO GENERATE PREDICTIONS BASED ON EXP DATA
def full_compute_linear_regression(path_csv,
                                   finalDf,
                                   column_name = None
                                   ):
    '''
    The purpose of this function is to compute the multilinear regression 
    given a csv file. 
    INPUTS:
        path_csv: [str]
            path to csv file
        finalDf: [df]
            PCA values, e.g. 
                         PC1       PC2     label
                0   1.622689 -0.508636        C1
                1   1.292417  0.260971        C2
                2   0.187438 -0.686356        C4
        column_name: [str]
            column name that you want the label from
    OUTPUTS:
        y_array: [np.aray]
            y array (exp values)
        y_pred: [np.array]
            predicted y values
        X_df: [df]
            Dataframe for X
        regr: [obj]
            regression object
    '''
    ## LOADING
    exp_data = pd.read_csv(path_csv)
    
    ## DEFINING X VALUES
    X_df = finalDf.loc[ finalDf['label'].isin(exp_data['label'].to_numpy())]
    
    ## GETTING ARRAY
    X_array = X_df[pca_columns].to_numpy()
    
    ## FINDING STRING THAT IS NOT LABEL
    if column_name is None:
        str_not_label = exp_data.columns[exp_data.columns!='label'][0]
    else:
        str_not_label = column_name
    
    ## DEFINING Y ARRAY
    y_array = np.array([exp_data[exp_data['label'] == each_x][str_not_label].tolist()[0] for each_x in  X_df.label.to_list()])
    
    ## COMPUTING LINEAR REGRESSION
    regr, y_pred = generate_linear_regression_with_bias(X_array = X_array, 
                                                        y_array = y_array)

    ## PRINTING RESULTS
    print_linear_regression_accuracy(regr = regr, 
                                     y_array = y_array, 
                                     y_pred = y_pred)
    
    
    return y_array, y_pred, X_df, regr

### FUNCTION TO MAKE PREDICTION
def pred_from_regression(finalDf,
                         regr,
                         ):
    '''
    The purpose of this function is to make a prediction from linear regression. 
    Note, we are assuming you have some bias of 1's
    INPUTS:
        finalDf: [df]
            dataframe object with pca details
        regr: [obj]
            regression object
    OUTPUTS:
        y_pred_test: [np.array]
            predicted values for all dataframe objects
    '''
    ## GETTING ARRAY
    X_array = finalDf[pca_columns].to_numpy()
    
    ## ADDING BIAS (column of 1's)
    X_array_with_bias = add_ones_to_last_col(X_array)
    
    ## PREDICTING
    y_pred_test = regr.predict(X_array_with_bias)
    return y_pred_test  

### FUNCTION TO PRINT OUT INFORMATION
def print_csv_pred_model(y_exp,
                         y_pred,
                         labels,
                         equation_string,
                         path_csv_dir,
                         csv_name
        ):
    '''
    This function prints out csv information for model
    INPUTS:
        y_exp: [list]
            list of experimental data
        y_pred: [list]
            list of predicted data
        labels: [list]
            list of the labels
        equation_string: [str]
            equation stirng
        path_csv_dir: [str]
            path to csv file
        csv_name: [str]
            path to cs file name
    '''

    print("Writing equation and data to: %s"%(path_csv_dir))
    ## EXPORTING PREDECTED VALUES
    df_model = pd.DataFrame(data = np.array([y_exp, y_pred, labels ]).T,
                            columns = ['y_exp', 'y_pred', 'label'])
    ## CSV NAME
    df_model.to_csv( os.path.join(path_csv_dir,
                                  csv_name + "_pred_model.csv"))
    
    ## PATH TO OUTPUT EQ
    path_output_eq = os.path.join(path_csv_dir,
                                  csv_name + "_pred_equation.csv")
    
    print("COMPUTING ACCURACY DICT")
    ## COMPUTING ACCURACY DICT
    accuracy_dict = metrics(y_fit = np.array(df_model['y_pred']).astype(float),
                            y_act = np.array(df_model['y_exp']).astype(float), 
                            want_dict = True)
    
    ## GETTING LINEAR EQUATION
    with open(path_output_eq, 'w') as f:
        f.write("%s\n"%(equation_string))
        f.write("Slope, %s\n"%(accuracy_dict['slope']))
        f.write("RMSE, %s\n"%(accuracy_dict['rmse']))
        
    return

### FUNCTION TO RUN FULL PREDICTION WITH STORAGE
def full_prediction_given_dict(csv_name,
                               output_string,
                               path_csv = PATH_CSV,
                               fig_limits = None,
                               xlabel = None,
                               ylabel = None,
                               want_title = True,
                               fig_size_cm = (8.3, 8.3),
                               save_fig = False,
                               output_image_path = OUTPUT_IMAGE_PATH,
                               fig_extension = FIG_EXTENSION,
                               output_csv_dir = None,
                               ):
    '''
    This function makes a full prediction given a dictionary. It will load the 
    data, then run the linear regresion analysis.
    INPUTS:
        csv_name: [str]
            name of the csv
        output_string: [str]
            name of the output for the equation
        path_csv: [str]
            path to csv file
        fig_limits: [dict]
            dictionary of figure limits
        xlabel: [str]
            label for x axis
        ylabel: [str]
            label for y axis
        want_title: [logical]
            True if you want the title
        save_fig: [logical]
            True if saving figure
        output_csv_dir: [str]
            output csv directory. If None, then we will not output any csv files
    OUTPUTS:
        void -- figure and axis are outputted
    '''
    ## GETTING PATH
    full_path_csv = os.path.join(PATH_CSV, csv_name + '.csv' ) # csv_dict[csv_name]
    
    ## GETTING LINEAR REGRESSION
    y_array, y_pred, X_df, regr= full_compute_linear_regression(path_csv = full_path_csv,
                                                                finalDf = finalDf,
                                                                column_name = output_string,
                                                                )
    
    ## GETTING THE EQUATION FOR LINEAR REGRESSION
    if want_title is True:
        full_str = find_equation_linear_regression(regr,
                                                   pca_columns,
                                                   fmt="%.2f",
                                                   output_string=output_string
                                                   )
    else:
        full_str = None
    
    ## DEFINING LABELS
    labels = X_df.label.to_list()
    
    ## PLOTTING PARITY PLOT
    fig, ax = plot_scatter_parity(exp_values = y_array,
                                  pred_values = y_pred,
                                  title = full_str,
                                  labels = labels,
                                  xlabel = xlabel,
                                  ylabel = ylabel,
                                  scatter_dict = { 'marker': '.',
                                                   'color': 'k',
                                                   's': 200},
                                  fig_size_cm = fig_size_cm,
                                  fig_limits = fig_limits)
    ## PRINTING CSV
    if output_csv_dir is not None:
        
        ## PRINTING CSV
        print_csv_pred_model(y_exp = y_array,
                             y_pred = y_pred,
                             labels = labels,
                             equation_string = full_str,
                             path_csv_dir = output_csv_dir,
                             csv_name = csv_name,
                             )
            
    ## STORING FIGURE
    print_tools.store_figure( fig = fig, 
                              fig_extension = fig_extension,
                              path = os.path.join(output_image_path, csv_name + '-' + output_string + '-model_vs_exp_parity' ),
                              bbox_inches = 'tight',
                              save_fig = save_fig,
                              )
    
    ## MAKING PREDICTIONS
    y_pred_test = pred_from_regression(finalDf = finalDf,
                                       regr = regr,
                                       )

    ## PLOTTING PREDICTION
    fig_bar, ax_bar = plot_bar_predicted_results(x_labels = finalDf.label.to_list(),
                                         x_train_labels = X_df.label.to_list(),
                                         y_pred = y_pred_test,
                                         fig_size_cm = fig_size_cm,
                                         ylabel = ylabel,
                                         want_sorted = True,
                                         )

    ## STORING FIGURE
    print_tools.store_figure( fig = fig_bar, 
                              fig_extension = fig_extension,
                              path = os.path.join(output_image_path, csv_name + '-' + output_string + '-model_predicted_for_each_lig' ),
                              bbox_inches = 'tight',
                              save_fig = save_fig,
                              )
    return fig, ax

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING DESCRIPTORS OF INTEREST
    descriptors_dict = {'Lig.-water Hbond': analyze_gmx_hbond, 
                        'Eccentricity': analyze_gmx_principal, 
                        'RMSF': analyze_gmx_rmsf,
                        'RadiusGyrate': analyze_gmx_gyrate,
                        'SASA': nanoparticle_sasa,}
    
    ## DEFINING DESIRED ORDER
    descriptor_order = [
            'RadiusGyrate',
            'RMSF',
            'Eccentricity',
            'SASA',
            'Lig.-water Hbond',
            'log P'
            ]
    
    ## DEFINING PICKLE DIRECTORY TO LOOK INTO
    # pickle_path = r"/mnt/r/scratch/nanoparticle_project/scripts/analysis_scripts/PICKLE_ROT"
    pickle_path = check_tools.check_path( r"R:\scratch\nanoparticle_project\scripts\analysis_scripts\PICKLE_ROT" )    
    # r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20191202\Images\extract_PCA_with_new_ROT_particles"
    

    
    ## DEFINING SAVE IMAGE
    save_fig = False
    
    #%%
#    r''' PRINTING SASA OVER TIME
    ## DEFINING PICKLE NAME    
#    pickle_name = 'EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1'
#    pickle_name = 'switch_solvents-50000-dmso-EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1'
    
    pickle_name = 'EAM_300.00_K_2_nmDIAM_ROT006_CHARMM36jul2017_Trial_1'
    pickle_name = 'switch_solvents-50000-dmso-EAM_300.00_K_2_nmDIAM_ROT006_CHARMM36jul2017_Trial_1'    
    pickle_list = ['EAM_300.00_K_2_nmDIAM_ROT006_CHARMM36jul2017_Trial_1',
                   'switch_solvents-50000-dmso-EAM_300.00_K_2_nmDIAM_ROT006_CHARMM36jul2017_Trial_1'
                   ]
    for pickle_name in pickle_list:
        ## LOADING THE PICKLE
        results = pickle_tools.load_pickle_from_descriptor(pickle_dir = pickle_path,
                                                           descriptor_class = nanoparticle_sasa,
                                                           pickle_name = pickle_name,
                                                           verbose = False
                                                           )
        ## GETTING SASA        
        sasa_values = results.sasa['all_atom_ligands']
        
        print(pickle_name, np.mean(sasa_values))
        
    #%%

    ## SETTING FIGURE LIMITS
    fig_limits =  {
             # 'x': [-0.5, 7.5, 1],
#             'y': [190, 260, 20]
             }
    
    ## PLOTTING SASA OVER TIME
    fig, ax = plot_sasa_over_time(results,
                                  fig_size_cm = (8.3, 8.3),
                                  fig_limits = fig_limits)
    
    ## PRINTING FIGURE
    print_tools.store_figure( fig = fig,
                              path = os.path.join(OUTPUT_IMAGE_PATH, pickle_name + '_sasa'),
                              fig_extension = 'png',
                              save_fig = True,
                              )
    

#    '''
    

    #%%
       
    ## DEFINING CLASS EXTRACTION
    extraction = class_extraction(descriptor_extraction_dict = DESCRIPTOR_EXTRACTION_DICT)
    
    ## DEFINING DECODING NAME
    decode_type = 'nanoparticle'
    
    ## DEFINING VERBOSITY
    verbose = True
    
    ## CHECKING FUNCTIONS TO PATH
    pickle_path = check_tools.check_path(pickle_path)

    ## LOOPING THROUGH ANALYSIS TYPES
    analysis_type_list =  ["All"] # ['For_Log_P', 'No_DMSO_data',
    # 'All', 
    # 'All', 'For_Log_P', 'No_DMSO_data', 'Subset'
    # , 'For_Log_P', 'No_DMSO_data', 'Subset'
    # 'All', 'For_Log_P', 'No_DMSO_data', 'Subset'
    '''
     'For_Log_P' -- log P analysis
     'All' -- all data 
     'No_DMSO_data' -- No DMSO data is considered
     'Subset' -- subset based on Christian's ligands
     'moyano_2014_fig_1'
    '''
    
    ## LOOPING THROUGH ANALYSIS TYPES
    for analysis_type in analysis_type_list:
        ## CLOSING ALL FIGURES
        plt.close('all')

        ## DEFINING INTERESTED DICT
        if analysis_type == 'Subset':
            interest_dict = {
                'ligand': ["ROT001", "ROT005", "ROT006", "ROT010", "ROT011", "ROT012", "ROT014",
                           ]
                }
            ## DEFINING SORTING LIST
            sorting_list = ['C1', 'C2', 'C4', 'C6', 'C8', 'CYC', 'C10',
                            ]
        
        ## ALL TYPE ANALYSIS
        elif analysis_type == 'All' or analysis_type == 'No_DMSO_data':
            interest_dict = {
                    'ligand': ["ROT001", "ROT005", "ROT006", "ROT010", "ROT011", "ROT012", "ROT014",
                               "ROT002", "ROT003", "ROT004", "ROT007", "ROT008", "ROT009",
                               "ROT015", "ROT016",]
                    }
            ## DEFINING SORTING LIST
#            sorting_list = ['C1', 'C2', 'C4', 'C6', 'C8', 'CYC','C10',
#                            'C3OH', 'C3NH2', 'BENZ', 'CYC_BENZ', 'MET_BENZ', 'BRANCH_C4', 'CYC_ISO', 'OH']
            
            sorting_list = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'NP6', 'NP7', 'NP8'] # , 'NP9'
            
            ## FOR NO DMSO
            if analysis_type == 'No_DMSO_data':
                interest_dict['cosolvent'] = ['None']
        ## LOG P
        elif analysis_type == 'For_Log_P':
            interest_dict = {
                    'ligand': ["ROT001", "ROT005", "ROT006", "ROT010", "ROT011", "ROT012", 
                               "ROT002", "ROT003", "ROT004", "ROT007", "ROT008", "ROT015"]
                    } # , "ROT009" "ROT014",
            ## DEFINING SORTING LIST
            sorting_list = ['C1', 'C2', 'C4', 'C6', 'CYC','C10',
                            'C3OH', 'C3NH2', 'BENZ', 'CYC_BENZ', 'MET_BENZ', 'CYC_ISO',] # 'C8'  'OH'
        
        ## GETTING NEW OUTPUT IMAGE PATH
        output_image_path = os.path.join(OUTPUT_IMAGE_PATH, analysis_type)
        ## MAKING DIRECTORY
        if os.path.isdir(output_image_path) is not True:
            os.mkdir(output_image_path)
        
        ## LOADING ALL DESCRIPTORS
        descriptors_df = load_descriptors(descriptors_dict = descriptors_dict,
                                          pickle_path = pickle_path,
                                          interest_dict = interest_dict,
                                          decode_type = decode_type,
                                          )
         
        
        #%%
        
        csv_dict = {'membrane_binding'  : 'membrane_binding.csv',
                    'log P'             : 'log_P_values.csv',
                    'moyano_2014_fig_1' : 'moyano_2014_fig_1.csv',
                    'moyano_2014_fig_2' : 'moyano_2014_fig_2.csv',
                    }
        
        #%%
        
        ## REARRANGING COLUMNS SO DESCRIPTORS ARE AT THE END
        descriptors_df = dataframe_move_cols_to_end(df = descriptors_df,
                                                    cols_at_end = list(descriptors_dict.keys()))
        
        ## GETTING DESCRIPTORS
        # descriptor_array = np.array(descriptors_df[list(descriptors_dict.keys())])
        
        #%%
        #################################
        ### MATCHING ALL SWITCH NAMES ###
        #################################
        
    ## DEFINING KEY LABEL
#    ligand_name_labels = {
#            'ROT001' : 'C1',
#            'ROT010' : 'C2', 
#            'ROT011' : 'C4',
#            'ROT005' : 'C6',
#            'ROT014' : 'C8',
#            'ROT012' : 'C10',
#            'ROT006' : 'CYC',
#            'ROT002' : 'C3OH',
#            'ROT003' : 'C3NH2',
#            'ROT004' : 'BENZ',
#            'ROT007' : 'CYC_BENZ',
#            'ROT008' : 'CYC_ISO',
#            'ROT009' : 'OH',
#            'ROT015' : 'MET_BENZ',
#            'ROT016' : 'BRANCH_C4',
#            }
    
        ligand_name_labels = {
                'ROT001' : 'NP1',
                'ROT002' : 'NP2',
                'ROT003' : 'NP3',
                'ROT004' : 'NP4',
                'ROT005' : 'NP5',
                'ROT006' : 'NP6',
                'ROT007' : 'NP7',
                'ROT008' : 'NP8',
                'ROT009' : 'NP9',
                
                'ROT010' : 'C2', 
                'ROT011' : 'C4',
                'ROT014' : 'C8',
                'ROT012' : 'C10',
                'ROT015' : 'MET_BENZ',
                'ROT016' : 'BRANCH_C4',
                }
            
        
        ## MATCHING BASED ON NAME
        match_prefix = 'switch'
        
        ## DEFINING MATCH NAME
        match_column_name = 'cosolvent'
        
        ## DEFINING DESIRED COLUMNS
        column_matching = [ 'diameter', 'ligand', 'shape', 'temperature', 'trial']
        
        ## MATCHING DATAFRAME
        descriptors_df_reorg = reorganize_rows_by_column_matching(   df = descriptors_df,
                                                                     match_prefix = match_prefix,
                                                                     match_column_name = match_column_name,
                                                                     column_matching = column_matching)
        
        ## DROPPING MISSING COLUMNS
        descriptors_df_reorg = pd_drop_na_columns(descriptors_df_reorg)
        
        #####################
        ### ADDING LABELS ###
        #####################
        ## ADDING LABELS TO DFS
        descriptors_df_reorg = add_label_to_dict(df = descriptors_df_reorg,
                                               name_labels = ligand_name_labels,
                                               column_name = 'ligand',
                                               label_name = 'label')
    
        ## REORDERING DATAFRAME BASED ON A COLUMN
        descriptors_df_reorg = order_df(df = descriptors_df_reorg,
                                       ordered_classes = sorting_list,
                                       col_name = 'label')
    
        ## ADDING LOG P DESCRIPTORS
        if analysis_type == 'For_Log_P':
            ## GETTING PATH
            path_csv = os.path.join(PATH_CSV, csv_dict['log P'])
            log_p_results = pd.read_csv(path_csv)
            log_p_storage = []
            ## REORDER LOG P
            for each_label in descriptors_df_reorg.label.tolist():
                ## LOCATING LOG P
                log_p_storage.append(float(log_p_results[log_p_results['label']==each_label]['log P']))
            
            ## ADDING TO COLUMN
            descriptors_df_reorg['log P'] = log_p_storage
    
        
        #%%
        
        ### DEFINING COLUMNS NOT TO NORMALIZE
        columns_not_to_normalize = ['identifier', 
                                    'diameter', 
                                    'ligand', 
                                    'shape', 
                                    'temperature', 
                                    'trial', 
                                    'cosolvent',
                                    'label']
        
        ## REORGANIZING THE DATA
        descriptors_df_reorg = reorder_descriptor_cols(df = descriptors_df_reorg,
                                                       descriptor_order = descriptor_order,
                                                       columns_not_to_normalize = columns_not_to_normalize)

    
        #%%
        ###################
        ### NORMALIZING ###
        ###################
        
        ### DEFINING INDICES THAT ARE NOT NORMALIZED
        columns_to_normalize = list(descriptors_df_reorg.columns[~descriptors_df_reorg.columns.isin(columns_not_to_normalize)])
        
        ## DEFINING ARRAY
        descriptors_array = np.array(descriptors_df_reorg[columns_to_normalize])
        
        ## MINMAX RESCALING
        min_max_scalar = MinMaxScaler()
        ## FITTING
        descriptor_normalized = min_max_scalar.fit_transform( descriptors_array ) 
    
        ## COPYING DATAFRAME
        descriptors_df_norm = descriptors_df_reorg.copy()
        ## STORING
        descriptors_df_norm[columns_to_normalize] = descriptor_normalized
        
        
        #%%
        
        #################
        ### R2 MATRIX ###
        #################
        
        ## GETTING NORMALIZED ARRAY
        descriptors_norm_array = descriptors_df_norm[columns_to_normalize].to_numpy()
                
        ## COMPUTING R SQUARED MATRIX
        r2_matrix= compute_r_sq_matrix(descriptors_array = descriptors_norm_array)
        
        ## CREATING DATAFRAME
        r2_matrix_df = pd.DataFrame(r2_matrix, 
                                    columns = columns_to_normalize)
        r2_matrix_df['label'] = columns_to_normalize
        
        
        ## DEFINING TICKS
        cbar_kw = {'ticks': np.arange(0.0, 1.2, 0.2)}
        cbar_y_tick_labels = [ "%.1f"%(each_value) for each_value in cbar_kw['ticks']]
        
        ## PLOTTING HEAT MAP
        fig, ax = plot_heat_map(  descriptors_df_norm = r2_matrix_df,
                                  y_labels = columns_to_normalize,
                                  descriptor_keys =columns_to_normalize,
                                  title = None,
                                  cmap = CMAP, # CMAP
                                  cbar_kw = cbar_kw,
                                  want_color_bar=True,
                                  cbar_y_tick_labels = cbar_y_tick_labels,
                                  cbarlabel="Normalized values") # 'Rescaled descriptors'
        
        ## STORING FIGURE
        print_tools.store_figure( fig = fig, 
                                  path = os.path.join(output_image_path, "r2_matrix"),
                                  save_fig = save_fig,
                                  )
    
    
        #%%
        
        ###########################
        ### NORMALIZED HEAT MAP ###
        ###########################
        
        ## DEFINING FIGURE SIZE IN CENTIMETERS
#        fig_size_cm=(8.3, 8.3)
#        fig_size_cm=(9,9)
        fig_size_cm=(8,8)
        figsize = cm2inch( *fig_size_cm )
        
        ## DEFINING LABELS
        labels = list(descriptors_df_norm['label'])
        
        ## DEFINING KEYS
        descriptor_keys = list(columns_to_normalize)    
        
        ## DEFINING TICKS
        cbar_kw = {'ticks': np.arange(0, 1.2, 0.2)}
        
        ## PLOTTING HEAT MAP
        fig, ax = plot_heat_map(  descriptors_df_norm = descriptors_df_norm,
                                  y_labels = labels,
                                  figsize = figsize,
                                  descriptor_keys =descriptor_keys,
                                  title = None,
                                  cmap = CMAP, # CMAP
                                  cbar_kw = cbar_kw,
                                  want_color_bar=True,
                                  want_text = False, # Turning False text
                                  cbarlabel="Normalized values",
                                  want_minor = True) # 'Rescaled descriptors'
        
        ## GETTING TIGHT LAYOUT
        fig.tight_layout()
        
        #%%
        ## GETTING PATH TO IMAGE
        path_to_image =  os.path.join(output_image_path, "heat_map_rescaled")
        
        
        ## STORING FIGURE
        print_tools.store_figure( fig = fig, 
                                  path = path_to_image,
                                  fig_extension = 'svg',
                                  save_fig = True,
                                  bbox_inches = 'tight',
                                  )
        
        ## STORING CSV
        descriptors_df_norm.to_csv(path_to_image + ".csv")
        
        #%%
        
        ## DEFINING TICKS
        cbar_kw = {'ticks': np.arange(0.0, 1.2, 0.2)}
        cbar_y_tick_labels = [ "%.1f"%(each_value) for each_value in cbar_kw['ticks']]
        ## GETTING PLOTS OF NORMAL
        fig, ax = plot_heat_map(  descriptors_df_norm = descriptors_df_norm,
                                  descriptor_actual_values = descriptors_df_reorg,
                                  y_labels = labels,
                                  descriptor_keys =descriptor_keys,
                                  title =  None,
                                  cbar_kw = cbar_kw,
                                  cbar_y_tick_labels = cbar_y_tick_labels,
                                  cmap = CMAP,
                                  want_color_bar=True,
                                  cbarlabel="Normalized values") # 'Descriptors for ROT ligands',
        
        path_to_image =  os.path.join(output_image_path, "heat_map_values")
        
        ## STORING FIGURE
        print_tools.store_figure( fig = fig, 
                                  path = path_to_image,
                                  save_fig = save_fig,
                                  bbox_inches = 'tight',
                                  )
        
        ## STORING CSV
        descriptors_df_reorg.to_csv(path_to_image + ".csv")
    
        #%%
        #####################################################################################
        #####################################################################################
        #####################################################################################
        
        ####################################
        ### PRINCIPAL COMPONENT ANALYSIS ###
        ####################################
        ## DEFINING NUMBER OF COMPONENTS
        n_components = 2
        # default is 2
        
        ## RUNNING PRINCIPAL COMPONENT ANALYSIS
        finalDf, pca, pca_columns = run_principal_component_analysis(descriptors_df_reorg = descriptors_df_reorg,
                                                                     descriptor_keys = descriptor_keys,
                                                                     n_components = n_components)
        
    
        #%%
        ## DEFINING TARGETS
        targets = descriptors_df_reorg.label
        # ['C1', 'C2', 'C4', 'C6', 'C8', 'CYC', 'C10',
        # 'C3OH', 'C3NH2', 'BENZ', 'CYC_BENZ', 'CYC_ISO', 'OH']
        
        ## DEFINING AXIS LIMITS
#        axis_limits = [-4, 7, 2]
        axis_limits = [-4, 4, 2]
        ## DEFINING FIGURE SIZE
        fig_size_cm=(9, 9) # cm
        fig_size_cm=(8, 8) # cm
        fig_size_cm=(7, 7) # cm
#        fig_size_cm=(8.3, 8.3) # cm
        
        ## PLOTTING PCA
        fig, ax = plot_pca(df = finalDf,
                           fig_size_cm=fig_size_cm,
                           axis_limits = axis_limits,
                           want_grid = False, # Turning grid off
                           )
        
        ## SETTING AX TICKS
        ticks = np.arange(axis_limits[0], axis_limits[1] + axis_limits[2], axis_limits[2])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        ## TIGHT
        fig.tight_layout()
        
        #%%
        ## STORING FIGURE
        print_tools.store_figure( fig = fig, 
                                  path = os.path.join(output_image_path, "pca_descriptors"),
                                  save_fig = True,
                                  fig_extension = 'svg',
                                  bbox_inches = 'tight',
                                  )
        
    
        #%%
        
        ##################################
        ### MOST IMPORTANT DESCRIPTORS ###
        ##################################
        ## EXPLAINED VARIANCE
        '''
        pca.explained_variance_ratio_
        When going down to lower dimensional space, you lose some variance or information 
        about the descriptors. Therefore, we use the explained variance ratio to get the extent of
        informatoin. I find the following:
            array([0.65562783, 0.23847322])
        This means that PC1 has 65.5% of the variance and PC2 has 23.8% of the variance. 
        '''
        ## IMPORTANCE OF EACH FEATURE REFLECTED BY MAGNITUDE OF CORRESPONDING EIGEN VECTORS
        '''
        print(abs( pca.components_ ))
        PCA.COMPONENTS_ HAS SHAPE OF [N_COMPONENTS, N_FEATURES]. THUS PC1 IS FOR FIRST ROW
            print(abs( pca.components_ ))
            [[ 0.35717825  0.32809815  0.39091418  0.36331948  0.3972422   0.15791641
               0.2794203   0.3890957   0.26028876]
             [ 0.30004756  0.42100611  0.1438234   0.2910471   0.12939831  0.64769119
               0.35627141  0.23537439  0.09356026]]
        
        Variables with higher values are more important
        '''
        ## USING PCA TO GET THE MOST IMPORTANT DESCRIPTORS
        importance_features_dict = pca_get_most_important_features(pca = pca, 
                                                                   descriptor_keys = descriptor_keys)
    
        #%%
        
        ## DEFINING Y LIMITS
        y_limits = [0, 0.8, 0.2]
        
        ## LOOPING THROUGH EACH COMPONENT
        for index, pc_name in enumerate(importance_features_dict):
    
            ## DEFINING FEATURES AND EIGEN VALUES
            features_list = np.array(importance_features_dict[pc_name]['features'])
            abs_eigenvalue = np.array(importance_features_dict[pc_name]['abs. eigenvalues'])
            
            ## PLOTTING
            fig, ax = plot_importance_pc_values( features_list = features_list,
                                                 abs_eigenvalue = abs_eigenvalue,
                                                 pc_name = pc_name,
                                                 variance_value = pca.explained_variance_ratio_[index],
                                                 y_limits = y_limits,)
            ## STORING FIGURE
            print_tools.store_figure( fig = fig, 
                                      # path = os.path.join(DEFAULT_IMAGE_OUTPUT, 'importance_' + pc_name ),
                                      path = os.path.join(output_image_path, 'importance_' + pc_name ),
                                      bbox_inches = 'tight',
                                      save_fig = save_fig,
                                      )
                
        #%%
        ##################################
        ### PAIRING PCA TO EXP RESULTS ###
        ##################################

        ## DEFINING FIGURE SIZE
        fig_size_cm=(8.3, 8.3)
        
        ### DICTIONARY FOR PREDICTIONS
        correlation_dict = {
                'membrane_binding':
                    {'csv_name': 'membrane_binding',
                     'output_string' : "Mass",
                     'fig_limits': {
                             'x': [-0.5, 7.5, 1],
                             'y': [-0.5, 7.5, 1]
                             },
                     'xlabel': "Exp. Mass Uptake (ng/cm$^2$)",
                     'ylabel': "Model Mass Uptake (ng/cm$^2$)",
                     },
                'immune_resp_1':
                    {'csv_name': 'moyano_2014_fig_1',
                     'output_string' : "Immune Res.",
                     'fig_limits': {
                             'x': [-0.1, 1.2, 0.2],
                             'y': [-0.1, 1.2, 0.2],
                             },
                     'xlabel': "Exp. immune response",
                     'ylabel': "Model immune response",
                     },
                'immune_resp_2':
                    {'csv_name': 'moyano_2014_fig_2',
                     'output_string' : "Immune Res.",
                     'fig_limits': {
                             'x': [0.03, 0.08, 0.01],
                             'y': [0.03, 0.08, 0.01],
                             },
                     'xlabel': "Exp. immune response",
                     'ylabel': "Model immune response",
                     },
                ## CHEN 2014
                'chen_2014_BLGA':
                    {'csv_name': 'chen_2014_fig_2',
                     'output_string' : "BLGA",
                     'fig_limits': {
                             'x': [0, 12, 2],
                             'y': [0, 12, 2],
                             },
                     'xlabel': "Exp. K$_b$ * 10$^5$ (M$^{-1}$)",
                     'ylabel': "Model K$_b$ * 10$^5$ (M$^{-1}$)",
                     },
                'chen_2014_BLGB':
                    {'csv_name': 'chen_2014_fig_2',
                     'output_string' : "BLGB",
                     'fig_limits': {
                             'x': [0, 12, 2],
                             'y': [0, 12, 2],
                             },
                     'xlabel': "Exp. K$_b$ * 10$^5$ (M$^{-1}$)",
                     'ylabel': "Model K$_b$ * 10$^5$ (M$^{-1}$)",
                     },
                ## SAHA 2016
                'saha_2016_uptake':
                    {'csv_name': 'saha_2016_fig5_7',
                     'output_string' : "Uptake",
                     'fig_limits': {
                             'x': [6, 16, 2],
                             'y': [6, 16, 2],
                             },
                     'xlabel': "Exp. Uptake (ng/well)",
                     'ylabel': "Model Uptake (ng/well)",
                     },
                    
                'saha_2016_C4BPA':
                    {'csv_name': 'saha_2016_fig5_7',
                     'output_string' : "NpSpCk of C4BPA",
                      'fig_limits': {
                              'x': [0.05, 0.50, 0.1],
                              'y': [0.05, 0.50, 0.1],
                              },
                     'xlabel': "Exp. NpSpCk of C4BPA",
                     'ylabel': "Model NpSpCk of C4BPA",
                     },
                'saha_2016_IGLC2':
                    {'csv_name': 'saha_2016_fig5_7',
                     'output_string' : "NpSpCk of IGLC2",
                      'fig_limits': {
                              'x': [1, 5, 0.5],
                              'y': [1, 5, 0.5],
                              },
                     'xlabel': "Exp. NpSpCk of IGLC2",
                     'ylabel': "Model NpSpCk of IGLC2",
                     },
                    
                ## LI 2014
                'li_2014_fig2':
                    {'csv_name': 'li_2014_fig2',
                     'output_string' : "MIC (nM)",
                       'fig_limits': {
                               'x': [-1500, 6000, 1500],
                               'y': [-1500, 6000, 1500],
                               },
                     'xlabel': "Exp. MIC (nM)",
                     'ylabel': "Model MIC (nM)",
                     },
                    
                }
            
            
        ## MAKING FULL PREDICTION
        fig, ax = full_prediction_given_dict(
                                    fig_size_cm = fig_size_cm,
                                    path_csv = PATH_CSV,
                                    want_title = True,
                                    save_fig = False,
                                    output_csv_dir = OUTPUT_IMAGE_PATH,
                                    **correlation_dict['immune_resp_1'],
                                    )
        
        #%%
        
        ## REMOVING TITLE
        ax.get_figure().gca().set_title("")
        
        fig.tight_layout()
        
        ## STORING FIGURE
        ## DEFINING PATH
        path_output = r"/Users/alex/Box Sync/Personal/2020_Job_Applications/20200724-DOW/Research_summary/Figure"
        print_tools.store_figure( fig = fig, 
                                  fig_extension = 'svg',
                                  path = os.path.join(path_output, 'immune_response' ),
                                  bbox_inches = 'tight',
                                  save_fig = True,
                                  )
        
#        ## UPDATING FIGURE SIZE
#        plot_tools.update_fig_size(fig,
#                    fig_size_cm = (4.2,4.2),
#                    tight_layout = True)
        
        
        #%%
        ## LOOPING THROUGH ANALYSIS TYPE
        if analysis_type == 'All':
            # LOOPING THROUGH EACH CORRELATION DICT
            for each_correlation_dict in correlation_dict:
                ## MAKING FULL PREDICTION
                full_prediction_given_dict(
                                            fig_size_cm = fig_size_cm,
                                            path_csv = PATH_CSV,
                                            save_fig = True,
                                            **correlation_dict[each_correlation_dict]
                                            )
        else:
            ## MAKING FULL PREDICTION
            full_prediction_given_dict(
                                        fig_size_cm = fig_size_cm,
                                        path_csv = PATH_CSV,
                                        **correlation_dict['membrane_binding'],
                                        )
        # PCA EXPECTED TO HAVE 0 INTERCEPT
        # https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca
            
        #%% SUPPORT VECTOR MACHINES
        
        ## PCA: MISSING VALUES COULD BE  COMPUTED: 
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
            
        ## REFERENCE FOR MANIFOLD LEARNING
        # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
            
        
            
        # SUPPORT VECTOR REGRESSION (SVR)
        # https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
        
        #%% PLOTTING EXPERIMENTAL CORRELATIONS
        
        ### FUNCTION TO GENERATE LINEAR REGRESSION
        def generate_linear_regression_from_csv(csv_name,
                                                path_csv,
                                                x,
                                                y,
                                                output_image_path = '.',
                                                label = None,
                                                xlabel = None,
                                                ylabel = None,
                                                fig_limits = None,
                                                save_fig = False,
                                                output_csv_dir = None,
                                                ):
            '''
            The purpose of this function is to generate a linear regression 
            given a CSV file and labels. 
            INPUTS:
                csv_name: [str]
                    name of csv file
                path_csv: [str]
                    path to look for csv file
                x: [str]
                    column of x datas to look at
                y: [str]
                    column of y datas to look at
                output_image_path: [str]
                    place to put your output image
                xlabel: [str]
                    label for x-axis
                ylabel: [str]
                    label for y-axis
                fig_limits: [dict]
                    dictionary with figure limits
                save_fig: [logical]
                    True if you want to save the figure
                output_csv_dir: [str]
                    None if you don't want to print out csv
            OUTPUTS:
                fig, ax: [obj]
                    figure and axis for the plot
            '''
        
            ## DEFINING CSV FILE
            csv_file = csv_name + '.csv'
            
            ## DEFINING FULL PATH TO CSV
            full_csv_path = os.path.join(path_csv,
                                         csv_file)
            
            ## LOADING DATA
            exp_data = pd.read_csv(full_csv_path)
            
            ## GETITNG X, Y, LABELS
            X_array = exp_data[x].to_numpy()[:, np.newaxis]
            y_array = exp_data[y].to_numpy()
            data_labels = exp_data[label].to_numpy()

            ## GENERATING LINEAR REGRESSION
            regr, y_pred = generate_linear_regression_with_bias(X_array = X_array, 
                                                                y_array = y_array)
            
            ## PRINTING RESULTS
            print_linear_regression_accuracy(regr = regr, 
                                             y_array = y_array, 
                                             y_pred = y_pred)
            
            
            ## GETTING THE EQUATION FOR LINEAR REGRESSION
            full_str = find_equation_linear_regression(regr,
                                                       [x],
                                                       fmt="%.2f",
                                                       output_string= y,
                                                       )
                
            ## PLOTTING PARITY PLOT
            fig, ax = plot_scatter_parity(exp_values = y_array,
                                          pred_values = y_pred,
                                          title = full_str,
                                          labels = data_labels,
                                          xlabel = xlabel,
                                          ylabel = ylabel,
                                          scatter_dict = { 'marker': '.',
                                                           'color': 'k',
                                                           's': 200},
                                          fig_size_cm = fig_size_cm,
                                          fig_limits = fig_limits)
            
            if output_csv_dir is not None:
                ## PRINTING CSV
                print_csv_pred_model(y_exp = y_array,
                                     y_pred = y_pred,
                                     labels = data_labels,
                                     equation_string = full_str,
                                     path_csv_dir = output_csv_dir,
                                     csv_name = csv_name,
                                     )
            
            ## STORING FIGURE
            print_tools.store_figure( fig = fig, 
                                      path = os.path.join(output_image_path,  csv_name + '-' + y +'-exp_correlations_parity' ),
                                      bbox_inches = 'tight',
                                      save_fig = save_fig,
                                      )
            return fig, ax
        
        
        
        ## DEFINING OUTPUT CSV PATH
        path_output_image = os.path.join(OUTPUT_IMAGE_PATH,
                                         'exp_correlations')
        
        ## MAKING DIRECTORY
        if os.path.isdir(path_output_image) is not True:
            os.mkdir(path_output_image)
        
        ## LOADING FILE
        path_csv = PATH_CSV
        ## DEFINING CSV FILE
        csv_file = 'moyano_2014_fig_1.csv'
        
        ## DEFINING LABELS
        exp_dict = {
                ## MEMBRANE STUDIES BY LOCHBAUM
                'membrane_binding': {
                    'csv_name': correlation_dict['membrane_binding']['csv_name'],
                    'x': 'log P',
                    'y': correlation_dict['membrane_binding']['output_string'],
                    'label': 'label',
                    'xlabel': correlation_dict['membrane_binding']['xlabel'],
                    'ylabel': correlation_dict['membrane_binding']['ylabel'],
                    'fig_limits': correlation_dict['membrane_binding']['fig_limits']
                    },
                
                ## MOYANO 2014
                'immune_resp_1': {
                    'csv_name': 'moyano_2014_fig_1',
                    'x': 'log P',
                    'y': 'Immune Res.',
                    'label': 'label',
                    'xlabel': correlation_dict['immune_resp_1']['xlabel'],
                    'ylabel': correlation_dict['immune_resp_1']['ylabel'],
                    'fig_limits': correlation_dict['immune_resp_1']['fig_limits']
                    },
                'immune_resp_2': {
                    'csv_name': 'moyano_2014_fig_2',
                    'x': 'log P',
                    'y': 'Immune Res.',
                    'label': 'label',
                    'xlabel': correlation_dict['immune_resp_2']['xlabel'],
                    'ylabel': correlation_dict['immune_resp_2']['ylabel'],
                    'fig_limits': correlation_dict['immune_resp_2']['fig_limits']
                    },
                
                ## LI 2014
                'li_2014_fig2': {
                    'csv_name': correlation_dict['li_2014_fig2']['csv_name'],
                    'x': 'log P',
                    'y': correlation_dict['li_2014_fig2']['output_string'],
                    'label': 'label',
                    'xlabel': correlation_dict['li_2014_fig2']['xlabel'],
                    'ylabel': correlation_dict['li_2014_fig2']['ylabel'],
                    'fig_limits': correlation_dict['li_2014_fig2']['fig_limits']
                    },
                ## CHEN 2014
                'chen_2014_BLGA':{
                    'csv_name': correlation_dict['chen_2014_BLGA']['csv_name'],
                    'x': 'log P',
                    'y': correlation_dict['chen_2014_BLGA']['output_string'],
                    'label': 'label',
                    'xlabel': correlation_dict['chen_2014_BLGA']['xlabel'],
                    'ylabel': correlation_dict['chen_2014_BLGA']['ylabel'],
                    'fig_limits': correlation_dict['chen_2014_BLGA']['fig_limits']
                    },
                ## SECOND ONE
                'chen_2014_BLGB':{
                    'csv_name': correlation_dict['chen_2014_BLGB']['csv_name'],
                    'x': 'log P',
                    'y': correlation_dict['chen_2014_BLGB']['output_string'],
                    'label': 'label',
                    'xlabel': correlation_dict['chen_2014_BLGB']['xlabel'],
                    'ylabel': correlation_dict['chen_2014_BLGB']['ylabel'],
                    'fig_limits': correlation_dict['chen_2014_BLGB']['fig_limits']
                    },
                 ## SAHA 2016
                'saha_2016_C4BPA':{
                        'csv_name': correlation_dict['saha_2016_C4BPA']['csv_name'],
                        'x': 'log P',
                        'y': correlation_dict['saha_2016_C4BPA']['output_string'],
                        'label': 'label',
                        'xlabel': correlation_dict['saha_2016_C4BPA']['xlabel'],
                        'ylabel': correlation_dict['saha_2016_C4BPA']['ylabel'],
                        'fig_limits': correlation_dict['saha_2016_C4BPA']['fig_limits']
                        },
                
                'saha_2016_IGLC2':{
                        'csv_name': correlation_dict['saha_2016_IGLC2']['csv_name'],
                        'x': 'log P',
                        'y': correlation_dict['saha_2016_IGLC2']['output_string'],
                        'label': 'label',
                        'xlabel': correlation_dict['saha_2016_IGLC2']['xlabel'],
                        'ylabel': correlation_dict['saha_2016_IGLC2']['ylabel'],
                        'fig_limits': correlation_dict['saha_2016_IGLC2']['fig_limits']
                        },
                
                'saha_2016_uptake':{
                        'csv_name': correlation_dict['saha_2016_uptake']['csv_name'],
                        'x': 'log P',
                        'y': correlation_dict['saha_2016_uptake']['output_string'],
                        'label': 'label',
                        'xlabel': correlation_dict['saha_2016_uptake']['xlabel'],
                        'ylabel': correlation_dict['saha_2016_uptake']['ylabel'],
                        'fig_limits': correlation_dict['saha_2016_uptake']['fig_limits']
                        },
                
                
                
                }
                
        ## DEFINING LABELS
        labels = exp_dict['immune_resp_1']
        
        ## GENERATING LINEAR REGRESSION FROM CSV
        fig, ax = generate_linear_regression_from_csv(save_fig = save_fig, 
                                                      path_csv = path_csv, 
                                                      output_image_path = path_output_image,
                                                      output_csv_dir = OUTPUT_IMAGE_PATH,
                                                      **labels, )
        
        
        
        
        
        
        
        
        
    