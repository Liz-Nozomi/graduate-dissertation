# -*- coding: utf-8 -*-
"""
csv_tools.py
This script contains tools to work on multiple trajectories via csv files. The goal is to read the pickle file and export the results into a csv file.
This way, you can use the csv data to make a plot in another program, e.g. Origin

CREATED ON: 04/12/2018

CLASSES:
    csv_info_export: exports the csv data
    
FUNCTIONS:
    csv_info_new: creates a new csv info dictionary
    csv_info_add: adds csv info for a dictionary
    csv_info_check_type: checks the data type (i.e. type_1 or type_2)
    csv_info_decoder_add: addeds to csv_info the decoding names (great for sorting details for csv files)
    multi_csv_export: exports multiple data into a csv file
    csv_dict: [class] class function to combine add, check type, etc.
    
ASSUMPTIONS:
    CSV files can come in two flavors:
        type_1: each trajectory outputs multiple information as a single row
            e.g. Your_sim, data_1, data_2, ...
        type_2: each trajectory outputs a simulation plot as two/three columns (e.g. x, y, y_error)
            e.g. Your_sim
                x    y
                1    2
    NOTE that if your desired csv file is not one of these types, stop here! You may need to create your own functions at this point. Examples that are not supported:
        - two / three-dimensional matrices for 3D visualization purposes. For these cases, it may be more practical to store your variables into a pickle.
            
AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
** UPDATES **
20180413: Added function for csv_info to correctly account for dictionary types {'avg', 'std'}
20180529: Added functionality to multi_csv_export that can load trajectories, run analysis, then forget the pickle.
20180614: Adding class functionality to csv info
"""

## DEFINING MODULES
import os
import pandas as pd
import time
# 10 mins to pandas: http://pandas.pydata.org/pandas-docs/version/0.15/10min.html
## DECODER
from MDDescriptors.core.decoder import decode_name
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_multi_analysis_pickle

##############################################
### CLASS METHOD TO CREATE CSV FILES, ETC. ###
##############################################
class csv_dict:
    '''
    The purpose of this class is to create csv files based on input data.
    INPUTS:
        file_name: [str] file name
        decoder_type: [str] decoder type based on decode_name  -- used to interpret your file_name
    OUTPUTS:
        self.csv_info: [dict] main csv info
    ALGORITHM:
        - need a function to create a csv data structure
        - need a function to export the csv data
    FUNCTIONS:
        new: function to create new csv info
        add: function to add into the csv info
        export: function to export csv info
    '''
    ### INITIALIZATION
    def __init__(self, file_name, decoder_type = None):
        ### CREATING A CSV FILE
        self.csv_info = self.new(file_name)
        
        ### ADDING DECODER NAME IF AVAILABLE
        if decoder_type is not None:
            self.decoder_add( file_name = file_name, decoder_type = decoder_type )
        
        return
        
    ### FUNCTION TO CREATE NEW FILE BASED ON FILE NAME
    def new(self, file_name=None):
        '''
        The purpose of this function is to create a new csv_info dictionary:
            'data_title', 'data', 'labels', 'file_name'
            Note that 'file_name' is the name of the file that you have inputted
        INPUTS:
            file_name: name of the file as a string
        OUTPUTS:
            csv_info: dictionary to store csv information
        USAGE IN CLASS EXAMPLE: self.csv_info = csv_info_new(pickle_name)
        '''
        csv_info = {'file_name': file_name,
                    'data_title': [], 
                    'labels':[], 
                    'data':[]}
        return csv_info
    
    ### FUNCTION TO ADD DECODER NAMES
    def decoder_add(self, file_name, decoder_type=None):
        '''
        The purpose of this function is to add the decoded names into the csv file. We want this because it can help decode the information
        INPUTS:
            csv_info: dictionary to store csv information
            file_name: [str] name of the file
            decoder_type: [str] decoding scene from decoder.py
        OUTPUTS:
            self.csv_info: [dict] updated csv_info with the decoder information
        '''
        ## DECODING THE NAME
        decoded_name_dict = decode_name(file_name, decoder_type)
        ## ADDING TO CSV
        for each_key in decoded_name_dict.keys():        
            self.add( data_title = each_key, data = [ decoded_name_dict[each_key] ] )
        return
    
    ### FUNCTION TO DEAL WITH INPUTS FOR CSV_INFO
    def add(self, data_title, data, labels=None):
        '''
        The purpose of this function is to add to csv info
        INPUTS:
           self.csv_info: [dict] csv info dictionary contains the following options:
                'data_title': Data title you would like. This must always be defined!
                'data': list of data, which will be used
                    Note: Typically, this contains an [x,y]. If you have just a single value, simply insert [y]. 
                'labels': Labels for the data. For example, what is x called? Similarly, what is y called?
                    NOTE: if labels is None, then we will insert a [None]. This means that we will refer to the title as the label of this data.
            data: [list] list of data values
            data_title: [str] string with data title
            labels: [list] list of corresponding labels
        OUTPUTS:
            self.csv_info: [dict] updated csv info
        '''
        ## CHECKING DATA TYPE (SPECIAL CASES)
        ## SPECIAL CASE WHERE YOUR DATA IS SOMETHING LIKE {'avg', 'std'}
        if len(data) == 1 and type(data[0]) == dict:
            data_out_of_list = data[0]
            ## LOOP THROUGH THE KEYS AND ADD THE CURRENT LABEL
            for each_key in data_out_of_list.keys():
                new_title = data_title + '_' + each_key
                ## STORING NEW DATA
                self.csv_info['data_title'].append(new_title)
                self.csv_info['data'].append([data_out_of_list[each_key]])
                self.csv_info['labels'].append([None])
        else:
            ## STORING THE NEW DATA
            self.csv_info['data_title'].append(data_title)
            self.csv_info['data'].append(data)
        
            if labels == None:
                self.csv_info['labels'].append([None])
            else:
                ## CHECKING IF LENGTH OF DATA IS THE SAME AS THE LABELS
                if len(data) != len(labels):
                    print("POSSIBLE ERROR! Length of data is %d, whereas labels is %d"%(len(data),len(labels)))
                    print("DOUBLE CHECK YOUR INPUTS!")
                self.csv_info['labels'].append(labels)
        return
    
    ### FUNCTION TO EXPORT
    def export(self,  desired_titles = None, df_1 = None, dfs_2 = [], dfs_2_names = [], want_export=True):
        '''
        The purpose of this function is to export everything coming from the csv info
        INPUTS:
            self: class object
        OUTPUTS:
            text file exported
        '''
        csv_info_export( csv_info = self.csv_info, desired_titles = None, df_1 = None, dfs_2 = [], dfs_2_names = [], want_export=True )
        return
   

### FUNCTION TO CREATE CSV INFO DICTIONARY
def csv_info_new(file_name=None):
    '''
    The purpose of this function is to create a new csv_info dictionary:
        'data_title', 'data', 'labels', 'file_name'
        Note that 'file_name' is the name of the file that you have inputted
    INPUTS:
        file_name: name of the file as a string
    OUTPUTS:
        csv_info: dictionary to store csv information
    USAGE IN CLASS EXAMPLE: self.csv_info = csv_info_new(pickle_name)
    '''
    csv_info = {'file_name': file_name,
                'data_title': [], 
                'labels':[], 
                'data':[]}
    return csv_info
    
### FUNCTION TO DEAL WITH INPUTS FOR CSV_INFO
def csv_info_add(csv_info, data_title, data, labels=None):
    '''
    The purpose of this function is to add to csv info
    INPUTS:
        csv_info: [dict] csv info dictionary contains the following options:
            'data_title': Data title you would like. This must always be defined!
            'data': list of data, which will be used
                Note: Typically, this contains an [x,y]. If you have just a single value, simply insert [y]. 
            'labels': Labels for the data. For example, what is x called? Similarly, what is y called?
                NOTE: if labels is None, then we will insert a [None]. This means that we will refer to the title as the label of this data.
        data: [list] list of data values
        data_title: [str] string with data title
        labels: [list] list of corresponding labels
    OUTPUTS:
        csv_info: updated csv info
    '''      
    ## CHECKING DATA TYPE (SPECIAL CASES)
    ## SPECIAL CASE WHERE YOUR DATA IS SOMETHING LIKE {'avg', 'std'}
    if len(data) == 1 and type(data[0]) == dict:
        data_out_of_list = data[0]
        ## LOOP THROUGH THE KEYS AND ADD THE CURRENT LABEL
        for each_key in data_out_of_list.keys():
            new_title = data_title + '_' + each_key
            ## STORING NEW DATA
            csv_info['data_title'].append(new_title)
            csv_info['data'].append([data_out_of_list[each_key]])
            csv_info['labels'].append([None])
    else:
        ## STORING THE NEW DATA
        csv_info['data_title'].append(data_title)
        csv_info['data'].append(data)
    
        if labels == None:
            csv_info['labels'].append([None])
        else:
            ## CHECKING IF LENGTH OF DATA IS THE SAME AS THE LABELS
            if len(data) != len(labels):
                print("POSSIBLE ERROR! Length of data is %d, whereas labels is %d"%(len(data),len(labels)))
                print("DOUBLE CHECK YOUR INPUTS!")
            csv_info['labels'].append(labels)
    return csv_info

### FUNCTION TO ADD DECODER NAMES
def csv_info_decoder_add(csv_info, file_name, decoder_type=None):
    '''
    The purpose of this function is to add the decoded names into the csv file. We want this because it can help decode the information
    INPUTS:
        csv_info: dictionary to store csv information
        file_name: [str] name of the file
        decoder_type: [str] decoding scene from decoder.py
    OUTPUTS:
        csv_info: updated csv_info with the decoder information
    '''
    ## DECODING THE NAME
    decoded_name_dict = decode_name(file_name, decoder_type)
    ## ADDING TO CSV
    for each_key in decoded_name_dict.keys():        
        csv_info = csv_info_add(csv_info, data_title = each_key, data = [ decoded_name_dict[each_key] ] )
    return csv_info

### FUNCTION TO DENOTE THE TYPES
def csv_info_check_type(csv_info, index):
    '''
    This function checks the csv_info type by going through one of the indices, then looking if label is None or not. If label is None, then we have a type 1. If not, we have a type 2
    INPUTS:
        csv_info: [dict] csv info dictionary contains 'data_title', 'data', 'labels'
        index: [int] index within csv info to check the type
    OUTPUT:
        info_type: 
            'type_1'
            'type_2'
    '''
    if csv_info['labels'][index][0] is None:
        info_type = 'type_1'
    else:
        info_type = 'type_2'
    return info_type
    
### FUNCTION TO EXPORT FOR TYPE 1
def csv_info_export_type_1(csv_info,indexes, export_csv_name = 'export_type_1.csv', df=None, want_export=True):
    '''
    The purpose of this script is to export csv info for type 1:
        e.g. Your_sim, data_1, data_2, ...
    INPUTS:
        csv_info: [dict] csv info dictionary contains 'file_name', 'data_title', 'data', 'labels'
        indexes: [list] list of indexes that will be exported as a type 1 configuration
        export_csv_name: [string] name of the exported csv
        df: [pd.database] pandas database
    OUTPUTS:
        df: [pd.dataframe] pandas database
    '''
    ## CREATING LIST OF LABELS
    label_list = [ csv_info['data_title'][index] for index in indexes ]
    ## ADDING TO LABEL LIST
    label_list = ['File_name'] + label_list
    ## CREATING A LIST OF DATA
    data_list = [ csv_info['data'][index][0] for index in indexes ]
    ## ADDING TO DATA LIST
    data_list = [csv_info['file_name']] + data_list
    ## GETTING THE DATA
    df_type_1 = pd.DataFrame( data_list, index = label_list ).T # , columns = label_list
    if df is not None:
        df = df.append(df_type_1)
    else:
        df = df_type_1.copy()
    ### EXPORTING TO FILE
    if want_export is True:
        print("WRITING TO FILE: %s"%(export_csv_name))
        df.to_csv(export_csv_name)
    return df

### FUNCTION TO EXPORT FOR TYPE_2
def csv_info_export_type_2(csv_info, indexes, dfs = [], dfs_names = [], want_export=True):
    '''
    The purpose of this script is to export csv info for type 2:
        type_2: each trajectory outputs a simulation plot as two/three columns (e.g. x, y, y_error)
            e.g. Your_sim
                x    y
                1    2
    INPUTS:
        csv_info: [dict] csv info dictionary contains 'file_name', 'data_title', 'data', 'labels'
        indexes: [list] list of indexes that will be exported as a type 2 configuration
        dfs: [list] list of panda databases
        dfs_names: [list] list of strings corresponding to the databases
        want_export: True if you want to export
    OUTPUTS:
        dfs: [list] list of panda databases
        dfs_names: [list] list of strings associated to the panda databases
    '''
    file_name = csv_info['file_name']
    ## LOOPING THROUGH EACH INDEX
    for each_index in indexes:
        ## DEFINING THE CSV FILE LABELS
        labels = csv_info['labels'][each_index]
        ## DEFINING THE DATA
        data = csv_info['data'][each_index]
        ## DEFINING THE TITLE
        title = csv_info['data_title'][each_index]
        ## DEFINING CSV EXPORT NAME
        export_csv_name = csv_info['file_name'] + '_' + title + '.csv'
        
        ## CREATING DATABASE
        df = pd.DataFrame( data, index = labels ).T
        ## ADDING HEADER NAME
        columns = [ tuple([file_name, each_column]) for each_column in df.columns ]
        ## RENAMING THE COLUMNS
        df.columns = pd.MultiIndex.from_tuples(columns)
        
        ## STARTING NEW DATABASE IF THE DATABASE HAS NOT BEEN MADE
        if title not in dfs_names:
            ## APPENDING TO NAMES
            dfs_names.append(title)
            ## APPENDING TO DATABASE
            dfs.append(df)
        else:
            ## FINDING INDEX
            df_index = dfs_names.index(title)
            print("index: %d"%(df_index))
            ## ADDING TO CURRENT DATABASE
            dfs[df_index] = pd.concat( [dfs[df_index], df], axis = 1 )
        '''
        OUTPUT EXAMPLE:
                 Simulation time (ns)  Number of adsorbed thiols
            0                    0.00                        9.0
            1                    0.05                       92.0
            2                    0.10                       98.0
            3                    0.15                       99.0
        
        '''
        ## WRITING TO CSV
        if want_export is True:
            print("WRITING TO FILE: %s"%(export_csv_name))
            df.to_csv(export_csv_name)
    return dfs, dfs_names

################################
### CLASS TO EXPORT CSV INFO ###
################################

### FUNCTION TO EXPORT FOR A SINGLE TRAJECTORY
class csv_info_export:
    '''
    The purpose of this script is to export csv_info for one given dictionary. 
    INPUTS:
        csv_info: [dict] 
            csv info dictionary contains 'file_name', 'data_title', 'data', 'labels'
        desired_titles: [list] 
            desired titles as strings that you want data for. If None, we will try to export everything!
    OUTPUTS:
        csv file for all type 2 csvs (x, y's)
        csv files for all type 1 csv (single csv file)
        self.titles: [list] the available titles that will be outputted
        
    FUNCTIONS:
        check_titles: checks the title of the input
    '''
    ### INITIALIZATION
    def __init__(self, csv_info, 
                 desired_titles = None, 
                 df_1 = None, 
                 dfs_2 = [], 
                 dfs_2_names = [], 
                 want_export=True ):
        ## STORING CSV INFO
        self.csv_info = csv_info
        
        ## CHECKING THE TITLES
        self.check_titles(desired_titles)

        ## FINDING THE TYPES
        self.find_types()
        
        ## STARTING WITH INITIAL VARIABLES
        self.df_1, self.dfs_2, self.dfs_2_names = df_1 , dfs_2, dfs_2_names
        
        ## WRITING FOR TYPE 1
        if len(self.type_1_index) != None:
            self.df_1 = csv_info_export_type_1(self.csv_info, 
                                               self.type_1_index, 
                                               df = self.df_1, 
                                               want_export = want_export)
        
        if len(self.type_2_index) != None:
            self.dfs_2, self.dfs_2_names = csv_info_export_type_2(self.csv_info, 
                                                                  self.type_2_index, 
                                                                  dfs = self.dfs_2, 
                                                                  dfs_names = self.dfs_2_names, 
                                                                  want_export = want_export)
        
        return
    ### FUNCTION TO CHECK TITLES
    def check_titles(self,desired_titles):
        '''
        This function checks if the desired titles are within the csv info file
        INPUTS:
            self: class object
            desired_titles:[list] desired titles as a list of strings
        OUTPUTS:
            self.titles: [list] the available titles that will be outputted
        '''
        if desired_titles is not None:
            ## CREATING EMPTY LIST FOR CORRECTED TITLE
            self.titles = []
            ## FIRST CHECK IF DESIRED TITLE IS AVAILABLE
            for each_title in desired_titles:
                if each_title in self.csv_info['data_title']:
                    self.titles.append(each_title)
                else:
                    print("%s is not available in csv_info. Continuing without it... Double check inputs for desired titles"%(each_title))
        else:
            self.titles = self.csv_info['data_title'][:]
            print("Since no desired titles stated, we are printing all possible csv files!")
        return
    
    ### FUNCTION TO FIND THE TYPES
    def find_types(self):
        '''
        The purpose of this function is to find the types of each of the titles
        INPUTS:
            self: class object
        OUTPUTS:
            self.type_1_index: [list] type 1 indices
            self.type_2_index: [list] type 2 indices
        '''
        ## FINDING INDEX OF EACH TITLE
        title_index = [ index for index, each_title in enumerate(self.csv_info['data_title']) if each_title in self.titles ]
    
        ## CREATING EMPTY ARRAY FOR TYPE AND TYPE 2 INDEX
        self.type_1_index = []
        self.type_2_index = []
        ## LOOPING THROUGH EACH TITLE AND DESIGNATING WHETHER OR NOT THE INFO IS TYPE 1 OR TYPE 2
        for each_index in title_index:
            ## CHECKING THE TYPE OF CSV INFO YOU HAVE FOR EACH
            info_type = csv_info_check_type(self.csv_info, each_index)
            if info_type == 'type_1':
                self.type_1_index.append(each_index)
            elif info_type == 'type_2':
                self.type_2_index.append(each_index)
        return

### FUNCTION TO CHECK IF FILE EXISTS, THEN REMOVES IF TRUE
def remove_file(text_file):
    '''
    The purpose of this function is to remove any file
    INPUTS:
        self: class object
    OUTPUTS:
        removed file
    '''
    if os.path.exists(text_file) is True:
        print("Initial %s found! Removing..."%(text_file))
        os.remove(text_file)
        return

### FUNCTION TO RUN MULTIPLE CSV EXPORTS
def multi_csv_export(export_class, Date = None, Descriptor_class = None, traj_results = None, list_of_pickles = None,  desired_titles = None, 
                     decoder_type = None, export_text_file_name = 'export',
                     df_1 = None, dfs_2 = [], dfs_2_names = [], want_export = False, **EXPORT_CLASS_ARGS):
    '''
    The purpose of this function is to run multiple csv exports using csv_info_export function
    INPUTS:
        export_class: [class] plotting class/function
        ## METHOD 1: loading all trajectories, then sending them to here
            traj_results: list of results as a list given by "load_multi_traj_pickles"
            list_of_pickles: list of pickles from the "load_multi_traj_pickles"
        ## METHOD 2: loading each trajectory one at a time, then running the export class
            Date: [str, default=None] date that you want
                NOTE: This has been upgraded to check for multiple dates
            Descriptor_class: [class] class used for desccriptor
                NOTE: This has been upgraded to check for multiple descriptor classes
        desired_titles: [list] desired titles that you want. If None, then we will print all data taken from csv info        
        decoder_type: way to decode your directory name
        export_text_file_name: name of text file to export to (no .csv)
    '''
    ## REMOVING ANY TEXT FILE OF THE NAME
    remove_file(export_text_file_name)
    
    ## TEMPORARY FIX -- TURNING OFF ALL RETRIEVAL CAPABILITIES FOR DATABASES
    df_1 = None
    dfs_2 = []
    dfs_2_names = []
    
    ## SEEING IF ANY METHOD WAS SELECTED
    if Date is None and Descriptor_class is None and traj_results is None and list_of_pickles is None:
        print("Error! You have not selected a choice of method. Either the following: ")
        print("METHOD 1: [DEFAULT] Load all the trajectories and run the multi_csv_export using traj_results and list_of_pickles")
        print("METHOD 2: Load each trajectory individually then running the exporitng using Date and Descriptor_class")
        
    if Date is not None and Descriptor_class is not None:
        ## FINDING ALL POSSIBLE PICKLES
        list_of_pickles = find_multi_traj_pickle(Date, Descriptor_class)
    
    ## LOOPING THROUGH LIST OF PICKLES
    for index, each_pickle in enumerate(list_of_pickles):
        ## PRINTING
        print("\nCOLLECTING DATA FOR %s"%(each_pickle))
        if Date is not None and Descriptor_class is not None:
            ## CHECKING TO SEE IF WE HAVE ONE DATE AND ONE DESCRIPTOR CLASS
            if type(Date) == str and type(Descriptor_class) == type:
                traj_results = load_multi_traj_pickle(Date, Descriptor_class, each_pickle )
            ## CASE WHERE WE HAVE A LIST
            else:
                print("Since Date and Descriptor class is not a string and a class, we have a multiple trajectory case!")
                traj_results = load_multi_traj_multi_analysis_pickle( Dates = Date,
                                                                      Descriptor_classes = Descriptor_class,
                                                                      Pickle_loading_file_names = each_pickle
                                                                     )
                
        ## RUNNING EXPORTED CLASS
        try:
            if Date is not None and Descriptor_class is not None:
                exported_class = export_class(traj_results, each_pickle, **EXPORT_CLASS_ARGS)
            elif traj_results is not None and list_of_pickles is not None:
                exported_class = export_class(traj_results[index], list_of_pickles[index], **EXPORT_CLASS_ARGS)
            ## USING THE RESULTS OF THE EXPORTED CLASS
            exported_csv = csv_info_export(exported_class.csv_info, desired_titles, 
                                       df_1 = df_1, dfs_2 = dfs_2, dfs_2_names = dfs_2_names, want_export = want_export)
            ## GETTING ALL DFS
            df_1, dfs_2, dfs_2_names= exported_csv.df_1, exported_csv.dfs_2, exported_csv.dfs_2_names
        except Exception:
            print("ERROR! COULD NOT EXTRACT FROM %s..."%(each_pickle)); 
            print("CONTINUING ANYWAYS!")
            time.sleep(1)
            pass
        
    ## NOW, PRINTING
    # TEXT NAMES
    if df_1 is not None:
        df_1_text_name = export_text_file_name + '.csv'
        print("WRITING TO: %s"%(df_1_text_name))
        df_1.to_csv(df_1_text_name)
    
    if len(dfs_2) != 0 :
        ## PRINTING FOR TYPE 2
        for index, each_df in enumerate(dfs_2):
            df_2_text_name = export_text_file_name + '_' + dfs_2_names[index] + '.csv'
            print("WRITING TO: %s"%(df_2_text_name))
            each_df.to_csv(df_2_text_name)
        
    return 
    # df_1, dfs_2, dfs_2_names


### FUNCTION TO LOAD THE PICKLE, 
