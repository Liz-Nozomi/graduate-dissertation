# -*- coding: utf-8 -*-
"""
multi_traj.py
The purpose of this script is to load all trajectories, use a corresponding function to analyze the trajectories, then save the information for subsequent plotting/analysis

INPUTS:
    directories
    files within directories that you are interested in

# USAGE: from multi_traj import multi_traj_analysis

CLASSES:
    multi_traj_analysis: runs analysis on multiple trajectories
    
FUNCTIONS:
    load_multi_traj_pickle: loads trajectory pickle
    load_multi_traj_multi_analysis_pickle: loads multiple analysis for multiple pickles
    load_multi_traj_pickles: loads multiple trajectory pickles
    find_multi_traj_pickle: finds all pickle within the multi traj pickle storage -- Can be used in conjunction with load_multi_traj_pickle
    add_to_all_dicts_of_dicts: Adds dictionary to all dictionaries -- useful for similar input for descriptor classes
    find_class_from_list: extracts class from a list of classes
    find_function_name_from_list: extracts a function name from a list of functions
    load_pickle_for_analysis: function that reloads a pickle for subsequent analysis

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
    
*** UPDATES ***
20180319 - AKC - Made multi_traj_analysis as a form of a list so multiple descriptors can be run in the same gro / xtc file
20180325 - AKC - Made multi_traj loading pickle scheme enable multiple pickle extraction
20180420 - AKC - Added error function in case you enter an incorrect category (Script will continue anyways!)
20180429 - AKC - Added functionality where you can add 'structure_file' and 'xtc_file' as an input. This way, you do not have to constantly switch files if one is more necessary than the other.
20180521 - AKC - Added functionality to load_multi_traj_pickle, which can now take specific working directories
20180529 - AKC - Added encoding of "Latin1" via pickle load to improve compatibility with Python 2
20180629 - AKC - Added "load_multi_traj_multi_analysis_pickle", which can load multiple analysis tools
20180702 - AKC - Added "find_class_from_list" function for analysis that uses multiple analysis tools
20180710 - AKC - Added print function for multi_traj_analysis to output the size of the class object before saving. This is useful for debugging when pickle does not work!
20180717 - AKC - Fixed up script for finding pickles --- prevents errors when one pickle folder does not have the information that you want. Alto fixed up multi traj analysis to accept specific directory for analysis.
20180726 - AKC - Fixed bug in specific directory script --- find_specific_paths function not correctly using lists!
20180814 - AKC - Added 'analysis_classes' as a key word for inputs -- enables the loading of pickles within analysis tools!!!
20180820 - AKC - Added 'find_function_name_from_list' and 'load_pickle_for_analysis' functions for loading pickles within analysis tools
20181213 - AKC - Updating bug for specific analysis 
"""
### IMPORTING MODULES
import MDDescriptors.core.initialize as initialize
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import glob # Used to find the directories
import os
import sys
import time
import pickle # Used to store variables    
from datetime import datetime
from MDDescriptors.core.initialize import checkPath2Server

### DEFINING LOCATION FOR STORING PICKLE
PICKLE_LOCATION="PICKLE"

###########################################################
### CLASS FUNCTION TO LOAD TRAJECTORIES AND RUN PROGRAM ###
###########################################################
class multi_traj_analysis:
    '''
    The purpose of this class is to take the input directories, files, and function to run detailed analysis on them. Afterwards, this function will output the analyzed data as a variable file.
    INPUTS:
        Main_Directory_Parent_Path: Main parent directory path
        Main_Directory: 
            Main directory of analysis
        Categories: [list, default=None]
            Categories within the main directories. If None, we will loop through all possible directories.
        Files: Files to look within those directories
        Descriptor_Class: List of classes to operate the trajectories
        Descriptor_Inputs: List of inputs that correspond to each class
            KEYWORDS FOR MULTI TRAJ ANALYSIS:
                'analysis_classes': use this when you want to insert previous analysis pickles into your current descriptor class
                'structure_file' / 'xtc_file': use this if you want to load a different trajectory for specific descriptors, rather than your usual
        Specific_dir_within_category: [dict, default=None] Specific category and directory that you want to run your analysis tool on. If None, we will look for all folders for all specified categories.
            Input example:
                    ### DEFINING FILES WITHIN MAIN DIRECTORY
                    Specific_dir_within_category = { 'spherical': [ 'spherical_310.15_K_6_nmDIAM_octanethiol_CHARMM36_Trial_3',
                                                                    'spherical_310.15_K_6_nmDIAM_hexadecanethiol_CHARMM36_Trial_2',
                                                                    'spherical_310.15_K_6_nmDIAM_hexadecanethiol_CHARMM36_Trial_3',
                                                                   ],     
                    }
            In this example, we are running spherical category and only the directories specified! This is highly useful for restarting analysis tools that failed in the middle, and only lacks a couple of data points
    
        want_only_directories: [logical, default = False] True if you do not want trajectory to be imported, but simply loaded via directory. This is useful for cases when you are extracting information via gmx commands, but do not need mdtraj
        date_suffix: [str]
            date suffix for analysis
    OUTPUTS:
        ### DIRECTORY INFORMATION
            self.Full_Path_Main_Directory: Path to main directory of the analysis
            self.Full_Path_Categories: Path to categories within main directory
            self.Full_List_Analysis_Directories: List of list of directories within the categories
            self.flatten_analysis_dir_list: Flattened out directory list
            self.num_trajectories: counts the number of trajectories (note, starts at zero)
        ### RESULTS
            self.results: Contains all information about the results in a form of a dictionary
        ### PICKLE RESULTS
            self.path_working_dir: current working directory path
            self.pickle_file_path: pickle file path
            self.path_pickle_file: Path to pickle file
            self.pickle_file_path_date: Path to pickle file and date
    FUNCTIONS:
        print_summary: Prints summary at the end with a summary file (summary.txt)
        get_pickle_info: Gets pickle information (i.e. where do we store our pickle?)            
    '''
    ### INITIALIZATION
    def __init__(self, Main_Directory_Parent_Path, Main_Directory, Files, Descriptor_Class, Descriptor_Inputs, Categories = None, 
                 Specific_dir_within_category = None, want_only_directories = False,
                 date_suffix=''): 
        ## COLLECTING INPUTS
        self.input_Main_Directory_Parent_Path = Main_Directory_Parent_Path
        self.input_Main_Directory = Main_Directory
        self.input_Categories = Categories
        self.input_Files = Files
        self.Descriptor_Class = Descriptor_Class
        self.date_suffix = date_suffix
        
        ## CHECKING TO SEE IF YOU EVEN HAVE A DESCRIPTOR CLASS
        if len(Descriptor_Class) == 0:
            print("Error! Descriptor_Class does not have any input classes. Perhaps there is something wrong with your input!")
            print("Stopping here! Please check your inputs!")
            sys.exit()
            
        ## CHECKING TO SEE IF THE LENGTH OF INPUTS AND DESCRIPTORS MATCH
        if len(Descriptor_Class) != len(Descriptor_Inputs):
            print("Your input descriptor class and inputs do not match!")
            print("Total descriptor classes: %d"%(len(Descriptor_Class) ) )
            print("Total descriptor inputs: %d"%(len(Descriptor_Inputs) ) )
            print("Please double-check your inputs")
        
        ## DEFINING FULL PATH OF ANALYSIS
        self.Full_Path_Main_Directory = checkPath2Server( self.input_Main_Directory_Parent_Path + '\\' + self.input_Main_Directory )
        
        ## SEEING IF WE NEED SPECIFIC DIRECTORIES
        if Specific_dir_within_category != None:
            print("Specific directory within category is specified! If this is not what you intended, set variable Specific_dir_within_category to None")
            print("Otherwise, we will look for the specific directory of your interest and run the analysis tool just for that one!")
            self.find_specific_paths(Specific_dir_within_category = Specific_dir_within_category)
        else:
            ## FINDING THE FULL PATH TO ANALYSIS DIRECTORIES
            self.find_full_path()
            
        ## FLATTENING ANALYSIS LIST
        self.flatten_analysis_dir_list = self.flatten_list_of_list(self.Full_List_Analysis_Directories)
        
        ## CREATING COUNTER FOR NUMBER OF TRAJECTORIES
        self.num_trajectories=0

        ## LOOPING THROUGH EACH CATEGORY AND RUNNING ANALYSIS TOOL
        for index_category, current_category in enumerate(self.input_Categories):
            ## DEFINING CURRENT FILE
            current_file_information = self.check_dict_for_categories(current_category)
            
            ## STORING CURRENT FOLDER INFORMATIONS
            current_structure  = current_file_information['structure_file']
            current_xtc = current_file_information['xtc_file']
            
            ## LOOPING THROUGH EACH ANALYSIS DIRECTORY WITHIN CATEGORY
            for current_directory in self.Full_List_Analysis_Directories[index_category]:
                ## GETTING CURRENT DIRECTORY NAME
                self.current_directory_basename = self.find_dir_name(current_directory)
                
                ## LOADING THE TRAJECTORIES
                traj_data = import_tools.import_traj(directory = current_directory, # Directory to analysis
                                                     structure_file = current_structure, # structure file
                                                     xtc_file = current_xtc, # trajectories
                                                     want_only_directories = want_only_directories, # True if you do not want trajectories to be loaded!
                                                     )
                
                ## RUNNING ANALYSIS TOOLS
                for index, Descriptor_Class in enumerate(self.Descriptor_Class):
                    ## DEFINING CURRENT DETAILS FOR INPUT
                    current_descriptor_input = Descriptor_Inputs[index][current_category]
                    
                    ## SEEING IF YOU HAVE MULTIPLE ANALYSIS TOOLS THAT YOU WANT TO LOAD
                    if 'analysis_classes' in current_descriptor_input.keys():
                        print("Since 'analysis_classes' is part of your set of keys, we are adding the 'pickle_loading_file' variable, which will allow you to load the pickle that you want")
                        ## ADDING THE PICKLE LOADING FILE AS PART OF THE DESCRIPTOR
                        current_descriptor_input['pickle_loading_file'] = self.current_directory_basename
                    
                    ## CHECKING IF THE DESCRIPTOR HAS SPECIFICATIONS FOR THE TRAJECTORIES
                    if 'structure_file' in current_descriptor_input.keys() or 'xtc_file' in current_descriptor_input.keys():
                        print("NOTE: Since 'structure_file' and 'xtc_file' is placed within descriptor class, %s, we are editing the inputs to the trajectory"%(Descriptor_Class.__name__))
                        ## COPYING INPUT INFORMATION
                        specific_descriptor_input = dict(current_descriptor_input)
                        ## NOW, I NEED TO LOAD A NEW STRUCTURE FILE OR XTC FILE (WHICHEVER IS PRESENT)
                        # CHECKING STRUCTURE FILE
                        if 'structure_file' in current_descriptor_input.keys():
                            ## STORING THE STRUCTURE AND REMOVING THE VARIABLE FROM INPUT
                            specific_structure_file=current_descriptor_input['structure_file']; del specific_descriptor_input['structure_file']
                        else:
                            specific_structure_file=current_structure
                        # CHECKING XTC FILE
                        if 'xtc_file' in current_descriptor_input.keys():
                            ## STORING THE STRUCTURE AND REMOVING THE VARIABLE FROM INPUT
                            specific_xtc_file=current_descriptor_input['xtc_file']; del specific_descriptor_input['xtc_file']
                        else:
                            specific_xtc_file=current_structure
                        ## PRINTING
                        print("structure_file: %s"%(specific_structure_file))
                        print("xtc_file: %s"%(specific_xtc_file))
                        ## RELOADING TRAJECTORY AND RUNNING DESCRIPTOR FOR THAT
                        specific_traj_data = import_tools.import_traj(directory = current_directory, # Directory to analysis
                                                             structure_file = specific_structure_file, # structure file
                                                             xtc_file = specific_xtc_file, # trajectories
                                                             want_only_directories = want_only_directories, # True if you do not want trajectories to be loaded!
                                                             )
                        ## RUNNING SPECIFIC DESCRIPTOR RESULTS
                        Descriptor_Results = Descriptor_Class(traj_data = specific_traj_data, **specific_descriptor_input)
                        
                    else:
                        ## USING DESCRIPTOR CLASS WITH ITS INPUT
                        Descriptor_Results = Descriptor_Class(traj_data = traj_data, **current_descriptor_input)
                    #########################
                    #### STORING PROCESS ####
                    #########################
                    ## GETTING PICKLE INFORMATION
                    self.get_pickle_info(Descriptor_Class)
                    ## STORING INTO RESULTS
                    self.results={'PATH': self.flatten_analysis_dir_list[self.num_trajectories],
                                  'DIRECTORY_NAME': self.current_directory_basename,
                                  'RESULTS': Descriptor_Results,
                                          }
                    ## STORING INFORMATION
                    self.store_pickle()
                ## ADDING TO TRAJECTORY NUMBER
                self.num_trajectories += 1
                

        ## PRINTING SUMMARY
        self.print_summary()
        
    ### FUNCTION TO PRINT SUMMARY (OUTPUTS summary.txt)
    def print_summary(self):
        '''This function prints summary of what was done for this class'''
        print("-----------------------------------")
        print("------- multi_traj_analysis  ------")
        print("-----------------------------------")
        print("NUMBER OF TRAJECTORIES: %s"%(self.num_trajectories))
        print("DIRECTORY PATH: %s"%(self.input_Main_Directory_Parent_Path))
        print("CATEGORIES: %s"%(', '.join(self.input_Categories)))
        print("CHARACTERIZATION NAME(s): %s"%( ', '.join([ Descriptor_class.__name__ for Descriptor_class in self.Descriptor_Class ]) ))
        print("PICKLE LOCATION: %s"%(self.pickle_file_path_date))
        ''' summary text file outputted is turned off tentatively. 
        with open(self.pickle_file_path_date + '/summary.txt', 'w') as f:
            f.write("NUMBER OF TRAJECTORIES: %s\n"%(self.num_trajectories))
            f.write("DIRECTORY PATH: %s\n"%(self.input_Main_Directory_Parent_Path))
            f.write("CATEGORIES: %s\n"%(', '.join(self.input_Categories)))
            f.write("CHARACTERIZATION NAME: %s\n"%(self.Descriptor_Class.__name__ ))
            f.write("PICKLE LOCATION: %s\n"%(self.pickle_file_path_date))
        '''

    ### FUNCTION TO STORE ALL INFORMATION WITHIN A PICKLE FILE
    def store_pickle(self):
        '''
        The purpose of this function is to store all the results into a pickle file
        INPUTS:
            self: class property
        OUTPUTS:
            pickle file within the pickle directory under the name of the class used
        '''
        ## PRINTING TOTAL SIZE OF THE OBJECT
        # print("Total size of saving pickle is: %d bytes or %d MB"%(sys.getsizeof(self), sys.getsizeof(self) / 1000000.0 ))
        ## DEPRECIATED, see size by: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
        with open(self.pickle_file_path_date + '/' + self.current_directory_basename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self], f, protocol=2)  # <-- protocol 2 required for python2   # -1
        print("Data collection was complete at: %s\n"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    ### FUNCTION TO GET STORAGE INFORMATION FOR PICKLE
    def get_pickle_info(self, Descriptor_Class):
        '''
        The purpose of this script is to get your script name and output paths for storage for your variables.
        Inputs:
            Descriptor_Class: Class for your descriptor
        Outputs:
            self.path_working_dir: current working directory path
            self.pickle_file_path: pickle file path
            self.path_pickle_file: Path to pickle file
            self.pickle_file_path_date: Path to pickle file and date
        '''
        ## DEFINING CURRENT WORKING DIRECTORY
        self.path_working_dir = os.getcwd()
        ## DEFINING PATH TO PICKLE DIRECTORY
        self.pickle_file_path = PICKLE_LOCATION + '/' + Descriptor_Class.__name__ 
        ## DEFINING NAME OF STORAGE VARIABLE
        Date=time.strftime("%y%m%d") # Finding date
        ## DEFINING PATH TO PICKLE DIRECTORY WITH THE DATE
        self.pickle_file_path_date = self.pickle_file_path + '/' + Date + self.date_suffix
        ## CHECKING IF DIRECTORY EXISTS
        initialize.check_dir(self.pickle_file_path_date)
        return
        
    ### FUNCTION TO FIND FULL PATHS TO ANALYSIS DIRECTORIES
    def find_full_path(self):
        ''' 
        Function to find all full paths to analysis directories
        INPUTS:
            self: class property
            self.Full_Path_Main_Directory: Path to main directory of the analysis
        OUTPUTS:
            self.Full_Path_Categories: Path to categories within main directory
            self.Full_List_Analysis_Directories: List of list of directories within the categories
        '''
        ## FINDING LIST OF DIRECTORIES WITHIN THE PATHS
        self.Full_Path_Categories = [ self.Full_Path_Main_Directory +'/'+ x for x in self.input_Categories]
        ## USING GLOB TO FIND EACH DIRECTORY AND SORTING
        self.Full_List_Analysis_Directories=[ sorted(glob.glob( x + '/*')) for x in self.Full_Path_Categories ]
        ## CHECKING IF THE CATEGORY EXISTS
        for each_category in self.Full_Path_Categories:
            if os.path.isdir(each_category) is not True:
                print("WARNING!!!")
                print("The following category does not exist: %s"%(each_category) )
                print("Continuing...... this may cause errors!")
                print("Pausing 5 seconds so you can see the error")
                time.sleep(5)
        return
    
    ### FUNCTION TO FIND FULL PATHS FOR SPECIFIC ANALYSIS DIRECTORIES
    def find_specific_paths(self, Specific_dir_within_category):
        '''
        The purpose of this function is to find specific paths based on the full path scripts. It will simply limit the number of analyssi directories available, 
        INPUTS:
            Specific_dir_within_category: [dict] dictionary of a specific category with a list of directories
        OUTPUTS:
            self.Full_Path_Categories: Path to categories within main directory
            self.Full_List_Analysis_Directories: List of list of directories within the categories
        '''
        ## FINDING ALL CATEGORIES
        category_list = list(Specific_dir_within_category.keys())
        ## UPDATING CATEGORY LIST
        self.input_Categories = category_list[:]
        ## LOOPING THROUGH CATEGORY LIST AND GETTING PATH TO CATEGORIES
        self.Full_Path_Categories = [ self.Full_Path_Main_Directory +'/'+ x for x in category_list]
        ## USING GLOB TO FIND EACH DIRECTORY AND SORTING
        self.Full_List_Analysis_Directories=[ [ sorted(glob.glob( x + '/' + y ))[0] for y in Specific_dir_within_category[category_list[idx]] ]\
                                             for idx, x in enumerate(self.Full_Path_Categories)
                                                                              ]
        
#        print(self.Full_List_Analysis_Directories)
#        print(self.Full_Path_Categories)
#        print(self.input_Categories)
#        time.sleep(10)
        return
    
    ### FUNCTION TO LOOK INTO FILES AND CHECK TO MAKE SURE ALL INPUT FILES ARE SINGLE
    def check_dict_for_categories(self,category_item):
        '''
        The purpose of this function is to look into the dictionary and ensure that there is only one entry for each file type
        INPUTS:
            self: class property
            category_item: Item that you are interested in (This will look through the dictionary and try to comb that key word)
        OUTPUTS: 
            checked_dict: Item that has been checked
        '''
        checked_dict = {}
        ### LOOPING THROUGH EACH KEY
        for each_key in self.input_Files.keys():
            ## SEE IF THE KEY IS NOT A DICTIONARY
            if type(self.input_Files[each_key]) is not dict: ## THIS MEANS THAT THE CURRENT INPUT FILE IS INDEED A STRING
                checked_dict[each_key] = self.input_Files[each_key]
            else: ## THIS MEANS THAT WE HAVE A DICTIONARY ITEM THAT NEEDS TO BE ATTACHED ONTO A CATEGORY
                checked_dict[each_key] = self.input_Files[each_key][category_item]
        return checked_dict
    
    ### FUNCTION TO FLATTEN OUT LIST
    @staticmethod
    def flatten_list_of_list(list_of_lists):
        '''
        The purpose of this function is to simply flatten the nested lists
        INPUTS:
            list_of_lists: List of lists (e.g. [ [a, b],[c]])
        OUTPUTS:
            flatten_list: flatten list (e.g. [a, b, c, ...])
        '''
        flatten_list = [item for items in list_of_lists for item in items]
        return flatten_list
    
    ### FUNCTION TO FIND THE DIRECTORY NAME GIVEN THE PATH
    @staticmethod
    def find_dir_name( directory ):
        '''
        The purpose of this function is to find the directory name given the path.
        INPUTS:
            directory: [string] full path to the directory
                e.g. 'R:\\scratch\\SideProjectHuber\\Analysis\\\\\\180302-Spatial_Mapping/TBA\\TBA_50_GVL'
        OUTPUTS:
            baseline_directory: [string] name of the directory you are at (i.e. last directory without parent)
                e.g. 'TBA_50_GVL'
        '''
        print("Finding basename for %s"%(directory))
        try:
            baseline_index=directory.rindex('\\')
        except:
            baseline_index=directory.rindex('/')
        baseline_directory= directory[baseline_index+1:]
        return baseline_directory


### FUNCTION TO FIND ALL PICKLES AS A LIST
def find_multi_traj_pickle(Date, Descriptor_class, turn_off_multiple = False):
    '''
    The purpose of this script is to find all the pickles given the class and the date
    INPUTS:
        Date: [str or list] Desired date as a string
        Descriptor_class: [class or list] Class that you used to run the multi traj function
    NOTE: If these inputs are lists, we will find all the pickles and check to see if the pickle exists in both classes
    OUTPUTS:
        list_of_pickles: Pickle files as a list (ignoring summary.txt)
            - list of pickle changes if you have multiple classes as a list. The code will try to match the pickles with the same names.
            - those missing a pickle will be omitted from the list of pickles. This is to prevent errors when running this script.
    '''
    import os
    ## FINDING CURRENT WORKING DIRECTORY
    current_work_dir=os.getcwd()
    if type(Descriptor_class) == type or turn_off_multiple == True:
        Descriptor_class = [Descriptor_class]
        Date = [Date]
        
    ## DEFINING EMPTY LIST OF PICKLES
    list_of_pickles = []
    for idx, each_descriptor_class in enumerate(Descriptor_class):
        ## DEFINING PICKLE DIRECTORY
        Pickle_directory = checkPath2Server(current_work_dir + '\\' + PICKLE_LOCATION + '\\' + each_descriptor_class.__name__ + '\\' + Date[idx])
        ## USING GLOB TO VIEW ALL DIRECTORIES
        current_list_of_pickles=[ os.path.basename(filename) for filename in sorted(glob.glob( Pickle_directory + '/*')) ]
        ## REMOVING SUMMARY .TXT
        current_list_of_pickles = set([ each_pickle for each_pickle in current_list_of_pickles if each_pickle != "summary.txt" ]) # GENERATING A SET
        ## EXTENDING THE LIST
        list_of_pickles.append(current_list_of_pickles)
        
    ## REMOVING DUPLICATE PICKLES AND FINDING INTERSECTIONS
    list_of_pickles = list(set.intersection(*list_of_pickles))
    # list_of_pickles = list(set(list_of_pickles))
    
    return list_of_pickles
    
### FUNCTION TO LOAD MULTI TRAJ PICKLES
def load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file_name, current_work_dir = None ):
    '''
    The purpose of this script is to simply load the pickle after you have saved the results using multi_traj class
    INPUTS:
        Date: Desired date as a string
        Descriptor_class: Class that you used to run the multi traj function
        Pickle_loading_file_name: Name of the pickle file you are interested in
        current_work_dir: [str, default=None] current working directory you want to load your pickle from
    OUTPUTS:
        multi_traj_results: multi traj for that pickle
    '''
    ## FINDING CURRENT WORKING DIRECTORY
    if current_work_dir is None:
        current_work_dir=os.getcwd()
    ## DEFINING PICKLE DIRECTORY
    Pickle_directory = os.path.join(current_work_dir,
                                    PICKLE_LOCATION,
                                    Descriptor_class.__name__,
                                    Date,
                                    Pickle_loading_file_name)
    # PRINTING
    print("LOADING PICKLE FROM: %s"%(Pickle_directory) )    
    ## LOADING THE DATA
    with open(Pickle_directory,'rb') as f:
        # multi_traj_results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        if sys.version_info[0] > 2:
            multi_traj_results = pickle.load(f, encoding='latin1') ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        elif sys.version_info[0] == 2: ## ENCODING IS NOT AVAILABLE IN PYTHON 2
            multi_traj_results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        else:
            print("ERROR! Your python version is not 2 or greater! We cannot load pickle files below Python 2!")
            sys.exit()
        ## SEE REFERENCE: https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    ### LOADING ONLY THE RESULTS DATA
    multi_traj_results = multi_traj_results[0].results['RESULTS']
    
    return multi_traj_results

### FUNCTION TO LOAD MULTIPLE ANALYSIS TOOLS
def load_multi_traj_multi_analysis_pickle( Dates, Descriptor_classes, Pickle_loading_file_names, current_work_dir = None  ):
    '''
    The purpose of this function is to load multiple analysis tools based on pickles with the same name.
    This function is useful if you have more than one pickle that you want to extract via a list.
    Therefore, this tool is important for mixture of analysis tools.
    INPUTS:
        Dates: [list] list of strings for the dates
        Descriptor_classes: [list] list of multiple descriptor clsses
        Pickle_loading_file_names: [list or string] name of the pickle
            If list, then we will load the pickle file name based with respect to dates and descriptor class
            If string, we will use this file name throughout all loading
        current_work_dir: [str, default=None] current working directory you want to load your pickle from
    OUTPUTS:
        multi_traj_results_list: [list] list of multi traj results
    '''
    ## DEFINING EMPTY LIST
    multi_traj_results_list = []
    
    ## CHECKING THE LIST TO ENSURE THAT THE LENGTHS ARE THE SAME
    if len(Dates) != len(Descriptor_classes):
        print("Error! List of Dates(%d) and Descriptor class(%d) is not matching! Check your inputs!"%(len(Dates), len(Descriptor_classes)) )
        
    ## SEEING THE PICKLE LOADING FILE
    if type(Pickle_loading_file_names) == str:
        ## CONVERTING TO A LIST
        Pickle_loading_file_names = [Pickle_loading_file_names] * len(Dates)
        
    ## LOOPING THROUGH MULIPLE CLASSES
    for idx, each_class in enumerate(Descriptor_classes):
        ## FINDING THE DATE
        current_date = Dates[idx]
        ## USING SINGLE LOADING OF PICKLES
        multi_traj_results = load_multi_traj_pickle(Date = current_date,
                                   Descriptor_class = each_class,
                                   Pickle_loading_file_name = Pickle_loading_file_names[idx],
                                   )
        ## STORING MULTIPLE TRAJECTORY RESULTS
        multi_traj_results_list.append(multi_traj_results)
        
    return multi_traj_results_list

### FUNCTION TO FIND A CLASS FROM A LIST
def find_class_from_list(class_list, class_name):
    '''
    The purpose of this function is to find a class from a list
    INPUTS:
        class_list: [list] list of classes
        class_name: [str] name of the class you are interested in
    OUTPUTS:
        class_of_interest: [class] class object with the name you are interested in
    '''
    ## DO A LOOP TO TRY TO FIND THE CLASS NAME
    class_of_interest = [ each_class for each_class in class_list if each_class.__class__.__name__ == class_name][0]
    return class_of_interest



### FUNCTION TO LOAD MULTIPLE ANALYSIS FROM MULTIPLE PICKLES
def load_multi_traj_pickles(Date, Descriptor_class, turn_off_multiple = False):
    '''
    This script loads multiple pickle files
    INPUTS:
        Date: Desired date as a string
        Descriptor_class: Class that you used to run the multi traj function
    OUTPUTS:
        traj_results: list of class results
        list_of_pickles: list of pickles
    '''
    ## FINDING LIST OF PICKLES BASED ON THE DATE AND DESCRIPTOR CLASS
    list_of_pickles = find_multi_traj_pickle(Date, Descriptor_class, turn_off_multiple = turn_off_multiple)
    ## CREATING EMPTY RESULTS LIST
    traj_results=[]
    ## LOOPING THROUGH EACH DIRECCTORY AND LOADING THE ANALYSIS
    for Pickle_loading_file in list_of_pickles:
        ### EXTRACTING THE DATA
        multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
        ### STORING THE RESULTS
        traj_results.append(multi_traj_results)
    return traj_results, list_of_pickles

### FUNCTION TO ADD TO A LIST OF DICTS
def add_to_all_dicts_of_dicts(dicts_of_dicts, input_key, input_value):
    '''
    The purpose of this script is to look into a list of dictionaries and add to each entry
    INPUTS:
        dicts_of_dicts: list of dictionary items, e.g. {{'ACE':  },{'PRO': }},
        input_key: dictionary key (NOTE, MUST BE A STRING!)
        input_value: value you want that key to be assigned
    OUTPUTS:
        dicts_of_dicts: Updated dictionary 
    '''
    for each_key in dicts_of_dicts.keys():
        dicts_of_dicts[each_key][input_key] = input_value    
    return dicts_of_dicts

### FUNCTION TO FIND A FUNCTION FROM A LIST
def find_function_name_from_list(function_list, class_name):
    '''
    The purpose of this function is to find a class from a list
    INPUTS:
        function_list: [list] list of classes/functions
        class_name: [str] name of the class you are interested in
    OUTPUTS:
        class_of_interest: [class] class/function object with the name you are interested in
    '''
    ## DO A LOOP TO TRY TO FIND THE CLASS NAME
    class_of_interest = [ each_function for each_function in function_list if each_function.__name__ == class_name][0]
    return class_of_interest

### FUNCTION TO LOAD PICKLE BASED ON ANALYSIS CLASSES AND STRING
def load_pickle_for_analysis( analysis_classes, function_name, pickle_file_name, conversion_type = None, current_work_dir = None):
    '''
    The purpose of this function is to re-load a pickle for subsequent analysis. We will use the data for this pickle to compute new information, then save the entire thing under multi_traj analysis tools
    It is expected that this function is run several times to load specific information from a pickle
    INPUTS:
        analysis_classes: [list of list]
            list of list of analysis classes, e.g. [[ self_assembly_structure, '180814-FINAL' ], ...],
            Note: the first entry is the class function, the second entry is the date/location for loading the pickle.
        function_name: [function]
            name of the function you want to load.
        pickle_file_name: [str]
            string of the pickle file name that you want to load. This should be located within function_name > date > pickle_file_name
        current_work_dir: [str]
            path to directory that has the pickle folder
    OUTPUTS:
        multi_traj_results: [class]
            analysis tool based on your inputs
    '''
    ## IMPORTING DECODING FUNCTION
    from MDDescriptors.core.decoder import convert_pickle_file_names
    ## FINDING FUNCTION FROM THE LIST
    specific_analysis_class = [each_analysis for each_analysis in analysis_classes if each_analysis[0].__name__ == function_name ][0]
    
    ## CONVERSION OF PICKLE FILE NAME IF NEEDED
    updated_pickle_file_name = convert_pickle_file_names( pickle_file_name = pickle_file_name,
                                                          conversion_type = conversion_type,
                                                         )
    ## RUNNING MULTI TRAJ PICKLING
    multi_traj_results = load_multi_traj_pickle(specific_analysis_class[1], specific_analysis_class[0], updated_pickle_file_name, current_work_dir = current_work_dir )
    
    return multi_traj_results

