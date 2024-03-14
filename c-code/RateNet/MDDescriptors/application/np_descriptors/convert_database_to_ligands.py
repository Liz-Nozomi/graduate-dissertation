#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_database_to_ligands.py

The purpose of this script is to load a database, then analyze the ligand 
SMILES. If the ligand is within the database, then we will just output the 
ligand name. Otherwise, we will need to generate a mol2 file for the ligand.

Written by: Alex K. Chew (08/26/2020)

"""
import os
import numpy as np
import pandas as pd
import math


## IMPORTING RDKIT
from rdkit import Chem
from rdkit.Chem import Draw

## IMPORTING CUSTOM
from MDDescriptors.application.np_descriptors.global_vars import PATH_DATABASE

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING EXTENSION
FIG_EXTENSION = "png"
SAVE_FIG = True

## DEFINING SUBSTRUCTURE MATCHING
SUBSTRUCTURE_DICT={
        'C1CCSS1': {
                'substructure': "S1CCCS1"
                }
        }
        
## DEFINING DEFAULT
DEFAULT_SUBSTRUCTURE = "SCCCC"        

### FUNCTION TO CHECK SUBSTRUCTURE MATCH
def check_matching_substructure(smile_string = "O=C(NCCOCCO)CCCCC1CCSS1",
                                substructure_string="S1CCCS1"):
    '''
    This function checks substructures and see if there is a match. 
    This is useful when you are trying to find substructures and 
    change the assembly ligand type.
    INPUTS:
        smile_string: [str]
            string smiles that you are checking
        substructure_string: [str]
            substructure string that you are basing it on
    OUTPUTS:
        logical: [true/false]
            True if substructure is wthin smile_string
    '''
    ## GETING SMILES
    m = Chem.MolFromSmiles(smile_string)
    
    ## GETTING PATTERN
    patt = Chem.MolFromSmarts(substructure_string)
    return m.HasSubstructMatch(patt)

#####################################################
### CLASS OBJECT TO CREATE DATABASE OF LIG SMILES ###
#####################################################
class create_dataframe_of_lig_SMILES:
    '''
    The purpose of this class is to input a dataframe and get back 
    a new dataframe with just a unique list of ligands. The idea would 
    be to quickly extract GNP properties (e.g. ligand numbers, etc.), then 
    output a dataframe. If the dataframe exists, then we should 
    review the ligands, then input into the database.
    INPUTS:
         database_path: [str]
             path to the input database
    
    FUNCTIONS:
        find_unique_ligands: 
            finds unique ligands from a dataframe
        make_lig_dataframe:
            makes a ligand dataframe using the unique ligand names
    '''
    def __init__(self,
                 database_path):
        ## LOADING DATABASE
        self.database_df = pd.read_csv(database_path)
        
        ## GETTING UNIQUE LIGANDS
        self.unique_ligands, self.ligand_column_labels = self.find_unique_ligands(self.database_df)
        
        return
    
    ### FUNCTION TO FIND UNIQUE LIGANDS
    @staticmethod
    def find_unique_ligands(database_df,
                            ):
        '''
        This function finds all unique ligands
        INPUTS:
            database_df: [df]
                dataframe containing ligands. We will search for "SMILES" for ligands
        OUTPUTS:
            unique_ligands: [np.array]
                unique ligands
            ligand_column_labels: [list]
                list of column string labels
        '''

        ## GETTING ALL UNIQUE LIGANDS
        ligand_column_labels = [each_col for each_col in database_df.columns if "SMILES" in each_col ]
        
        ## GETTING ALL LIGAND NAMES
        ligand_smiles_df = database_df[ligand_column_labels]
        
        ## GETTING ALL UNIQUE LIGANDS
        unique_ligands = ligand_smiles_df.T.stack().unique()
        '''
        This code transposes, then collapes all columns to a Series object. We 
        then use the unique function to eliminate repeats. The command below 
        simply removes all hyphen examples.
        '''
        ## REMOVING ANY HYPHENS
        unique_ligands = unique_ligands[unique_ligands != '-']
        
        ## PRINTING
        print("Number of GNPs: %d"%(len(database_df) ))
        print("Number of unique ligands: %d"%(len(unique_ligands) ))
        
        return unique_ligands, ligand_column_labels
    
    ### FUNCTION TO 
    
    ### FUNCTION TO OUTPUT DATAFRAME
    @staticmethod
    def make_lig_dataframe(unique_ligands,
                           lig_prefix = "LIG",
                           lig_resname_prefix = "L",
                           want_substructures = True,
                           substructure_dict = SUBSTRUCTURE_DICT):
        '''
        This function creates a ligand dataframe based on the input 
        unique ligands
        INPUTS:
            unique_ligands: [np.arra]
                array of unique ligand labels
            lig_prefix: [str]
                ligand prefix you want in front of each name
            lig_resname_prefix: [str]
                prefix for residue name
            want_substructures: [logical]
                True if you want substructures
        OUTPUTS:
            output_df: [dataframe]
                dataframe containing ligand information
        '''
        ## CREATING DATAFRAME
        output_df = []
        
        ## GETTING LIST OF SUBSTRUCTURES
        if want_substructures is True:
            substructure_list = list(substructure_dict.keys())
        
        ## CREATING A DATABASE
        for idx, each_lig in enumerate(unique_ligands):
            
            ## CREATING WARNING
            if idx > 999:
                print("Warning! Index greater than 999, we have only so many residue names!")
                print("Consider changing residue nomenclature!")
            
            ## CHECKING SUBSTRUCTURE
            if want_substructures is True:
                ## CREATING LIST
                list_of_substructures = []
                for each_sub in substructure_list:
                    
                    ## GETTING SUBSTRUCTURE STRING
                    substructure_string = substructure_dict[each_sub]['substructure']
                    ## CHECKING SUBSTRUCTURE
                    logical_substructure = check_matching_substructure(smile_string = each_lig,
                                                                       substructure_string=substructure_string)
                    
                    if logical_substructure is True:
                        list_of_substructures.append(each_sub)
                
                ## CHECKING IF NO SUBSTRUCTURES FOUND
                if len(list_of_substructures) == 0:
                    ## APPEND DEFAULT
                    list_of_substructures.append(DEFAULT_SUBSTRUCTURE)
                    
                ## CHECKING IF LEN GREATER THAN ONE
                if len(list_of_substructures) > 1:
                    print("Warning! Substructure is found multiple times for %s"%(each_lig))
                    print(list_of_substructures)
                    print("Not sure how to handle multiple substructure searches at this time")
            
                ## COMBINING SUBSTRUCTURE
                sub_structure_string = "-".join(list_of_substructures)
            
            
            ## CREATING DICTIONARY
            lig_dict = {
                    'name': lig_prefix + "%d"%(idx),
                    'resname': lig_resname_prefix + str(f'{idx:03d}'),
                    'SMILES': each_lig,
                    'substructure': sub_structure_string,
                    }
            ## STORING
            output_df.append(lig_dict)
        
        ## CREATING DATAFRAME
        output_df = pd.DataFrame(output_df)
        
        return output_df

### FUNCTION TO CREATE DICT
def generate_dict_lig_name_to_num(lig_smiles_labels,
                                  num_lig_labels):
    '''
    This function generates ligand name to number dictionary. This is 
    useful for dataframes that have things like "Ligand1 SMILES" and a 
    corresponding "#Ligand1". 
    INPUTS:
        lig_smiles_labels: [list]
            list of labels for SMILES
        num_lig_labels: [list]
            list of labels for #Ligands
    OUTPUTS:
        lig_name_to_num_dict: [dict]
            dictionary containing keys, e.g. 
                {'Ligand1 SMILES': '#Ligand1',
                 'Ligand2 SMILES': '#Ligand2',
                 'Ligand3 SMILES': '#Ligand3',
                 'Ligand4 SMILES': '#Ligand4'}
    '''
    ## DEFINING DICT
    lig_name_to_num_dict = {}
    ## CREATING LIGAND DICTIONARY
    for each_lig in lig_smiles_labels:
        if each_lig.endswith('SMILES'):
            string_no_space_and_smiles = each_lig[:-len(" SMILES")]
        else:
            string_no_space_and_smiles = each_lig
        
        ## GETTING INDEX THAT MATCHED
        index_matching = [ num_lig_idx for num_lig_idx, each_num_lig in enumerate(num_lig_labels) if string_no_space_and_smiles in each_num_lig][0]
        ## CREATING DICTIONARY
        lig_name_to_num_dict[each_lig] = num_lig_labels[index_matching]
    return lig_name_to_num_dict

### FUNCTION TO CREATE LIGAND DATABASE GIVEN GNP DATABASE
def create_lig_and_gnp_database_from_input(database_path,
                                           want_output_csv = False):
    '''
    This function creates ligand and gold nanoparticle database based 
    on the input of the database path. 
    INPUTS:
        database_path: [str]
            path to the database
        want_output_csv: [logical]
            True if you want to output the csv file
    OUTPUTS:
        output_df: [dataframe]
            ligand data frame
        storage_for_gnps_df: [dataframe]
            dataframe for gold nanoparticles
    '''
    ## CREATING OBJECT
    lig_smiles = create_dataframe_of_lig_SMILES(database_path = database_path)
    
    ## CREATING UNIQUE DATAFRAME
    output_df = lig_smiles.make_lig_dataframe(lig_smiles.unique_ligands,
                                              lig_prefix = "LIG",
                                              lig_resname_prefix = "L",
                                              want_substructures = True)
    
    ## GETTING BASENAME
    database_name = os.path.basename(database_path)
    path_to_database = os.path.dirname(database_path)
    
    ## GETTING FILE NAME AND EXTENSION
    database_file_name, database_file_ext = os.path.splitext(database_name)
    
    ## GETTING NEW FILE
    new_database_name = database_file_name + "_LIGANDS" + database_file_ext
    
    ## DEFINING PATH
    new_database_path = os.path.join(path_to_database,
                                     new_database_name)
    
    ## OUTPUTTING CSV
    if want_output_csv is True:
        ## OUTPUTTING TO CSV
        output_df.to_csv(new_database_path, index = False)


    ## GETING GNP FILE
    gnp_database_name =  database_file_name + "_GNP" + database_file_ext
    path_gnp_database = os.path.join(path_to_database,
                                     gnp_database_name)
    
    
    
    ## GETTING ALL UNIQUE LIGANDS
    num_lig_column_labels = [each_col for each_col in lig_smiles.database_df.columns if each_col.startswith('#') ]
    

    ## GENERATING DICTIONARY FROM LIG NAME TO NUMBER
    lig_name_to_num_dict = generate_dict_lig_name_to_num(lig_smiles_labels = lig_smiles.ligand_column_labels,
                                                         num_lig_labels = num_lig_column_labels)
                    
    ## DEFINING STORAGE
    storage_for_gnps = []
    
    ## LOOPING THROUGH EACH INDICES
    for idx, row in lig_smiles.database_df.iterrows():
        ## GETTING EMPTY
        gnp_stored = []
        ## GETTING ALL LIGANDS
        name = row['Index']
        
        ## GETTING SIZE
        size = row['Size']
        ## APPENDING
        gnp_stored.extend([name, size])
        ## LOOPING THROUGH EACH AND STORING
        for each_lig in lig_smiles.ligand_column_labels:
            ## GETTING LIGAND
            ligand_smiles = row[each_lig]
            
            ## OPERATE ONLY IF SMILES IS NOT '-'
            if ligand_smiles != '-':
                ## GETTING UNIQUE LIGAND
                current_lig_name = output_df['name'][output_df['SMILES'] == ligand_smiles].iloc[0]
                ## GETTING NUMBER
                num_key = lig_name_to_num_dict[each_lig]
                ## GETTING NUMBER
                num_lig = int(row[num_key])
                ## APPENDING
                gnp_stored.extend([current_lig_name, num_lig])
            
        ## APPENDING
        storage_for_gnps.append(gnp_stored)
    
    ## GETTING DATAFRAME
    storage_for_gnps_df = pd.DataFrame(storage_for_gnps,
                                       )
    
    ## GETTING COLUMN LIST
    columns = ['Index', 'Size']
    [ columns.extend(["Ligand%d"%(idx), "#Ligand%d"%(idx)] ) for idx in range( int( (len(storage_for_gnps_df.columns) - 2) /2.0 )) ]
                      
    ## RENAMING COLUMNS
    storage_for_gnps_df.columns = columns
    
    ## OUTPUTTING CSV
    if want_output_csv is True:
    
        ## PRINTING TO CSV
        storage_for_gnps_df.to_csv(path_gnp_database, index = False)

    return output_df, storage_for_gnps_df



#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    
    ## DEFINING PATH
    path_to_database = r"/Volumes/akchew/scratch/MDLigands/database_ligands"
    # PATH_DATABASE
    
    ## DEFINING DATABASE NAME
    database_name = "logP_exp_data.csv"
    database_path = os.path.join(path_to_database, database_name )
    
    ## DEFINING IF YOU WANT CSV
    want_output_csv = False
        
    ## CREATING DATAFRAMES FOR LIG AND GNP
    output_df, storage_for_gnps_df = create_lig_and_gnp_database_from_input(database_path = database_path,
                                                                            want_output_csv = False)
    
    #%% CONVERTING LIGAND TO MOL2FILE
    
    '''
    # Installing pybel
    source activate py36_mdd
    pip install pybel --> Incorrect!
    
    Installing openbabel
    pip install openbabel -- > Does not correctly install
    
    Below correctly installs openbabel:
    conda install -n py36_mdd openbabel -c conda-forge
    
    Reference: https://openbabel.org/docs/dev/UseTheLibrary/PythonDoc.html
    
    Installing openeye tools
    conda install -n py36_mdd -c openeye openeye-toolkits
    
    '''
    
    #%%
    import openbabel
    smiles = ['CCCC', 'CCCN']
    mols = [openbabel.readstring("smi", x) for x in smiles] # Create a list of two molecules
    
    
    #%%
    
    from openbabel import pybel
    from openbabel import openbabel as ob
    smiles = 'CCCC'
    mol = pybel.readstring("smi", smiles)
    
    ## GETTING ATOMS
    print(len(mol.atoms))
    
    ## ADDING HYDROGENS
    mol.OBMol.AddHydrogens()
    
    print(len(mol.atoms))
    
    ff = ob.OBForceField.FindForceField("mmff94")
    ff.Setup(mol)
    
    #%%
    
    
    mol.write("mol2", "outputfile.mol2")
    
    
    
    #%%
    
    ## LOADING OPENBABEL
    from openbabel import openbabel

    ## LOOPING THROUGH EACH ROW    
    for idx, row in output_df.iterrows():
        if idx == 0:
            ## DEFINING NAME
            lig_name = row['name']
            ## GETTING SMILES
            smiles_string = row['SMILES']
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("smi", "mol2")
            
            ## CREATING MOLECULE
            mol = openbabel.OBMol()
            obConversion.ReadString(mol, smiles_string)
            
            ## PRINTING
            print("Number of atoms: %d"%(mol.NumAtoms()))
            
            ## ADDING HYDROGENS
            mol.AddHydrogens()
            print("Number of atoms with hydrogens: %d"%(mol.NumAtoms()))
#            gen3d = openbabel.OBOp.FindType("Gen3D")
#            gen3d.Do(mol) 
            
            ## GETTING PATH
            path_to_output = os.path.join(path_to_database, lig_name + ".mol2")
            ## WRITING
            obConversion.WriteFile(mol, path_to_output)
            
        
        
    #%%
    
    ## USING OPENEYE
    from openeye import oechem
    
    smiles = 'CCO'
    
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(smiles)
    
    #%%
    
    mols = []
    mol = oechem.OEMol()
    for mol in ims.GetOEMols():
        mols.append(oechem.OEMol(mol))
        
    #%%
        
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_MOL2H)
    oms.openstring()
    
    
    #%%
    for mol in mols:
        oechem.OEWriteMolecule(oms, mol)

    
    
    #%%
    ifs = oechem.oemolistream()
    
    ## SETTING FORMAT
    ifs.SetFormat(oechem.OEFormat_USM)
    
    ## OPENING
    ifs.open(smiles_string)
    
    #%%
    ofs = oechem.oemolostream()
    
    mol = oechem.OEGraphMol()
    
    while oechem.OEReadMolecule(ifs, mol):
        oechem.OEWriteMolecule(ofs, mol)
        
    #%%
            
            
            
    '''
            ## USING RDKIT TO LOAD MOLECULE
            m = Chem.MolFromSmiles(smiles_string)
            ## ADDING HYDROGENS
            m_with_hydrogens = Chem.AddHs(m)
            ## GETTING PATH
            path_to_output = os.path.join(path_to_database, lig_name + ".pdb")
            ## OUTPUTTING
            Chem.MolToPDBFile(m_with_hydrogens,path_to_output)
    '''
    
    
    