# -*- coding: utf-8 -*-
"""
accessible_hydroxyl_fraction.py
The purpose of this script is to calculate the accessible_hydroxyl_fraction value for a particular molecule. The delta parameter is a measure of hydrophilicity, where we take the ratio of hydroxyl groups to the total molecule.
INPUTS:
    trajectory: traj from md.traj
    input_details: input arguments, should be a dictionary
        'Solute': Solute name (single) -- will be used to calculate the hydroxyl fraction
        'SASA_type': Type of sasa you want. e.g.
            'oxy2carbon' - oxygen to carbon
            'oxy2carbon_OH_only' - oxygen alcohol to carbon ratio
            'alcohol2allSASA' - alcohol oxygen + hydrogen divided by entire molecule SASA
        'probe_radius': probe radius in nm of the rolling ball [OPTIONAL, default=0.14 ]
        'num_sphere_pts': number of points to represent your sphere [OPTIONAL, default=960] 
    
OUTPUTS:
    delta class from "calc_accessible_hydroxyl_fraction" class function

FUNCTIONS:
    ensure_type: Ensure type taken from mdtraj, which used to edit the sasa function
    custom_shrake_rupley: Custom shrake and rupley algorith based on parameters from Bondi
CLASS:
    calc_accessible_hydroxyl_fraction: calculates accessible hydroxyl fraction
    
USAGE:
    from accessible_hydroxyl_fraction import calc_accessible_hydroxyl_fractions

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
from MDDescriptors.core.check_tools import check_exists # Checks existance of a variable
from MDDescriptors.core.calc_tools import find_total_residues # Finds the total number of residues
import MDDescriptors.core.initialize as initialize # Checks file path
import numpy as np
import mdtraj as md
import time

### FUNCTION FROM THE MDTRAJ SOURCE CODE TO ENSURE THE CORRECT TYPE
def ensure_type(val, dtype, ndim, name, length=None, can_be_none=False, shape=None,
    warn_on_cast=True, add_newaxis_on_deficient_ndim=False):
    """Typecheck the size, shape and dtype of a numpy array, with optional
    casting.

    Parameters
    ----------
    val : {np.ndaraay, None}
        The array to check
    dtype : {nd.dtype, str}
        The dtype you'd like the array to have
    ndim : int
        The number of dimensions you'd like the array to have
    name : str
        name of the array. This is used when throwing exceptions, so that
        we can describe to the user which array is messed up.
    length : int, optional
        How long should the array be?
    can_be_none : bool
        Is ``val == None`` acceptable?
    shape : tuple, optional
        What should be shape of the array be? If the provided tuple has
        Nones in it, those will be semantically interpreted as matching
        any length in that dimension. So, for example, using the shape
        spec ``(None, None, 3)`` will ensure that the last dimension is of
        length three without constraining the first two dimensions
    warn_on_cast : bool, default=True
        Raise a warning when the dtypes don't match and a cast is done.
    add_newaxis_on_deficient_ndim : bool, default=True
        Add a new axis to the beginining of the array if the number of
        dimensions is deficient by one compared to your specification. For
        instance, if you're trying to get out an array of ``ndim == 3``,
        but the user provides an array of ``shape == (10, 10)``, a new axis will
        be created with length 1 in front, so that the return value is of
        shape ``(1, 10, 10)``.

    Notes
    -----
    The returned value will always be C-contiguous.

    Returns
    -------
    typechecked_val : np.ndarray, None
        If `val=None` and `can_be_none=True`, then this will return None.
        Otherwise, it will return val (or a copy of val). If the dtype wasn't right,
        it'll be casted to the right shape. If the array was not C-contiguous, it'll
        be copied as well.

    """
    # Importing modules
    import collections
    import warnings
    from mdtraj.utils.six.moves import zip_longest
    
    # Defining class
    class TypeCastPerformanceWarning(RuntimeWarning):
        pass
    
    
    
    if can_be_none and val is None:
        return None

    if not isinstance(val, np.ndarray):
        if isinstance(val, collections.Iterable):
            # If they give us an iterator, let's try...
            if isinstance(val, collections.Sequence):
                # sequences are easy. these are like lists and stuff
                val = np.array(val, dtype=dtype)
            else:
                # this is a generator...
                val = np.array(list(val), dtype=dtype)
        elif np.isscalar(val) and add_newaxis_on_deficient_ndim and ndim == 1:
            # special case: if the user is looking for a 1d array, and
            # they request newaxis upconversion, and provided a scalar
            # then we should reshape the scalar to be a 1d length-1 array
            val = np.array([val])
        else:
            raise TypeError(("%s must be numpy array. "
                " You supplied type %s" % (name, type(val))))

    if warn_on_cast and val.dtype != dtype:
        warnings.warn("Casting %s dtype=%s to %s " % (name, val.dtype, dtype),
            TypeCastPerformanceWarning)

    if not val.ndim == ndim:
        if add_newaxis_on_deficient_ndim and val.ndim + 1 == ndim:
            val = val[np.newaxis, ...]
        else:
            raise ValueError(("%s must be ndim %s. "
                "You supplied %s" % (name, ndim, val.ndim)))

    val = np.ascontiguousarray(val, dtype=dtype)

    if length is not None and len(val) != length:
        raise ValueError(("%s must be length %s. "
            "You supplied %s" % (name, length, len(val))))

    if shape is not None:
        # the shape specified given by the user can look like (None, None 3)
        # which indicates that ANY length is accepted in dimension 0 or
        # dimension 1
        sentenel = object()
        error = ValueError(("%s must be shape %s. You supplied  "
                "%s" % (name, str(shape).replace('None', 'Any'), val.shape)))
        for a, b in zip_longest(val.shape, shape, fillvalue=sentenel):
            if a is sentenel or b is sentenel:
                # if the sentenel was reached, it means that the ndim didn't
                # match or something. this really shouldn't happen
                raise error
            if b is None:
                # if the user's shape spec has a None in it, it matches anything
                continue
            if a != b:
                # check for equality
                raise error

    return val

### CUSTOM FUNCTION OF THE SHRAKE AND RUPLEY BASED ON THE BONDI VDW PARAMETERS
def custom_shrake_rupley(traj, probe_radius=0.14, n_sphere_points=960, mode='atom', verbose=False):
    """Compute the solvent accessible surface area of each atom or residue in each simulation frame.

    Parameters
    ----------
    traj : Trajectory
        An mtraj trajectory.
    probe_radius : float, optional
        The radius of the probe, in nm.
    n_sphere_points : int, optional
        The number of points representing the surface of each atom, higher
        values leads to more accuracy.
    mode : {'atom', 'residue'}
        In mode == 'atom', the extracted areas are resolved per-atom
        In mode == 'residue', this is consolidated down to the
        per-residue SASA by summing over the atoms in each residue.
    verbose : logical, optional
        True if you want to print every step

    Returns
    -------
    areas : np.array, shape=(n_frames, n_features)
        The accessible surface area of each atom or residue in every frame.
        If mode == 'atom', the second dimension will index the atoms in
        the trajectory, whereas if mode == 'residue', the second
        dimension will index the residues.

    Notes
    -----
    This code implements the Shrake and Rupley algorithm, with the Golden
    Section Spiral algorithm to generate the sphere points. The basic idea
    is to great a mesh of points representing the surface of each atom
    (at a distance of the van der waals radius plus the probe
    radius from the nuclei), and then count the number of such mesh points
    that are on the molecular surface -- i.e. not within the radius of another
    atom. Assuming that the points are evenly distributed, the number of points
    is directly proportional to the accessible surface area (its just 4*pi*r^2
    time the fraction of the points that are accessible).

    There are a number of different ways to generate the points on the sphere --
    possibly the best way would be to do a little "molecular dyanmics" : put the
    points on the sphere, and then run MD where all the points repel one another
    and wait for them to get to an energy minimum. But that sounds expensive.

    This code uses the golden section spiral algorithm
    (picture at http://xsisupport.com/2012/02/25/evenly-distributing-points-on-a-sphere-with-the-golden-sectionspiral/)
    where you make this spiral that traces out the unit sphere and then put points
    down equidistant along the spiral. It's cheap, but not perfect.

    The gromacs utility g_sas uses a slightly different algorithm for generating
    points on the sphere, which is based on an icosahedral tesselation.
    roughly, the icosahedral tesselation works something like this
    http://www.ziyan.info/2008/11/sphere-tessellation-using-icosahedron.html

    References
    ----------
    .. [1] Shrake, A; Rupley, JA. (1973) J Mol Biol 79 (2): 351--71.
    """

    # these van der waals radii are taken from GROMACS 4.5.3
    # and the file share/gromacs/top/vdwradii.dat
    #using the the same vdw radii for P and S based upon
    #http://en.wikipedia.org/wiki/Van_der_Waals_radius
    
    if verbose == True:
        ## UPDATED -- THESE TAKE BONDI's paper radii
        print(r"Using Bondi Radii, reference:")
        print(r"Bondi, A. van der Waals Volumes and Radii. J. Phys. Chem. 68, 441-451 (1964).")
    
    BONDI_ATOMIC_RADII = {'C': 0.170,  'F': 0.147,  'H': 0.120,
                     'N': 0.155, 'O': 0.152, 'S': 0.180,
                     'P': 0.180,
                     'Au': 0.213, # Taken from reference: 1. Batsanov, S. S. Van der Waals Radii of Elements. Inorg. Mater. Transl. from Neorg. Mater. Orig. Russ. Text 37, 871–885 (2001).
                     'VS': 0.213, # Same as gold, called "virtual site" since it is technically a bulk gold
                     } # in nm

    xyz = ensure_type(traj.xyz, dtype=np.float32, ndim=3, name='traj.xyz', shape=(None, None, 3), warn_on_cast=False)
    if mode == 'atom':
        dim1 = xyz.shape[1]
        atom_mapping = np.arange(dim1, dtype=np.int32)
    elif mode == 'residue':
        dim1 = traj.n_residues
        atom_mapping = np.array(
            [a.residue.index for a in traj.top.atoms], dtype=np.int32)
        if not np.all(np.unique(atom_mapping) ==
                      np.arange(1 + np.max(atom_mapping))):
            raise ValueError('residues must have contiguous integer indices '
                             'starting from zero')
    else:
        raise ValueError('mode must be one of "residue", "atom". "%s" supplied' %
                         mode)

    out = np.zeros((xyz.shape[0], dim1), dtype=np.float32)
    atom_radii = [BONDI_ATOMIC_RADII[atom.element.symbol] for atom in traj.topology.atoms]
    radii = np.array(atom_radii, np.float32) + probe_radius

    md.geometry._geometry._sasa(xyz, radii, int(n_sphere_points), atom_mapping, out)

    return out

## FUNCTION TO LOOK INTO ITP FILE AND EXTRACT ALCOHOLS/ETHERS
def findOxygenBondingWithinITP( currentITPFile_Location ):
    '''
    This function simply looks into your itp file, extracts all the data, and find the connectivities of the oxygens
    INPUTS:
        currentITPFile_Location: Full path to your itp file
    OUTPUTS:
        oxy_bond_info: All connectivity information of the oxygens
    '''
    
    ### Defining functions ###
    
    ## Function to read ITP file ##
    def readITP( currentITPFile_Location ):
        '''
        The purpose of this function is to read the itp file, remove the comments, and output each line as a list
        INPUTS:
            currentITPFile_Location: Full path to your itp file
        OUTPUTS:
            clean_itp: itp data as a list without comments (semicolons)
        '''
        with open(currentITPFile_Location,'r') as ITPFile:
             itp_data=ITPFile.read().splitlines()
        
        # Replacing all tabs (cleaning up)
        itp_data = [ eachLine.replace('\t', ' ') for eachLine in itp_data ]
        
        # Cleaning up itp file of all comments
        clean_itp =[ eachLine for eachLine in itp_data if not eachLine.startswith(";") ]
        
        return clean_itp
    
    ## Function to extract data from the itp file ##
    def extractDataType( clean_itp, desired_type ):
        '''
        The purpose of this function is to take your itp file and the desired type (i.e. bonds) and get the information from it. It assumes your itp file has been cleaned of comments
        INPUTS:
            clean_itp: itp data as a list without comments (semicolons)
            desired_type: Types that you want (i.e. [bonds])
        OUTPUTS:
            DataOfInterest: Data for that type as a list of list
        '''
        # Finding bond index
        IndexOfExtract = clean_itp.index(desired_type)
        
        # Defining the data we are interested in
        DataOfInterest = []
        currentIndexCheck = IndexOfExtract+1 
        
        # Using while loop to go through the list to see when this thing ends
        while ('[' in clean_itp[currentIndexCheck]) is False and currentIndexCheck != len(clean_itp) - 1: # Checks if the file ended
            # Appending to the data set
            DataOfInterest.append(clean_itp[currentIndexCheck])
            # Incrementing the index
            currentIndexCheck+=1     
            
        # Removing any blanks and then splitting to columns
        DataOfInterest = [ currentLine.split() for currentLine in DataOfInterest if len(currentLine) != 0 ]
        
        return DataOfInterest
    
    ## Function to create a dictionary with itp data ##
    def itp2dictionary(clean_itp):
        '''
        The purpose of this script is to take your clean itp file, then extract by types
        INPUTS:
            clean_itp: itp data as a list without comments (semicolons)
        OUTPUTS
            itp_dict: dictionary containing your different itp types
        '''
        # Finding all types that you can extract
        allTypes = [ eachLine for eachLine in clean_itp if '[' in eachLine ]
        
        # Finding all data
        data_by_type = [ extractDataType( clean_itp=clean_itp, desired_type=eachType) for eachType in allTypes ]
        
        # Creating dictionary for each one
        itp_dict = {}
        for currentIndex in range(len(allTypes)):
            itp_dict[allTypes[currentIndex]] = data_by_type[currentIndex]
            
        return itp_dict
    
    ## Finding bond information ##
    def findBondingFromITP( itp_dict ):
        '''
        The purpose of this script is to look into your itp dictionary, then find all the atom names, and correlate them to bonding information
        INPUTS:
            itp_dict: dictionary containing your different itp types
        OUTPUS:
            bond_by_atom_name: List of bonds
        '''
        
        # Defining the itp types you want
        atom_itp_name='[ atoms ]'    
        bond_itp_name = '[ bonds ]'
                          
        # Defining the data
        atom_itp_data = itp_dict[atom_itp_name]
        bond_itp_data = itp_dict[bond_itp_name]
        print("There are %s bonds and %s number of atoms"%(len(bond_itp_data), len(atom_itp_data)))
        
        # Getting atom numbers and atom names
        atom_num_names = [ [currentLine[0], currentLine[4]] for currentLine in atom_itp_data ]
        
        # Defining empty bonding data
        bond_by_atom_name=[]
        
        # Now, getting bonding data
        for currentBond in range(len(bond_itp_data)):
            
            # Defining atom numbers
            atom_A = bond_itp_data[currentBond][0]
            atom_B = bond_itp_data[currentBond][1]
            
            # Finding atom names
            atom_names = [ currentAtom[1] for currentAtom in atom_num_names if atom_A in currentAtom or atom_B in currentAtom ]
            
            # Appending the data
            bond_by_atom_name.append(atom_names)
        
        return bond_by_atom_name
    
    ## Function to extract oxygens from bonding information ##
    def findOxyBonding( bond_by_atom_name ):
        '''
        The purpose of this script is to go through the bond_by_atom_name and find the bonds between oxygen atoms
        INPUTS:
            bond_by_atom_name: List of bonds by atom name
        OUTPUTS:
            oxy_bond_info: List of oxygen, and its corresponding bonded atoms
        
        '''
        # Finding all the unique oxygen names that is sorted
        allOxygenNames = sorted(list(set([ eachAtom for eachBond in bond_by_atom_name for eachAtom in eachBond if 'O' in eachAtom])))
        print("There are a total of %s oxygens, finding oxygen bonding information"%(len(allOxygenNames)))
        
        # Defining a list to store bonding information
        oxy_bond_info = []
        # Go through each oxygen and find it's corresponding bond
        for eachOxygen in range(len(allOxygenNames)):
            # Defining current oxygen
            currentOxygen = allOxygenNames[eachOxygen]
            
            # Looping through each bond and finding the oxygen
            bonds_with_oxygen = [ currentBond for currentBond in bond_by_atom_name if currentOxygen in currentBond ]
            
            # Looping to remove the oxygen (getting the atoms that are in fact bonded)
            [ currentBond2Oxy.remove(currentOxygen) for currentBond2Oxy in bonds_with_oxygen]
            
            # Finding atoms that are currently bound
            atoms_currently_bound = [currentBond[0] for currentBond in bonds_with_oxygen ]
            
            # Now, creating a simple list to denote the bonding atoms
            oxy_bond_info.append( [ currentOxygen ] + atoms_currently_bound )
                
        return oxy_bond_info
    
    
    ## Function to look at the oxygen and determine its functionality
    def defineOxygenBondType( oxy_bond_info ):
        '''
        The purpose of this script is to look at your oxygen bonding to see what type it is (e.g. alcohol, ether ,etc.)
        INPUTS:
            oxy_bond_info: List of oxygen, and its corresponding bonded atoms
        OUTPUTS:
            oxy_bond_info: Same list, but with labels of alcohol, ether, etc. as the last column
            
        Note: This contains a bonding database to determine whether or not you have a alcohol, etc.
        '''
        # bond_database -- 1st is number of carbons, --2nd is number of hydrogens
        bond_database=[ [ 1, 1 , "alcohol"], 
                        [ 2, 0 , "ether"]
                       ]
    
        # Now, counting number of carbons / hydrogens bonded to the oxygen
        totalCarbons = [ len([currentAtom for currentAtom in currentBond if 'C' in currentAtom ]) for currentBond in oxy_bond_info ]
        totalHydrogens = [ len([currentAtom for currentAtom in currentBond if 'H' in currentAtom ]) for currentBond in oxy_bond_info ]
        
        # Going through each oxygen and designating its name
        for eachOxyBond in range(len(oxy_bond_info)):
            # Defining the oxygen bond
            currentTotal_Carbon = totalCarbons[eachOxyBond]
            currentTotal_Hydrogen = totalHydrogens[eachOxyBond]
            
            # Going to database and adding to oxy_bond_info
            try:
                index2Bond_Database = [ currentBond for currentBond in range(len(bond_database)) if bond_database[currentBond][0] == currentTotal_Carbon and bond_database[currentBond][1] ==  currentTotal_Hydrogen][0]
                
                # Printing
                print("Since you have %s carbons and %s hydrogens bonded to your oxygen at %s, it is an %s bond"%(currentTotal_Carbon,currentTotal_Hydrogen,
                                                                                                                 oxy_bond_info[eachOxyBond][0], bond_database[index2Bond_Database][-1]))
                
                # Adding to oxygen bond information
                oxy_bond_info[eachOxyBond] = oxy_bond_info[eachOxyBond] +  [ bond_database[index2Bond_Database][-1] ]
            except:
                print("Your alcohol is not an alcohol or an ether! It is probably some other functionalities!")
                print("Ignoring, please see bonding between: %s"%(oxy_bond_info[eachOxyBond]))
                print("Pausing... 3 seconds so you can see this!")
                time.sleep(3)

        return oxy_bond_info
    
    
    ### Main Script ###
    
    # Reading itp file
    clean_itp = readITP( currentITPFile_Location )
    
    # Extraction of itp file data
    itp_dict = itp2dictionary(clean_itp)
    
    # Now, going through atom data and extracting the value and atom name
    bond_by_atom_name = findBondingFromITP( itp_dict )
        
    # Go through the bonds, find the oxygens, then see if the oxygen is an alcohol oxygen or ether oxygen
    oxy_bond_info = findOxyBonding( bond_by_atom_name )
    
    # Now, designate the bond based on what we know
    oxy_bond_info = defineOxygenBondType( oxy_bond_info )
    
    return oxy_bond_info

#################################################################
### CLASS FUNCTION TO CALCULATE ACCESSIBLE HYDROXYL FRACTIONS ###
#################################################################
class calc_accessible_hydroxyl_fraction:
    '''
    The purpose of this class is to calculate the accessible hydroxyl fraction
    INPUTS:
        traj_data: Data taken from import_traj class
        solute_itp_file: Name of solute ITP file -- should be within the same directory as the trajectory information
        input_details: input arguments, should be a dictionary
            'Solute': Solute name (single) -- will be used to calculate the hydroxyl fraction
            'SASA_type': Type of sasa you want. e.g.
                'oxy2carbon' - oxygen to carbon
                'oxy2carbon_OH_only' - oxygen alcohol to carbon ratio
                'alcohol2allSASA' - alcohol oxygen + hydrogen divided by entire molecule SASA
            'probe_radius': probe radius in nm of the rolling ball [OPTIONAL, default=0.14 ]
            'num_sphere_pts': number of points to represent your sphere [OPTIONAL, default=960]                
            
    OUTPUTS:
        ## INPUT VALUES
            self.solute_itp_file: itp file name
            self.solute_name: Residue name of the solute
            self.SASA_type: Type of the SASA you want
            self.probe_radius: Probe radius
            self.num_sphere_pts: Number of points on a sphere
        ## SOLUTE INFORMATION
            self.itp_file_path: itp file of the solute
        ## RESULTS:
            self.accessible_hydroxyl_frac: accessible hydroxyl fraction for the solute
        
    FUNCTIONS:
        findSASA_Oxy2CarbonRatio: Main function to calculate the hydroxyl fraction
    '''
    ### INITIALIZING
    def __init__(self, traj_data, solute_itp_file, **input_details):
        ## STORING INPUTS
        self.solute_itp_file = solute_itp_file
        self.solute_name = input_details['Solute']
        self.SASA_type = input_details['SASA_type']
        self.probe_radius = check_exists( input_details['probe_radius'], 0.14 )
        self.num_sphere_pts = check_exists( input_details['num_sphere_pts'], 960 )
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## DEFINING SOLUTE ITP FILE
        self.itp_file_path = initialize.checkPath2Server(traj_data.directory + '/' + self.solute_itp_file)
        
        ## RUNNING MAIN SCRIPT TO CALCULATE SASA
        self.accessible_hydroxyl_frac = self.findSASA_Oxy2CarbonRatio(
                                                                        traj = traj,
                                                                        residueName = self.solute_name,
                                                                        Probe_radius = self.probe_radius,
                                                                        Num_Sphere_Pts = self.num_sphere_pts,
                                                                        SASA_type = self.SASA_type,
                                                                        itp_location = self.itp_file_path,
                                                                        wantCustomSASA = True, # True if you want Bondi VDW parameters, otherwise this will resort back to default
                                                    )
        
    
    ### FUNCTION TO CALCULATE THE SASA RATIO 
    @staticmethod
    def findSASA_Oxy2CarbonRatio(traj = None, 
                                 residueName = None, 
                                 Probe_radius = 0.14 , 
                                 Num_Sphere_Pts = 960, 
                                 SASA_type = 'oxy2carbon', 
                                 itp_location = None,
                                 wantCustomSASA = True):
        '''
        The purpose of this script is to simply find the SASA oxygen to carbon ratio. This algorithm looks for the residue you are interested in, then truncates the trajectory for a speedy SASA calculation. The SASA for oxygens / carbons are time-averaged and pair averaged before taking the ratio. In general, the SASA should not change with different simulations. More or less, the SASA should remain the same for each solute.
        INPUTS:
            traj: trajectory from md.traj
            residueName: Name of your residue
            Probe_radius: radius of your probe
            Num_Sphere_Pts: Number of sphere points
            SASA_type: Type of your SASA, e.g.:
                'oxy2carbon' - oxygen to carbon
                'oxy2carbon_OH_only' - oxygen alcohol to carbon ratio
                'alcohol2allSASA' - alcohol oxygen + hydrogen divided by entire molecule SASA
                'oxygen2allSASA' - all oxygen divided by the entire molecule SASA
            wantCustomSASA: If you want SASA based on the current script (True) or SASA based on md.traj
        OUTPUTS:
            SASA_oxy_to_carbon: scalar with the ratio of OXYGEN to CARBON SASAs
        
        '''
        # Importing modules
        import os.path
        import sys
        
        # Printing
        print("-------------------- SASA Calculation -------------------- ")
        print("Finding SASA for current trajectory with residue name: %s"%(residueName))
        
        # Finding residue of solute
        num_residues, index_residues = find_total_residues(traj=traj, resname=residueName)
    
        # Getting residue
        currentResidue=[ traj.topology.residue(current_res_index) for current_res_index in index_residues]
    
        # Getting all the atoms within residue
        currentAtomIndex=[ currentAtom.index for currentRes in currentResidue for currentAtom in currentRes.atoms ]
        
        # Sorting before atom slicing
        currentAtomIndex.sort()
        
        print("Slicing trajectory..............")
        # Atom slicing to get new trajectory
        new_traj = traj.atom_slice(atom_indices = currentAtomIndex, inplace=False )
        
        # Calculating SASA
        if wantCustomSASA is True:
            print("Using CUSTOM SASA")
            sasa = custom_shrake_rupley(traj=new_traj[:],
                     probe_radius=Probe_radius, # in nms
                     n_sphere_points=Num_Sphere_Pts, # Larger, the more accurate
                     mode='atom' # Extracted areas are per atom basis
                     )
            
        else:
            print("Using MDTRAJ SASA")
            sasa = md.shrake_rupley(traj=new_traj[:],
                                 probe_radius=Probe_radius, # in nms
                                 n_sphere_points=Num_Sphere_Pts, # Larger, the more accurate
                                 mode='atom' # Extracted areas are per atom basis
                                 )
    
        
        # Getting indices of all carbons
        carbon_index = [[currentAtom.index  for currentAtom in currentRes.atoms if currentAtom.element.symbol == 'C' ] for currentRes in new_traj.topology.residues if currentRes.index in index_residues ]
    
        if SASA_type == 'oxy2carbon' or SASA_type == 'oxygen2allSASA':
            print("Running SASA for all oxygen")
            # Getting indices of all oxygens
            oxygen_index = [[currentAtom.index  for currentAtom in currentRes.atoms if currentAtom.element.symbol == 'O' ] for currentRes in new_traj.topology.residues if currentRes.index in index_residues ]
            print("Found a total of %s oxygen atoms"%( np.sum([ len(oxygen_index[currentRes]) for currentRes in range(len(index_residues)) ] ) ) )
    
        elif SASA_type == 'oxy2carbon_OH_only' or SASA_type == 'alcohol2allSASA':
            print("Running SASA for oxygen alcohols to carbon")
            # Check if itp file is there
            if os.path.isfile(itp_location) is False:
                print("Error! No ITP file found, look at: %s"%(itp_location) )
                sys.exit()
            else:    
                # Need connectivity from ITP file
                oxy_bond_info = findOxygenBondingWithinITP( itp_location )
                
                # Finding all oxygens that are alcohols
                oxy_alcohol = [ currentAtom for currentBondInfo in oxy_bond_info if currentBondInfo[-1] == 'alcohol' for currentAtom in currentBondInfo if 'O' in currentAtom ]
                
                # Getting indices of all oxygens
                oxygen_index = [[currentAtom.index for currentAtom in currentRes.atoms if currentAtom.element.symbol == 'O' and currentAtom.name in oxy_alcohol  ] for currentRes in new_traj.topology.residues if currentRes.index in index_residues]
                print("Found a total of %s alcohol atoms"%( np.sum([ len(oxygen_index[currentRes]) for currentRes in range(len(index_residues)) ] ) ) )
                
                # np.mean(sasa[:,[oxygen_index[0] + hydrogen_index[0]]], axis=0)
                if SASA_type == 'alcohol2allSASA':
                    print("Running SASA for alcohols to entire molecule")
                    # Finding all hydrogens
                    hyd_alcohol = [ currentAtom for currentBondInfo in oxy_bond_info if currentBondInfo[-1] == 'alcohol' for currentAtom in currentBondInfo if 'H' in currentAtom ]
                    
                    # Finding indexes
                    hydrogen_index = [[currentAtom.index for currentAtom in currentRes.atoms if currentAtom.element.symbol == 'H' and currentAtom.name in hyd_alcohol  ] for currentRes in new_traj.topology.residues if currentRes.index in index_residues]
                    
                    # Adding to current oxygen index
                    oxygen_index = [ oxygen_index[0] + hydrogen_index[0]]
    
        else:
            print("Error! The SASA type %s is not available. Please see script"%(SASA_type))
            sys.exit()
    
    
        # Now, extracting the value that we want
        oxygen_SASA_values = sasa[:,oxygen_index]
        carbon_SASA_values = sasa[:,carbon_index]
        
        # Averaging all carbons and oxygens, then taking ratio
        SASA_oxy_to_carbon = np.sum(np.mean(oxygen_SASA_values, axis=0))/np.sum(np.mean(carbon_SASA_values, axis=0))
        
        # If all SASA
        if SASA_type == 'alcohol2allSASA' or SASA_type == 'oxygen2allSASA':
            print("Summing SASA for entire molecule")
            totalSASA = np.sum((np.mean( sasa, axis=0 )))
            SASA_oxy_to_carbon = np.sum(np.mean(oxygen_SASA_values, axis=0)) / totalSASA
            
        # Printing
        print("The average SASA ratio for %s type was: %s"%(SASA_type, SASA_oxy_to_carbon))
        return SASA_oxy_to_carbon

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    #analysis_dir=r"180302-Spatial_Mapping" # Analysis directory
    analysis_dir=r"180316-ACE_PRO_DIO_DMSO"
    specific_dir="TBA\\TBA_50_GVL" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    specific_dir="ACE/mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir=r"Planar_310.15_ROT_TMMA_10x10_CHARMM36_withGOLP" # Directory within analysis_dir
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    #xtc_file=r"mixed_solv_prod_last_90_ns_center_rot_trans_center.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    xtc_file=r"mixed_solv_last_50_ns_whole.xtc"
    itp_file=r"acetone.itp"
    #itp_file=r"tBuOH.itp"
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    
    ### DEFINING INPUT DATA
    input_details={
                'Solute'        : 'ACE',             # Solute of interest
                'SASA_type'     : 'alcohol2allSASA', # Type of SASA you want
                'probe_radius'  :  0.14,             # Probe radius in nm (size of the probe, by default 0.14 nm for the VDW of water)
                'num_sphere_pts':   960,             # Number of sphere points, the higher, the more accurate
                }
    
    ### CALLING ACCESSIBLE HYDROXYL FRACTION CLASS
    accessible_hydroxyl_frac = calc_accessible_hydroxyl_fraction(traj_data, itp_file, **input_details)
    