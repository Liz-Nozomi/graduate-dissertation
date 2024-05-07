#!/bin/bash

# change as required
LOCATION="/Users/liz/krakenx/build"

wdir=`pwd`

# single molecule containing 3D coordinates in SDF format
sfile=$1 
# Corresponding mopac output 
mfile=$2
# Prefix the descriptor names if required
prefix=$3
fname=`basename $sfile .sdf`

ofile=$fname"_desc.txt"


ls $mfile/*.out > mopac.txt
ls $sfile/*.sdf > mols.txt

echo "# File containing the list of mopac files to be analysed" > pars.txt
echo "# Provide full paths" >> pars.txt
echo "lstMopFileName=$wdir/mopac.txt" >> pars.txt
echo "" >> pars.txt
echo "# File containing the list of SDF files to be analysed" >> pars.txt
echo "# Should be in the same order as the mopac files" >> pars.txt
echo "lstSDFFileName=$wdir/mols.txt" >> pars.txt
echo "" >> pars.txt
echo "# Output file for descriptors" >> pars.txt
echo "outputFileName=$wdir/$ofile" >> pars.txt
echo "" >> pars.txt
echo "# Descriptor List" >> pars.txt
echo "# Indicate Yes/No or yes/no or Y/N or y/n and add additional parameters where required" >> pars.txt
echo "# a blank after the \"=\" sign implies No" >> pars.txt
echo "" >> pars.txt
echo "# Calculate EVA descriptors" >> pars.txt
echo "# Assumes that the mopac files have the vibrational frequencies" >> pars.txt
echo "EVA=N" >> pars.txt
echo "evaSigma=2" >> pars.txt
echo "evaL=1" >> pars.txt
echo "evaMinVal=1" >> pars.txt
echo "evaMaxVal=4000" >> pars.txt
echo "" >> pars.txt
echo "" >> pars.txt
echo "# Calculate EEVA descriptors" >> pars.txt
echo "EEVA=N" >> pars.txt
echo "eevaSigma=0.050" >> pars.txt
echo "eevaL=0.025" >> pars.txt
echo "eevaMinVal=-45" >> pars.txt
echo "eevaMaxVal=10" >> pars.txt
echo "" >> pars.txt
echo "" >> pars.txt
echo "# 3D-MORSE" >> pars.txt
echo "Morse=N" >> pars.txt
echo "" >> pars.txt
echo "# 3D-WHIM" >> pars.txt
echo "whim=N" >> pars.txt
echo "" >> pars.txt
echo "# 3D-autocorrelation" >> pars.txt
echo "autocorrelation=N" >> pars.txt
echo "" >> pars.txt
echo "# RDF" >> pars.txt
echo "rdf=N" >> pars.txt
echo "#RDFBETA=" >> pars.txt
echo "" >> pars.txt
echo "# BCUT" >> pars.txt
echo "bcut=N" >> pars.txt
echo "" >> pars.txt
echo "# Coulomb matrix" >> pars.txt
echo "COULOMBMATRIX=N" >> pars.txt
echo "# CPSA" >> pars.txt
echo "cpsa=Y" >> pars.txt
echo "" >> pars.txt
echo "# Charge" >> pars.txt 
echo "chargedesc=Y" >> pars.txt
echo "" >> pars.txt
echo "# MOPAC" >> pars.txt
echo "# Basic MOPAC calculated quantities" >> pars.txt
echo "mopac=Y" >> pars.txt
echo "" >> pars.txt
echo "# Geometry/shape descriptors" >> pars.txt
echo "geometry=Y" >> pars.txt
echo "" >> pars.txt
echo "# provide a prefix if required" >> pars.txt
echo "prefix=$prefix" >> pars.txt
echo "" >> pars.txt
echo "# Graph eigenvalues" >> pars.txt 
echo "graphenergy=Y" >> pars.txt
echo "" >> pars.txt
echo "# Distance profile" >> pars.txt
echo "dip=N" >> pars.txt
echo "" >> pars.txt
echo "" >> pars.txt
echo "# specify charge calculation scheme (MOPAC/EEM/UDF). Default is MOPAC charges." >> pars.txt
echo "# Specifying a user-defined scheme requires the user to provide a file containing" >> pars.txt
echo "# charges for each molecule." >> pars.txt
echo "ChargeType=MOPAC" >> pars.txt
echo "#lstChargeFileName=" >> pars.txt
echo "" >> pars.txt
echo "" >> pars.txt
echo "# Weighting schemes for RDF,MORSE,WHIM,BCUT and auto/cross correlation" >> pars.txt
# Atomic weights can be changed accordingly
echo "charge=Yes" >> pars.txt
echo "selfpol=Yes" >> pars.txt
echo "nucleardeloc=Yes" >> pars.txt
echo "electrophilicdeloc=Yes" >> pars.txt
echo "radicaldeloc=Yes" >> pars.txt
echo "chgden=No" >> pars.txt
echo ""


java -jar $LOCATION/KrakenX.jar pars.txt

rm pars.txt mopac.txt mols.txt


