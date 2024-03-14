#!/bin/bash

# combine.sh
# This code checks simulations and sees which sims need to be exchanged


## CREATING FUNCTION THAT CHECKS CPT TIME
function check_cpt_time () {
	## DEFINING INPUT VARIABLE
	cpt_file_="$1"

	## RUNNING
	time_ps=$(gmx check -f ${cpt_file_} 2>&1 | grep "Last frame" | awk '{print $NF}')

	## PRINTING
	echo "${time_ps}"

}

## PATH TO SIM. THIS ONE WILL BE UPDATED.
path_to_sim_to_update="/home/akchew/scratch/nanoparticle_project/nplm_sims/20200822-Hydrophobic_contacts_PMF_Bn_new_FF/UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1/4_simulations"

## PATH TO SIM TO CHECK
path_to_sim_to_check="/home/akchew/scratch/nanoparticle_project/nplm_sims/20200822-Hydrophobic_contacts_PMF_Bn_new_FF/UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1_stampede/4_simulations"

## GOING INTO FILE
cd "${path_to_sim_to_check}"

## DEFINING PROD FILE
prod_cpt="nplm_prod.cpt"

## DEFINING ARCHIVE SIMS
archive_sims="archive_sims"

## DEFINING IF YOU WANT TO MOVE
want_move=true 
# false if you do not want to move, but just print out the details

## READING ALL FILES
read -a sims_list <<< $(ls -d ./*/ | sort -V)

## LOOPING
for each_folder in ${sims_list[@]}; do

	## GETTING LAST FRAME OF CURRENT SIM TO CHECK
	current_last_frame=$(check_cpt_time "${each_folder}/${prod_cpt}")

	## PERFORMING THE SAME
	current_folder_sim_update="${path_to_sim_to_update}/${each_folder}"
	sim_to_update_last_frame=$(check_cpt_time "${current_folder_sim_update}/${prod_cpt}")

	## CHECKING IF CURRENT LAST FRAME IS LARGER
	if [ 1 -eq "$(echo "${sim_to_update_last_frame} < ${current_last_frame}" | bc)" ]; then 
		# awk 'BEGIN {print ('${current_last_frame}' >= '${sim_to_update_last_frame}')}'; then
		# (( $(echo "${current_last_frame} > ${sim_to_update_last_frame}") | bc -l )); then
		is_larger=true
	else
		is_larger=false
	fi

	## PRINTING
	echo "Checking: ${each_folder} - ${current_last_frame} ns <--> ${sim_to_update_last_frame} ns (comparison) == ${is_larger} copy"

	## CHECKING
	if [[ "${is_larger}" == true ]]; then

		## DEFINING PATH TO ARCHIVE SIMS
		path_archive="${path_to_sim_to_update}/../${archive_sims}"

		## CREATING DIRECTORY
		if [ ! -e "${path_archive}" ]; then
			mkdir "${path_archive}"
		fi

		## PRINT STEPS
		echo "   Archiving ${current_folder_sim_update}"
		echo "   Copying ${each_folder} -> ${current_folder_sim_update}"

		## CHECKING IF YOU WANT THE MOVE
		if [[ "${want_move}" == true ]]; then
			# MOVING FILE
			mv "${current_folder_sim_update}" "${path_archive}"

			# COPYING OVER FILE
			cp -r "${each_folder}" "${current_folder_sim_update}"
		fi

	fi





done

## CHECKING FILE

# for file in $(ls | sort -V); do echo $file; gmx check -f $file/nplm_prod.cpt 2>&1 | grep "Last frame"; done
