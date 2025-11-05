#%%
#Vision preprocessing script for workshop
#Authors: Nicole Jurjew, Viet Dao, Zekai Li
# Data file: https://filesender.surf.nl/?s=download&token=7f1c595b-8e60-4e79-a73d-b7b98dfb25ee
# A phantom dataset and a reconstruction parameter file is provided along with the preprocessing script.
#%%
import os
import numpy as np
import sys
import re
import stir
import stirextra

import matplotlib.pyplot as plt
import Visiondata_preprocess_functions as vpf

#%%

## You'll need to run the e7tools with the following commands
## to get smoothed randoms, a mu-map and norm-sino out:
##
## in any batch-file, add the following 2 lines, then execute
##  set cmd= %cmd% -d ./Debug
##  set cmd= %cmd% --os scatter_520_2D.mhdr
##
## the e7tools provide the prompts sinogram in a compressed file-format.
## STIR can't read that, so you'll have to uncompress it first:
## C:\Siemens\PET\bin.win64-VG80\intfcompr.exe -e path\to\compressed\sinogram\filename.mhdr ^
## --oe path\to\UNcompressed\sinogram\NEWfilename.mhdr

#%%
## Here's a list of files you need (from the e7tools)
## - prompts-sino_uncompr_00.s.hdr & .s
## - smoothed_randoms_00.h33 & .s (in the "Debug" folder)
## - umap_00.h33 & .i (in the "Debug" folder)
##      note: we highly recommend using this mu-map, as it is "cut to the FOV" and is positioned at
##      scanner center.
## - norm3d_00.h33 & .a (in the "Debug" folder)
## - scatter_520_2D.s.hdr & .s (wherever you set the path to)
## - acf_00.h33 & .a (in the "Debug" folder)

#%%
##### if you have any Siemens image data that you want to read in, make sure:
##### change the word "image data" to "imagedata"
##### remove line "!image relative start time (sec)...""
##### remove line "!image duration (sec):=1200..."

#%%

## Your own file folder path here
data_folder_PATH = './Hoffman_phantom_raw_data_for_workshop'

apply_DOI_adaption = False

####### existing files #######
prompts_header_filename = 'uncompressed_emission_00.s.hdr' #Uncompressed prompts header
mu_map_header = 'umap_00.h33' # the *.h33 header
randoms_data_filename = 'smoothed_rand_00.s'
scatter_2D_header_filename = 'scatter_520_2D_00_00.s.hdr'
norm_sino_data_filename= 'norm3d_00.a'
attenuation_corr_factor_data_filename = 'acf_00.a'
STIR_output_folder = 'processing'

####### variables to chose #######
# header-name for prompts as we don't want to overwrite the Siemens header
prompts_header_to_read_withSTIR = prompts_header_filename[:-6] + '_readwSTIR.s.hdr'
# STIR writes the (DOI-adapted) prompts out to a STIR-file:
prompts_filename_STIR_corr_DOI = 'prompts.hs'
# STIR writes a non-TOF sinogram, name:
nonTOF_template_sinogram_name = 'template_nonTOF.hs'
# header-name for attenuation correction factors as we don't want to overwrite the Siemens header
attenuation_corr_factor_header_to_read_withSTIR = attenuation_corr_factor_data_filename[:-2] + '_readwSTIR.s.hdr'
# header-name for randoms as we don't want to overwrite the Siemens header
norm_sino_to_read_withSTIR = norm_sino_data_filename[:-2] + '_readwSTIR.s.hdr'
# STIR writes the DOI-adapted, negative corrected norm-sino to
norm_filename_fSTIR = 'norm_sino_fSTIR.hs'
# header-name for randoms as we don't want to overwrite the Siemens header
randoms_header_to_read_withSTIR = randoms_data_filename[:-2] + '_readwSTIR.s.hdr'
# header-name for randoms as we don't want to overwrite the Siemens header
scatter_2D_header_to_read_withSTIR = scatter_2D_header_filename[:-6] + '_readwSTIR.s.hdr'
# STIR writes the (DOI-adapted) randoms to
randoms_adapted_DOI_filename = randoms_data_filename[:-2] + '_DOI_fSTIR.hs'
# STIR writes the (DOI-adapted), iSSRBd, unnormalized scatter to
scatter_3D_unnorm_filename = 'scatter_3D_unnormalized.hs'
# STIR writes the additive term (that's normalized scatter + normalized randoms, attenuation corrected) to:
additive_term_filename_fSTIR = 'additive_term.hs'
# STIR writes the multiplicative term (that's norm_sino * attenuation_CORRECTION_factors) to:
multi_term_filename_fSTIR = 'mult_factors_forSTIR.hs'

#%%
os.chdir(data_folder_PATH)

try:
    os.mkdir(STIR_output_folder)
except FileExistsError:
    print("STIR output folder exists, files are overwritten")
    
#%%
###################### PROMPTS FILE ############################
##### first, we check if the prompts file is compressed
vpf.check_if_compressed(prompts_header_filename)

##### as the e7tools run on Windows, the data file-name needs to be changed
vpf.change_datafilename_in_interfile_header(prompts_header_to_read_withSTIR, prompts_header_filename ,prompts_header_filename[:-4])

# ## we're ready to read the prompts with STIR now
prompts_from_e7 = stir.ProjData.read_from_file(prompts_header_to_read_withSTIR)
prompts_from_e7 = stir.ProjDataInMemory(prompts_from_e7)

##!!!! Question 1: How to do a check if the reading works?

#%%
###################### DOI ADAPTION ############################
## after comparing e7tools and STIR forward projections, we've found out we have to change
## the crystal depth of interaction (DOI) from 7mm to 10mm to minimize the differences.
if apply_DOI_adaption: vpf.DOI_adaption(prompts_from_e7, 10)

## write to file so you can use it later
prompts_from_e7.write_to_file(os.path.join(STIR_output_folder, prompts_filename_STIR_corr_DOI))
# prompts_from_e7.write_to_file('prompts.hs')

# %%
###################### MU-MAP ############################
## to read in the mu-map, we have to convert the Siemens *.h33 header to STIR *.hv header via a STIR -script
# note: here, a file .h33.tmp is created, I don't know why, but can be deleted
cmd_string_conv_intf_to_stir = 'convertSiemensInterfileToSTIR.sh \
    {} {}'.format(mu_map_header, mu_map_header[:-3]+'hv')

try:
    exit_status = os.system(cmd_string_conv_intf_to_stir)
    if exit_status != 0:
        raise Exception('Command failed with status: {}'.format(exit_status))
except Exception as e:
    print('An error occurred: {}'.format(e))

#### now let's read it from file and plot to see if it worked
mu_map = stir.FloatVoxelsOnCartesianGrid.read_from_file(mu_map_header[:-3]+'hv')
mu_map_arr = stirextra.to_numpy(mu_map)

plt.figure()
vpf.plot_2d_image([1,1,1],mu_map_arr[mu_map_arr.shape[0]//2,:,:],'mu-map')
plt.savefig(os.path.join(STIR_output_folder,'mu_map.png'), transparent=False, facecolor='w')
plt.show()

#%%
###################### NON-TOF TEMPLATE ############################
## you might need a non-TOF sinogram with the same geometry (for attenuation factors e.g.)
## so we're creatiing one here
proj_info = prompts_from_e7.get_proj_data_info()
nonTOF_proj_info = proj_info.create_non_tof_clone()
nonTOFtemplate=stir.ProjDataInterfile(prompts_from_e7.get_exam_info(), nonTOF_proj_info, os.path.join(STIR_output_folder,nonTOF_template_sinogram_name))

#%%
######## select display variables
central_slice = proj_info.get_num_axial_poss(0)//2
TOF_bin = proj_info.get_num_tof_poss()//2
view = proj_info.get_num_views()//2
# to draw line-profiles, we'll average over a few slices, specify how many:
thickness_half = 5


# %%
###################### NORMALIZATION ############################
### we're using the prompts-header as a template for the norm-header
vpf.change_datafilename_in_interfile_header(norm_sino_to_read_withSTIR, prompts_header_filename, norm_sino_data_filename)
vpf.change_datatype_in_interfile_header(norm_sino_to_read_withSTIR, 'float', 4)

#%%
#### ready to read in norm-sino with STIR
norm_sino = stir.ProjData.read_from_file(norm_sino_to_read_withSTIR)
norm_sino = stir.ProjDataInMemory(norm_sino)

###################### DOI ADAPTION ############################
## after comparing e7tools and STIR forward projections, we've found out we have to change
## the crystal depth of interaction (DOI) from 7mm to 10mm to minimize the differences.
if apply_DOI_adaption: vpf.DOI_adaption(norm_sino, 10)
norm_sino_arr = stirextra.to_numpy(norm_sino)

##### in case there were bad miniblocks during your measurement, the norm-file 
##### might contain negative values. We'll set them to a very high value here, such
##### that the detection efficiencies (1/norm-value) will be 0 (numerically)
norm_sino_arr[norm_sino_arr<=0.] = 10^37
norm_sino.fill(norm_sino_arr.flat)

#### this is the data STIR needs in an Acquisition Sensitivity model, so we'll write it out
norm_sino.write_to_file(os.path.join(STIR_output_folder,norm_filename_fSTIR))

##!!!! Question 2: What can we check here (hint: check plot_2d_image in vision_processing_function.py)?

#%%
#### to compute scatter and for use in SIRF, we'll need the detection efficiencies
#### which are computed as 1/norm-sino
det_eff = stir.ProjDataInMemory(norm_sino)
det_eff_arr = np.nan_to_num(np.divide(1, norm_sino_arr))

#%%
###################### RANDOMS ############################
### below, we will use the prompts-header as a template for a header to read in Siemens randoms
vpf.change_datafilename_in_interfile_header(randoms_header_to_read_withSTIR, prompts_header_filename, randoms_data_filename)
### need to change the file type, because prompts are in unsigned short, randoms are in float
vpf.change_datatype_in_interfile_header(randoms_header_to_read_withSTIR, 'float', 4)

### The data type descriptions are true for the prompts, but not
### for the randoms, so we need to remove them (and some other info) from the header.
vpf.remove_scan_data_lines_from_interfile_header(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)
vpf.remove_IMGDATADESC_lines_from_interfile_header(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)

### The first set of sinograms in the randoms file is the "delayeds", so we need
### to tell STIR to ignore that by setting a data offset. The other offset fields,
### we can just remove, in line with the "data types" we removed above.
vpf.remove_data_offset(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)
vpf.add_data_offset(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)

# %%
# #### read in again & plot to see if it worked
randoms = stir.ProjData.read_from_file(randoms_header_to_read_withSTIR)
if apply_DOI_adaption: vpf.DOI_adaption(randoms, 10)
randoms.write_to_file(os.path.join(STIR_output_folder,randoms_adapted_DOI_filename))
randoms_arr = stirextra.to_numpy(randoms)

for i in range(33):
    plt.figure()
    vpf.plot_2d_image([1,1,1],randoms_arr[i, central_slice,:,:],'randoms, TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'randoms_sino.png'), transparent=False, facecolor='w')
    plt.show()
#%%
###################### SCATTER ############################
# alter e7-header of scatter manually to read 2D scatter
seg0_max_rd = 9 # TODO: get this via proj-data-info; currently cannot “downcast” to ProjDataInfoCylindrical
vpf.remove_data_offset(scatter_2D_header_to_read_withSTIR, scatter_2D_header_filename)
vpf.remove_scan_data_lines_from_interfile_header(scatter_2D_header_to_read_withSTIR, scatter_2D_header_to_read_withSTIR)
vpf.replace_siemens_convention_in_interfile_header(scatter_2D_header_to_read_withSTIR, scatter_2D_header_to_read_withSTIR)
vpf.change_max_ring_distance(scatter_2D_header_to_read_withSTIR, scatter_2D_header_to_read_withSTIR, seg0_max_rd)

#%%
## read in 2D scatter with STIR
scatter_2D_normalized = stir.ProjData.read_from_file(scatter_2D_header_to_read_withSTIR)
if apply_DOI_adaption: vpf.DOI_adaption(scatter_2D_normalized, 10)

#%%
## we hand the object of the final 3D sinogram over to the inverse SSRB function.
scatter_3D_normalized = stir.ProjDataInMemory(prompts_from_e7)
stir.inverse_SSRB(scatter_3D_normalized, scatter_2D_normalized)

#%%
# plot to see if it worked
scatter_3D_norm_arr = stirextra.to_numpy(scatter_3D_normalized)
for i in range(33):
    plt.figure()
    vpf.plot_2d_image([1,1,1],scatter_3D_norm_arr[i, central_slice,:,:],'scatter, normalized, TOF bin {}'.format(i))
    plt.show()
#%%
## the Siemens output scatter is normalized, so we need to apply the detection efficiencies
scatter_3D_unnormalized = stir.ProjDataInMemory(scatter_3D_normalized)

de = stir.BinNormalisationFromProjData(det_eff)
de.set_up(scatter_3D_unnormalized.get_exam_info(), scatter_3D_unnormalized.get_proj_data_info()) #, randoms.get_proj_data_info())
de.apply(scatter_3D_unnormalized)

scatter_3D_unnormalized.write_to_file(os.path.join(STIR_output_folder,scatter_3D_unnorm_filename))
#%%
scatter_3D_unnormalized_arr = stirextra.to_numpy(scatter_3D_unnormalized)
for i in range(33):
    plt.figure()
    vpf.plot_2d_image([1,1,1],scatter_3D_unnormalized_arr[i, central_slice,:,:],'scatter, unnormalized, TOF bin {}'.format(i))
    plt.show()

#!!! Question 3: What do we expect to see if we check as question 1?
#%%
###################### ADDITIVE TERM ############################
#### the additive term is what is added to the projected estimate, before any multiplicative factors
#### are applied. Therefore, we need to normalize randoms and add (normalized) scatter to it.
add_sino = stir.ProjDataInMemory(prompts_from_e7) #randoms)
add_sino.fill(randoms_arr.flat)

# normalise
norm_projdata = stir.ProjData.read_from_file(os.path.join(STIR_output_folder,norm_filename_fSTIR))
norm = stir.BinNormalisationFromProjData(norm_projdata)
norm.set_up(add_sino.get_exam_info(), add_sino.get_proj_data_info()) #, randoms.get_proj_data_info())
norm.apply(add_sino)

# normalised randoms + scatter
add_sino.sapyb(1, scatter_3D_normalized, 1)

#%%
###################### ATTENUATION CORRECTION FACTORS ############################
# to get the correct additive term, we need to apply attenuation correction.
# we'll use the ones from the e7tools here.
vpf.change_datafilename_in_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, prompts_header_filename, attenuation_corr_factor_data_filename)
vpf.change_datatype_in_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, 'float', 4)
vpf.remove_scan_data_lines_from_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)
vpf.remove_IMGDATADESC_lines_from_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)
vpf.remove_data_offset(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)
vpf.remove_tof_dimension(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)

#%%
acf_sino_nonTOF = stir.ProjData.read_from_file(attenuation_corr_factor_header_to_read_withSTIR)
acf_sino_nonTOF = stir.ProjDataInMemory(acf_sino_nonTOF)
if apply_DOI_adaption: vpf.DOI_adaption(acf_sino_nonTOF, 10)

#%%
#### expand to TOF as normalization data is TOF
acf_sino = stir.ProjDataInMemory(prompts_from_e7)
ai_arr = stirextra.to_numpy(acf_sino_nonTOF)
expanded_arr = np.repeat(ai_arr, 33, axis=0)
acf_sino.fill(expanded_arr.flat)

#%%
###################### ATTENUATION FACTORS ############################
afs = np.divide(1,expanded_arr)
for i in range(33):
    plt.figure()
    vpf.plot_2d_image([1,1,1],expanded_arr[i, central_slice,:,:],'acfs, TOF bin {}'.format(i))
    plt.show()

#%%
AC_correction = stir.BinNormalisationFromProjData(acf_sino)
AC_correction.set_up(add_sino.get_exam_info(), add_sino.get_proj_data_info())
AC_correction.apply(add_sino)

add_sino.write_to_file(os.path.join(STIR_output_folder,additive_term_filename_fSTIR))

#%%
# let's see what it looks like
additive_term_arr = stirextra.to_numpy(add_sino)

for i in range(33):
    plt.figure()
    vpf.plot_2d_image([1,1,1],additive_term_arr[i, central_slice,:,:],'additive term, TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'additive_term_sino.png'), transparent=False, facecolor='w')
    plt.show()

#%%
######### now let's write out the multiplicative factors f. STIR
multi_factors_STIR = stir.ProjDataInMemory(norm_sino)
AC_correction = stir.BinNormalisationFromProjData(acf_sino)
AC_correction.set_up(multi_factors_STIR.get_exam_info(), multi_factors_STIR.get_proj_data_info())
AC_correction.apply(multi_factors_STIR)

multi_factors_STIR.write_to_file(os.path.join(STIR_output_folder,multi_term_filename_fSTIR))
#%%
##### to compare the additive term with the acquisition data, we need to
##### pre-correct the prompts with the multiplicative factors
prompts_precorr_f_multi_fact = stir.ProjDataInMemory(prompts_from_e7)
AC_correction = stir.BinNormalisationFromProjData(acf_sino)
AC_correction.set_up(prompts_precorr_f_multi_fact.get_exam_info(), prompts_precorr_f_multi_fact.get_proj_data_info())
AC_correction.apply(prompts_precorr_f_multi_fact)

norm = stir.BinNormalisationFromProjData(norm_projdata)
norm.set_up(prompts_precorr_f_multi_fact.get_exam_info(), prompts_precorr_f_multi_fact.get_proj_data_info())
norm.apply(prompts_precorr_f_multi_fact)

#%%
#### PLOT ADDITIVE TERM
#### draw line-profiles to check if all's correct
prompts_precorr_arr = stirextra.to_numpy(prompts_precorr_f_multi_fact)
additive_term_arr = stirextra.to_numpy(add_sino)

fig, ax = plt.subplots(figsize = (8,6))

ax.plot(np.mean(prompts_precorr_arr[TOF_bin, central_slice-thickness_half:central_slice+thickness_half, 0, :], axis=(0)), label='Prompts, pre-corrected f. multi. factors')
ax.plot(np.mean(additive_term_arr[TOF_bin, central_slice-thickness_half:central_slice+thickness_half, 0, :], axis=(0)), label='additive term')

ax.set_xlabel('Radial distance (bin)')
ax.set_ylabel('total counts')
ax.set_title('TOF bin:' + str(TOF_bin))
ax.legend()
plt.tight_layout()
plt.suptitle('Lineprofiles - Avg over 10 central slices')
plt.savefig(os.path.join(STIR_output_folder,'additive_term_lineprofiles.png'), transparent=False, facecolor='w')
plt.tight_layout()

# %%
#### PLOT BACKGROUND TERM
#### draw line-profiles to check if all's correct
prompts_arr = stirextra.to_numpy(prompts_from_e7)
BG_arr = scatter_3D_unnormalized_arr + randoms_arr

#%%
fig, ax = plt.subplots(figsize = (8,6))

ax.plot(np.mean(prompts_arr[TOF_bin, central_slice-thickness_half:central_slice+thickness_half, 0, :], axis=(0)), label='Prompts')
ax.plot(np.mean(BG_arr[TOF_bin, central_slice-thickness_half:central_slice+thickness_half, 0, :], axis=(0)), label='BG term')

ax.set_xlabel('Radial distance (bin)')
ax.set_ylabel('total counts')
ax.set_title('TOF bin:' + str(TOF_bin))
ax.legend()
plt.tight_layout()
plt.suptitle('Lineprofiles - Avg over 10 central slices')
plt.savefig(os.path.join(STIR_output_folder,'BG_term_lineprofiles.png'), transparent=False, facecolor='w')
plt.tight_layout()
# %%
# Answer_question_1
prompts_arr = stirextra.to_numpy(prompts_from_e7)
print(np.sum(prompts_arr))
# Check the nonTOF prompt sinograms first
prompt_nonTOF = np.sum(prompts_arr, axis=0)
plt.imshow(prompt_nonTOF[0:644,0,:])
plt.clim([0,np.max(prompt_nonTOF)/10])

#%%
# Answer_question_2
for i in range(33):
    plt.figure()
    vpf.plot_2d_image([1,1,1],norm_sino_arr[i, central_slice,:,:],'norm TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'norm_sino.png'), transparent=False, facecolor='w')
    plt.show()
