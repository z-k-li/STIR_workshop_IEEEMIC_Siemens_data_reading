#%%
import stir
import stirextra
import os
import nibabel as nib
'''
User input value
'''
PATH 					= "/Users/vietdao/source/STIR_workshop_IEEEMIC_Siemens_data_reading/Hoffman_phantom_raw_data_for_workshop"
os.chdir(PATH)
image_template_filename = os.path.join("umap_00_python_recon.hv")
num_subset 				= 1
num_subiteration 		= 4
sens_prefix				= "sens"
out_filename_prefix		= "hoffman"
DEBUG 					= True
#%%
'''
EMML reconstruction goes as:

	x_{n+1} = x_{n} * (  X^T( y / (X*x_n+s) ) / X^T*1)
 
where:
	- y is the data.
	- x_{n+1} is the n+1 interation.
 	- X is the system matrix / forward projection (including multiplicative factors).
	- s is the additive term (scatter + randoms).
	- A^T*1 is the sensitivity.

We can also absorb the attenuation into the system matrix meaning:

x_{n+1} = x_{n} * (  (X^TA^T)*( y / ( (A*X) * x_n + A^{-1}*s) ) / (X^T*A^T)*1)

where:
	- A is the attenuation.

hence:
	- A*X is the multiplicative factor.
	- A^{-1}*s is the additive factor.

'''


add_projdata_filename 		= "./processing/additive_term.hs"
mult_projdata_filename 		= "./processing/mult_factors_forSTIR.hs"
measured_projdata_filename	= "./processing/prompts.hs"

f"stir_math -s --including-first --times-scalar 0 --add-scalar 1 ones.hs point_source_f1g1d0b0.hs"
f"stir_math -s --including-first --times-scalar 0 zeros.hs point_source_f1g1d0b0.hs"

#%%

import matplotlib.pyplot as plt
import stir, stirextra
import numpy as np

def plot_projdata_segment(projdata, seg_num=0, cmap="gray_r", filename = ""):
	"""
	Plot a sinogram of one segment from STIR ProjData, summed over axial positions.

	Parameters
	----------
	projdata : stir.ProjData
		The projection data object.
	seg_num : int
		Segment number to extract (default = 0).
	cmap : str
		Colormap for imshow (default = "gray_r").
	"""
	seg = projdata.get_segment_by_sinogram(seg_num)
	seg_np = stirextra.to_numpy(seg)   # shape: (n_axial, n_view, n_tang)

	print(f"Segment {seg_num} shape: {seg_np.shape}")

	# Sum over axial positions → shape (n_view, n_tang)
	seg_sum = np.sum(seg_np, axis=0)

	# Plot
	plt.figure(figsize=(7, 6))
	im = plt.imshow(seg_sum, cmap=cmap, aspect="auto")
	plt.title(f"{filename} seg {seg_num}, summed over axial positions")
	plt.xlabel("Tangential position")
	plt.ylabel("View number")
	plt.colorbar(im, fraction=0.046, pad=0.04, label="Sum counts")
	plt.tight_layout()
	plt.show()

# Example usage:
# plot_projdata_segment(projdata, seg_num=0)


import stir
import stirextra
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def extract_profile(image: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], num: int = None):
	"""Extract a line profile from 2D image between points p0=(x,y), p1=(x,y)."""
	x0, y0 = p0
	x1, y1 = p1
	length = int(np.hypot(x1 - x0, y1 - y0))
	if num is None:
		num = length
	x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
	coords = np.vstack((y, x))  # row=y, col=x
	return map_coordinates(image, coords, order=1, mode="reflect")

def compare_projdata_with_profiles(projdata1, projdata2,
								   seg_num: int = 0,
								   cmap: str = "gray_r",
								   profiles: list[tuple[tuple[int, int], tuple[int, int]]] = None,
								   filename1: str = "ProjData1",
								   filename2: str = "ProjData2"):
	"""
	Compare two STIR projection data objects for a given segment.

	Shows both sinograms side by side with shared colorbar,
	and below a single row with line profiles from both datasets.
	"""

	# --- Load and sum segment ---
	seg1 = projdata1.get_segment_by_sinogram(seg_num)
	seg2 = projdata2.get_segment_by_sinogram(seg_num)

	seg1_np = np.sum(stirextra.to_numpy(seg1), axis=0)  # (views, tang)
	seg2_np = np.sum(stirextra.to_numpy(seg2), axis=0)

	print(f"Segment {seg_num} shapes: {seg1_np.shape}, {seg2_np.shape}")

	# --- Shared color scale ---
	vmin = min(seg1_np.min(), seg2_np.min())
	vmax = max(seg1_np.max(), seg2_np.max())

	# Layout: 2 cols top row, 1 col bottom row
	fig = plt.figure(figsize=(12, 10))
	gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1])
	axprof = fig.add_subplot(gs[1, :])  # spans both columns

	# --- Plot sinograms ---
	im1 = ax1.imshow(seg1_np, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
	im2 = ax2.imshow(seg2_np, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

	ax1.set_title(f"{filename1} seg {seg_num}")
	ax2.set_title(f"{filename2} seg {seg_num}")
	for ax in (ax1, ax2):
		ax.set_xlabel("Tangential position")
		ax.set_ylabel("View number")

	from mpl_toolkits.axes_grid1 import make_axes_locatable

	# Place colorbar next to the right-hand sinogram, aligned in height
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.1)
	cbar = fig.colorbar(im1, cax=cax)
	cbar.set_label("Sum counts")



	# --- Line profiles ---
	if profiles:
		for (x0, y0), (x1, y1) in profiles:
			# Overlay on both images
			ax1.plot([x0, x1], [y0, y1], "r-", lw=1.5)
			ax2.plot([x0, x1], [y0, y1], "r-", lw=1.5)

			# Extract profiles
			prof1 = extract_profile(seg1_np, (x0, y0), (x1, y1))
			prof2 = extract_profile(seg2_np, (x0, y0), (x1, y1))

			# Plot both in the shared profile axis
			axprof.plot(prof1, label=f"{filename1} {((x0,y0),(x1,y1))}")
			axprof.plot(prof2, label=f"{filename2} {((x0,y0),(x1,y1))}")

	axprof.set_title("Line profiles")
	axprof.set_xlabel("Profile sample index")
	axprof.set_ylabel("Counts")
	axprof.legend()

	plt.tight_layout()
	plt.show()



#%%
# --- load geometry and data ---
projdata 					= stir.ProjData.read_from_file(measured_projdata_filename)
projinfo 					= projdata.get_proj_data_info()
exam_info 					= stir.ExamInfo()
out_projdata				= stir.ProjDataInMemory(exam_info, projinfo)  # or your template header
out_image 					= stir.FloatVoxelsOnCartesianGrid.read_from_file(image_template_filename)
image_template 				= stir.FloatVoxelsOnCartesianGrid.read_from_file(image_template_filename)
voxel_size_stir 			= image_template.get_voxel_size()
#%%
# Make or load an image on the correct grid:
# (example creates an empty image matching the projdata's default FOV)
old_img 	= stir.FloatVoxelsOnCartesianGrid.read_from_file(image_template_filename)
old_img.fill(1.0)  # toy test image
sens = [np.ones_like(stirextra.to_numpy(old_img)) for _ in range(num_subset)]

#%% ========================== Setup projector as parallelproj ===========================
# --- construct the Parallelproj projector pair ---
pair = stir.ProjectorByBinPairUsingParallelproj()     # registered name: "Parallelproj"
pair.set_up(projinfo, old_img)                        # must be called before use
fwd = pair.get_forward_projector()
bwd = pair.get_back_projector()

#%%
plot_projdata_segment(projdata, 0, filename="measured")

#%% =========================== calculate sensitivity =====================================
estimate = stir.ProjDataInMemory(exam_info, projinfo)
estimate.fill(1.0)
add_projdata = stir.ProjDataInMemory.read_from_file(add_projdata_filename)
new_img = old_img.get_empty_copy()
new_img.fill(0)
for subset_num in range(num_subset):
	bwd.back_project(new_img, estimate, subset_num, num_subset)
	new_img.write_to_file(f"sens_{subset_num}.hv")
	sens[subset_num] += stirextra.to_numpy(new_img)

#%%

for subset_num in range(num_subset):
	origin = image_template.get_origin()
	affine = np.diag([voxel_size_stir.x(), voxel_size_stir.y(), voxel_size_stir.z(), 1.0])
	affine[:3, 3] = [origin.x(), origin.y(), origin.z()]

	import nibabel as nib
	new_img_np_xyz = np.transpose(sens[subset_num], (2, 1, 0))
	nifti_img = nib.Nifti1Image(new_img_np_xyz, affine)
	nib.save(nifti_img, f"sens_{subset_num}.nii")
#%%
# =========================== forward projection =========================================
estimate = stir.ProjDataInMemory(exam_info, projinfo)
estimate.fill(0.0)
quotient = stir.ProjDataInMemory(exam_info, projinfo)
quotient.fill(0.0)
old_img.fill(1)
num_subiter = 1
while num_subiter <= num_subiteration:
	print(f"Performing Reconstruction for Iteration: {num_subiter}")
	for subset_num in range(num_subset):
		#estimate.fill(0.0)
		# optional: choose subsets (here whole data as one subset)
		# =========================== forward projection =========================================
		fwd.forward_project(estimate, old_img, subset_num, num_subset)

		# ======================== Plot forward project image ==================================
		plot_projdata_segment(estimate, 0, filename="estimates")

		# ======================== calculate value for back projection =========================
		# 'estimate' now contains y_hat = P x  (actually Parallelproj’s Joseph projector)
		# If you have additive terms or want to compare to measured, do it here.

		# --- backprojection of a ratio r = y / A*(X*x_{n}) + A^{-1}*s ---
		numerator 			= stirextra.to_numpy(projdata)
		estimated_np   		= stirextra.to_numpy(estimate)
		#add_factor_np 		= stirextra.to_numpy(add_projdata)
		#mult_projdata_np 	= stirextra.to_numpy(mult_projdata)
		denominator 		= estimated_np
		# Safe division: ratio = projdata / estimate, set 0 where denominator == 0
		quotient_np = np.ones_like(numerator, dtype=np.float32)                      
		np.divide(numerator, denominator, out=quotient_np, where=denominator != 0)   # safe divide
		quotient.fill(quotient_np.flat)
		plot_projdata_segment(quotient, 0, filename="quotient")
	
		# ======================== Back Projection ==============================================
		new_img = old_img.get_empty_copy()
		new_img.fill(0)
		bwd.back_project(new_img, quotient, subset_num, num_subset)


		# ================= divide by sensitivity and save as output ============================
		# Convert STIR image to NumPy
		y_back = stirextra.to_numpy(new_img)
		new_img_np = stirextra.to_numpy(old_img)*np.divide(y_back, sens[subset_num], out=np.zeros_like(y_back, dtype=np.float32), where=sens[subset_num]!=0)
		# Copy values back into STIR image
		old_img.fill(new_img_np.flat)
		# Save
		old_img.write_to_file(f"{out_filename_prefix}_{num_subiter}.hv")

		new_img_np_xyz = np.transpose(new_img_np, (2, 1, 0))
		origin = image_template.get_origin()
		affine = np.diag([voxel_size_stir.x(), voxel_size_stir.y(), voxel_size_stir.z(), 1.0])
		affine[:3, 3] = [origin.x(), origin.y(), origin.z()]
  
		nifti_img = nib.Nifti1Image(new_img_np_xyz, affine)
		nib.save(nifti_img, f"{out_filename_prefix}_{num_subiter}.nii")
		num_subiter +=1
  
  
		
# %%
