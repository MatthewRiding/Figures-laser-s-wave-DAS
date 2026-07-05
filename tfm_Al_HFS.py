import numpy as np


from classdefs.modfullmatrix import FullMatrixLinearPeriodic
from classdefs.modtfmconstructor import TFMConstructor
from classdefs.modfilterspec import FilterSpec
from classdefs.modmaskspec import MaskSpec
from classdefs.modfakesignal import FakeSignal
from classdefs.modlistedtfmimage import ListedTFMImage
from corevariables.modaluminium import material_Al
from classdefs.modmaterial import Material
from corevariables.modwavetypes import (dict_wave_types_send,
                                        dict_wave_types_receive)
from corevariables.modmaskbehaviours import dict_mask_behaviours
from functions.modloadfmclpfrommatfile import load_fmclp_from_mat_file
from functions.modconvertvtonm import convert_quartet_v_to_nm
from functions.modcomputetfmcomplex import compute_tfm_complex
from functions.modextractpcm import extract_pcm
from functions.modgeneratedelaymatrix import generate_delay_matrix
from functions.moddetrendfmc3d import detrend_fmc_3d
from functions.modfilterfmc3dbutter import bandpass_fmc3d_butter
from functions.modbuildtfmgrid import build_tfm_grid
from functions.modbuildxelements import build_x_elements_m

# Define parameters:
grid_size_x_mm = 55
grid_size_z_mm = 15
pitch_mm = 0.2
n_pixels_z = 100

# Wave speeds:
c_sv_nemo_mpers = 3160
c_l_nemo_mpers = 6420
c_lsaw_nemo_mpers = 6270
material_al_hfs_nemo = Material(c_l_nemo_mpers, c_sv_nemo_mpers,
                                c_lsaw_nemo_mpers)


# Generate fake image array:
def get_z_image():
    x_grid_mm, z_grid_mm = build_tfm_grid(grid_size_x_mm, grid_size_z_mm,
                                          n_pixels_z)
    return z_grid_mm


def get_max_horizontal_specular_angle_all_pixels_degrees():
    # Get pixel (x,z) coordinates:
    x_grid_m, z_grid_m = build_tfm_grid(grid_size_x_mm, grid_size_z_mm,
                                        n_pixels_z)
    # Max specular reflection angle assuming a horizontal reflector
    # is only non-zero beneath the array aperture.
    # Calculate this angle for each pixel:
    # Get array element locations:
    n_tx = 251
    x_elements_m = build_x_elements_m(n_tx, pitch_mm)
    # Find max element x coordinate:
    x_element_max_m = np.max(x_elements_m)
    # Right-angled triangle:
    opposite_m = x_element_max_m - np.abs(x_grid_m)
    angle_max_specular_horizontal_degrees = np.rad2deg(np.atan2(opposite_m,
                                                                z_grid_m))
    return angle_max_specular_horizontal_degrees, x_grid_m, z_grid_m


# Import data:


def import_displacements_nm():
    displacements_3d_v = load_fmclp_from_mat_file(
        r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_10_15_Matt_HFS_FMC\Scan 2 1024 av wfm 29\FMC_v7.mat")
    displacements_3d_nm = convert_quartet_v_to_nm(displacements_3d_v)
    # De-trend:
    displacements_3d_detrend_nm = detrend_fmc_3d(displacements_3d_nm)

    return displacements_3d_detrend_nm


def import_full_matrix():
    # Import de-trended displacements in nm:
    displacements_3d_detrend_nm = import_displacements_nm()
    # Specify time limits:
    t_min_us = -1
    t_max_us = 18.98
    full_matrix_Al_HFS = FullMatrixLinearPeriodic(displacements_3d_detrend_nm,
                                                  t_min_us, t_max_us)
    return full_matrix_Al_HFS


def import_full_matrix_filtered():
    # Import de-trended displacements in nm:
    displacements_3d_detrend_nm = import_displacements_nm()
    # Specify time limits:
    t_min_us = -1
    t_max_us = 18.98
    full_matrix_Al_HFS = FullMatrixLinearPeriodic(displacements_3d_detrend_nm,
                                                  t_min_us, t_max_us)
    # Apply fourier domain filter:
    full_matrix_Al_HFS.displacements_3d_nm = bandpass_fmc3d_butter(
        displacements_3d_detrend_nm,
        full_matrix_Al_HFS.frequency_sampling_hz,
        butter_order=10,
        band_min_MHz=2.5,
        band_max_MHz=(full_matrix_Al_HFS.frequency_sampling_hz/2.01) / 10**6)

    return full_matrix_Al_HFS


def get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                               string_wave_type_send, string_wave_type_receive,
                               mask_spec_gen=None, mask_spec_det=None):
    filter_spec = FilterSpec(butter_order=10, band_min_mhz=2.5,
                             band_max_mhz=(full_matrix_Al_HFS.frequency_sampling_hz/2.01) / 10**6)
    wave_type_send = dict_wave_types_send[string_wave_type_send]
    wave_type_receive = dict_wave_types_receive[string_wave_type_receive]
    tfm_constructor = TFMConstructor('',
                                     n_elements=full_matrix_Al_HFS.n_elements,
                                     pitch_mm=pitch_mm,
                                     grid_size_x_mm=grid_size_x_mm,
                                     grid_size_z_mm=grid_size_z_mm,
                                     n_pixels_z=n_pixels_z,
                                     material=material_al_hfs_nemo,
                                     wave_type_send=wave_type_send,
                                     wave_type_receive=wave_type_receive,
                                     filter_spec=filter_spec,
                                     mask_spec_gen=mask_spec_gen,
                                     mask_spec_det=mask_spec_det)
    return tfm_constructor


def compute_tfm_ll():
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='L',
                                                 string_wave_type_receive='L')
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='L-L: ')

    return summed_displacement_image_complex_nm


def compute_tfm_ss_0_crit():
    angle_crit_deg = np.rad2deg(material_al_hfs_nemo.critical_angle_radians)
    mask_spec_gen = MaskSpec(
        angle_crit_deg, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(
        angle_crit_deg, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S subcrit: ')

    return summed_displacement_image_complex_nm


def compute_tfm_ss_50_up():
    mask_spec_gen = MaskSpec(50, dict_mask_behaviours['Ignore below'])
    mask_spec_det = MaskSpec(50, dict_mask_behaviours['Ignore below'])
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S 50+: ')

    return summed_displacement_image_complex_nm


def compute_tfm_ss_50_to_60():
    mask_spec_both = MaskSpec((50, 60), dict_mask_behaviours['Ignore outside'])

    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_both,
                                                 mask_spec_det=mask_spec_both)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S 50-60: ')

    return summed_displacement_image_complex_nm


def compute_tfm_SS_no_masks():
    tfm_constructor = get_tfm_constructor_Al_HFS(string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=None,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='L-L: ')

    return summed_displacement_image_complex_nm


def compute_tfm_SS_sub_60():
    mask_spec_gen = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S sub 60: ')

    return summed_displacement_image_complex_nm


def compute_tfm_h_s_no_masks():
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='H',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=None,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='H-S no masks: ')

    return summed_displacement_image_complex_nm


def compute_tfm_h_s_sub_60():
    mask_spec_gen = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_HFS(string_wave_type_send='H',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='H-S: ')

    return summed_displacement_image_complex_nm


def compute_tfm_h_s_gen_supercrit():
    angle_crit_deg = np.rad2deg(material_al_hfs_nemo.critical_angle_radians)
    mask_spec_gen = MaskSpec(
        angle_crit_deg, dict_mask_behaviours['Ignore below'])
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='H',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='H-S: ')

    return summed_displacement_image_complex_nm


def compute_tfm_hybrid_s_no_masks():
    tfm_constructor = get_tfm_constructor_Al_HFS(full_matrix_Al_HFS,
                                                 string_wave_type_send='Shear hybrid',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=None,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='Hybrid-S: ')

    return summed_displacement_image_complex_nm


def compute_tfm_hybrid_s_sub_60():
    mask_spec_gen = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_HFS(string_wave_type_send='Shear hybrid',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_HFS, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='Hybrid-S: ')

    return summed_displacement_image_complex_nm


class PCMAlHFS():
    def __init__(self, fmc_linear_periodic, xz_pixel_m, listed_tfm_image):
        self.x_pixel_m, self.z_pixel_m = xz_pixel_m
        # Compute the delay matrix for this pixel and wave set:
        delay_matrix_s = generate_delay_matrix(self.x_pixel_m, self.z_pixel_m,
                                               listed_tfm_image)
        # Use delay matrix to extract PCM:
        self.pcm_complex_nm = extract_pcm(
            fmc_linear_periodic.displacements_3d_nm,
            fmc_linear_periodic.time_vector_us,
            delay_matrix_s)


def extract_PCMs_Al_HFS():
    # Import full matrix:
    full_matrix_Al_HFS_filtered = import_full_matrix_filtered()

    # Make an instance of the PCMALHFS class for each of the wave sets and
    # their chosen pixels:

    def get_listed_tfm_image_Al_HFS(string_wave_type_send,
                                    string_wave_type_receive):
        listed_tfm_image = ListedTFMImage(
            '', get_tfm_constructor_Al_HFS(full_matrix_Al_HFS_filtered,
                                           string_wave_type_send,
                                           string_wave_type_receive),
            full_matrix_Al_HFS_filtered.n_elements)
        return listed_tfm_image

    pcm_ll = PCMAlHFS(full_matrix_Al_HFS_filtered, (0, 12 * 10**-3),
                      get_listed_tfm_image_Al_HFS('L', 'L'))
    pcm_ss = PCMAlHFS(full_matrix_Al_HFS_filtered, (0, 12 * 10**-3),
                      get_listed_tfm_image_Al_HFS('T', 'T'))
    pcm_hs = PCMAlHFS(full_matrix_Al_HFS_filtered, (0, 12 * 10**-3),
                      get_listed_tfm_image_Al_HFS('H', 'T'))
    pcm_hybrid_s = PCMAlHFS(full_matrix_Al_HFS_filtered,
                            (0 * 10**-3, 12 * 10**-3),
                            get_listed_tfm_image_Al_HFS('Shear hybrid', 'T'))

    return pcm_ll, pcm_ss, pcm_hs, pcm_hybrid_s


if __name__ == '__main__':
    # Import raw full matrix:
    full_matrix_Al_HFS = import_full_matrix()

    # Compute TFM images and save numpy arrays:

    # L-L image:
    summed_displacement_image_complex_nm_LL = compute_tfm_ll()
    np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_HFS_detrend_complex_LL',
            summed_displacement_image_complex_nm_LL)

    # S-S images:
    # Subcrit:
    # summed_displacement_image_complex_nm_s_s_subcrit = compute_tfm_ss_0_crit()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_Nemo_HFS_detrend_complex_S_S_subcrit',
    #         summed_displacement_image_complex_nm_s_s_subcrit)
    # > 50 degrees:
    # summed_displacement_image_complex_nm_s_s_50_up = compute_tfm_ss_50_up()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_HFS_complex_S_S_50_up',
    #         summed_displacement_image_complex_nm_s_s_50_up)
    # 50 <= x <= 60 degrees:
    # summed_displacement_image_complex_nm_s_s_50_to_60 = compute_tfm_ss_50_to_60()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_Nemo_HFS_detrend_complex_S_S_50_to_60',
    #         summed_displacement_image_complex_nm_s_s_50_to_60)
    # No mask:
    # summed_displacement_image_complex_nm_s_s_no_mask = compute_tfm_SS_no_masks()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_HFS_complex_S_S_no_masks',
    #         summed_displacement_image_complex_nm_s_s_no_mask)
    # Sub-60:
    # summed_displacement_image_complex_nm_s_s_sub_60 = compute_tfm_SS_sub_60()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_Nemo_HFS_detrend_complex_S_S_sub_60',
    #         summed_displacement_image_complex_nm_s_s_sub_60)

    # H-S images:
    # No masks:
    # summed_displacement_image_complex_nm_h_s_no_masks = compute_tfm_h_s_no_masks()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_HFS_detrend_complex_H_S_no_masks',
    #         summed_displacement_image_complex_nm_h_s_no_masks)
    # Sub-60:
    # summed_displacement_image_complex_nm_h_s_sub_60 = compute_tfm_h_s_sub_60()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_HFS_complex_H_S_sub_60',
    #         summed_displacement_image_complex_nm_h_s_sub_60)
    # Super crit:
    # summed_displacement_image_complex_nm_h_s_gen_super_crit = compute_tfm_h_s_gen_supercrit()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_Nemo_HFS_detrend_complex_H_S_gen_super_crit',
    #         summed_displacement_image_complex_nm_h_s_gen_super_crit)

    # Hybrid-S image:
    # summed_displacement_image_complex_nm_hybrid_s = compute_tfm_hybrid_s_no_masks()
    # np.save(r'data\tfm_images\Al_HFS\tfm_image_Al_Nemo_HFS_detrend_complex_Hybrid_S_no_masks',
    #         summed_displacement_image_complex_nm_hybrid_s)
