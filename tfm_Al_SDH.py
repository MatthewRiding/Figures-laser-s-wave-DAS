import numpy as np


from classdefs.modfullmatrix import FullMatrixLinearPeriodic
from classdefs.modtfmconstructor import TFMConstructor
from classdefs.modfilterspec import FilterSpec
from classdefs.modmaskspec import MaskSpec
from classdefs.modfakesignal import FakeSignal
from classdefs.modlistedtfmimage import ListedTFMImage
from corevariables.modaluminium import material_Al
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


# Define parameters:
grid_size_x_mm = 10
grid_size_z_mm = 10
pitch_mm = 0.2
n_pixels_z = 200

# Import data:


def import_displacements_nm():
    displacements_3d_v = load_fmclp_from_mat_file(
        r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_02_05_Matt_SDH_polarity_Al_FMC\Scan 2 long\FMC_v7.mat")
    displacements_3d_nm = convert_quartet_v_to_nm(displacements_3d_v)
    # De-trend:
    displacements_3d_detrend_nm = detrend_fmc_3d(displacements_3d_nm)

    return displacements_3d_detrend_nm


def import_full_matrix():
    # Import de-trended displacements in nm:
    displacements_3d_detrend_nm = import_displacements_nm()
    # Specify time limits:
    t_min_us = -0.5
    t_max_us = 9.49
    full_matrix_Al_SDH = FullMatrixLinearPeriodic(displacements_3d_detrend_nm,
                                                  t_min_us, t_max_us)
    return full_matrix_Al_SDH


def import_full_matrix_filtered():
    # Import de-trended displacements in nm:
    displacements_3d_detrend_nm = import_displacements_nm()
    # Specify time limits:
    t_min_us = -0.5
    t_max_us = 9.49
    full_matrix_Al_SDH = FullMatrixLinearPeriodic(displacements_3d_detrend_nm,
                                                  t_min_us, t_max_us)
    # Apply fourier domain filter:
    full_matrix_Al_SDH.displacements_3d_nm = bandpass_fmc3d_butter(
        displacements_3d_detrend_nm,
        full_matrix_Al_SDH.frequency_sampling_hz,
        butter_order=10,
        band_min_MHz=2.5,
        band_max_MHz=(full_matrix_Al_SDH.frequency_sampling_hz/2.01) / 10**6)

    return full_matrix_Al_SDH


def get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                               string_wave_type_send, string_wave_type_receive,
                               mask_spec_gen=None, mask_spec_det=None):
    filter_spec = FilterSpec(butter_order=10, band_min_mhz=2.5,
                             band_max_mhz=(full_matrix_Al_SDH.frequency_sampling_hz/2.01) / 10**6)
    wave_type_send = dict_wave_types_send[string_wave_type_send]
    wave_type_receive = dict_wave_types_receive[string_wave_type_receive]
    tfm_constructor = TFMConstructor('',
                                     n_elements=full_matrix_Al_SDH.n_elements,
                                     pitch_mm=pitch_mm,
                                     grid_size_x_mm=grid_size_x_mm,
                                     grid_size_z_mm=grid_size_z_mm,
                                     n_pixels_z=n_pixels_z,
                                     material=material_Al,
                                     wave_type_send=wave_type_send,
                                     wave_type_receive=wave_type_receive,
                                     filter_spec=filter_spec,
                                     mask_spec_gen=mask_spec_gen,
                                     mask_spec_det=mask_spec_det)
    return tfm_constructor


def compute_tfm_ll():
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='L',
                                                 string_wave_type_receive='L')
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='L-L: ')

    return summed_displacement_image_complex_nm


def compute_tfm_ss_0_crit():
    angle_crit_deg = np.rad2deg(material_Al.critical_angle_radians)
    mask_spec_gen = MaskSpec(angle_crit_deg, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(angle_crit_deg, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S subcrit: ')

    return summed_displacement_image_complex_nm


def compute_tfm_ss_50_up():
    mask_spec_gen = MaskSpec(50, dict_mask_behaviours['Ignore below'])
    mask_spec_det = MaskSpec(50, dict_mask_behaviours['Ignore below'])
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S 50+: ')

    return summed_displacement_image_complex_nm


def compute_tfm_ss_50_to_60():
    mask_spec_both = MaskSpec((50, 60), dict_mask_behaviours['Ignore outside'])

    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_both,
                                                 mask_spec_det=mask_spec_both)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S 50-60: ')

    return summed_displacement_image_complex_nm


def compute_tfm_SS_no_masks():
    tfm_constructor = get_tfm_constructor_Al_SDH(string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=None,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='L-L: ')

    return summed_displacement_image_complex_nm


def compute_tfm_SS_sub_60():
    mask_spec_gen = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='T',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='S-S sub 60: ')

    return summed_displacement_image_complex_nm


def compute_tfm_h_s_no_masks():
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='H',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=None,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='H-S no masks: ')

    return summed_displacement_image_complex_nm


def compute_tfm_h_s_sub_60():
    mask_spec_gen = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_SDH(string_wave_type_send='H',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='H-S: ')

    return summed_displacement_image_complex_nm


def compute_tfm_h_s_gen_supercrit():
    angle_crit_deg = np.rad2deg(material_Al.critical_angle_radians)
    mask_spec_gen = MaskSpec(angle_crit_deg, dict_mask_behaviours['Ignore below'])
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='H',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='H-S: ')

    return summed_displacement_image_complex_nm


def compute_tfm_hybrid_s_no_masks():
    tfm_constructor = get_tfm_constructor_Al_SDH(full_matrix_Al_SDH,
                                                 string_wave_type_send='Shear hybrid',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=None,
                                                 mask_spec_det=None)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='Hybrid-S: ')

    return summed_displacement_image_complex_nm


def compute_tfm_hybrid_s_sub_60():
    mask_spec_gen = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    mask_spec_det = MaskSpec(60, dict_mask_behaviours['Ignore above'])
    tfm_constructor = get_tfm_constructor_Al_SDH(string_wave_type_send='Shear hybrid',
                                                 string_wave_type_receive='T',
                                                 mask_spec_gen=mask_spec_gen,
                                                 mask_spec_det=mask_spec_det)
    (summed_displacement_image_complex_nm,
     displacements_fmc_3d_filtered_nm) = compute_tfm_complex(
         full_matrix_Al_SDH, tfm_constructor, signal_progress=FakeSignal(),
         worker_id='Hybrid-S: ')

    return summed_displacement_image_complex_nm


class PCMAlSDH():
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


def extract_PCMs_Al_SDH():
    # Import full matrix:
    full_matrix_Al_SDH_filtered = import_full_matrix_filtered()

    # Make an instance of the PCM_Al_SDH class for each of the wave sets and
    # their chosen pixels:

    def get_listed_tfm_image_Al_SDH(string_wave_type_send,
                                    string_wave_type_receive):
        listed_tfm_image = ListedTFMImage(
            '', get_tfm_constructor_Al_SDH(full_matrix_Al_SDH_filtered,
                                           string_wave_type_send,
                                           string_wave_type_receive),
            full_matrix_Al_SDH_filtered.n_elements)
        return listed_tfm_image

    pcm_ll = PCMAlSDH(full_matrix_Al_SDH_filtered, (0 * 10**-3, 4.5 * 10**-3),
                      get_listed_tfm_image_Al_SDH('L', 'L'))
    pcm_ss = PCMAlSDH(full_matrix_Al_SDH_filtered, (0 * 10**-3, 4.4 * 10**-3),
                      get_listed_tfm_image_Al_SDH('T', 'T'))
    pcm_hs = PCMAlSDH(full_matrix_Al_SDH_filtered, (0 * 10**-3, 4.4 * 10**-3),
                      get_listed_tfm_image_Al_SDH('H', 'T'))
    pcm_hybrid_s = PCMAlSDH(full_matrix_Al_SDH_filtered, (-0.05 * 10**-3, 4.4 * 10**-3),
                            get_listed_tfm_image_Al_SDH('Shear hybrid', 'T'))

    return pcm_ll, pcm_ss, pcm_hs, pcm_hybrid_s


if __name__ == '__main__':
    # Import raw full matrix:
    full_matrix_Al_SDH = import_full_matrix()

    # Compute TFM images and save numpy arrays:

    # L-L image:
    # summed_displacement_image_complex_nm_LL = compute_tfm_ll()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_LL',
    #         summed_displacement_image_complex_nm_LL)


    # S-S images:
    # Subcrit:
    # summed_displacement_image_complex_nm_s_s_subcrit = compute_tfm_ss_0_crit()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_S_S_subcrit',
    #         summed_displacement_image_complex_nm_s_s_subcrit)
    # > 50 degrees:
    # summed_displacement_image_complex_nm_s_s_50_up = compute_tfm_ss_50_up()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_complex_S_S_50_up',
    #         summed_displacement_image_complex_nm_s_s_50_up)
    # 50 <= x <= 60 degrees:
    # summed_displacement_image_complex_nm_s_s_50_to_60 = compute_tfm_ss_50_to_60()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_S_S_50_to_60',
    #         summed_displacement_image_complex_nm_s_s_50_to_60)
    # No mask:
    # summed_displacement_image_complex_nm_s_s_no_mask = compute_tfm_SS_no_masks()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_complex_S_S_no_masks',
    #         summed_displacement_image_complex_nm_s_s_no_mask)
    # Sub-60:
    # summed_displacement_image_complex_nm_s_s_sub_60 = compute_tfm_SS_sub_60()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_S_S_sub_60',
    #         summed_displacement_image_complex_nm_s_s_sub_60)


    # H-S images:
    # No masks:
    # summed_displacement_image_complex_nm_h_s_no_masks = compute_tfm_h_s_no_masks()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_H_S_no_masks',
    #         summed_displacement_image_complex_nm_h_s_no_masks)
    # Sub-60:
    # summed_displacement_image_complex_nm_h_s_sub_60 = compute_tfm_h_s_sub_60()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_complex_H_S_sub_60',
    #         summed_displacement_image_complex_nm_h_s_sub_60)
    # Super crit:
    # summed_displacement_image_complex_nm_h_s_gen_super_crit = compute_tfm_h_s_gen_supercrit()
    # np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_H_S_gen_super_crit',
    #         summed_displacement_image_complex_nm_h_s_gen_super_crit)


    # Hybrid-S image:
    summed_displacement_image_complex_nm_hybrid_s = compute_tfm_hybrid_s_no_masks()
    np.save(r'data\tfm_images\Al_SDH\tfm_image_Al_SDH_detrend_complex_Hybrid_S_masked_0_60',
            summed_displacement_image_complex_nm_hybrid_s)
