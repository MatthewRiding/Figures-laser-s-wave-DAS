from functions.modloadnmoperiodicampsfrommatfile import (
    load_nmo_periodic_amps_from_mat_file)
from classdefs.modparamsets import material_param_set, sdh_fmc_param_set


# Load the data:
fmc_2d_raw_v = load_nmo_periodic_amps_from_mat_file(
    r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_02_05_Matt_SDH_polarity_Al_FMC\Scan 2 long\FMC_v7.mat")

# Create a params dict to fill and return:
params_al_sdh = sdh_fmc_param_set(fmc_2d=fmc_2d_raw_v,
                                  pitch_mm=0.2,
                                  t_min_us=-0.5,
                                  t_max_us=9.49,
                                  x_sdh_mm=9.75,
                                  z_sdh_mm=5,
                                  radius_sdh_mm=0.5,
                                  rise_time_us=0.06)

material_al_sdh = material_param_set(c_L_mpers=6475,
                                     c_T_mpers=3170)
