from functions.modloadnmoperiodicampsfrommatfile import load_nmo_periodic_amps_from_mat_file

from classdefs.modparamsets import nmo_scan_param_set


# Load the data:
b_scan_ds_V_raw = load_nmo_periodic_amps_from_mat_file(
    r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_04_04_Matt_2mm_Al\Scan 1 NMO\b_scan_array.mat")

# Pass NMO scan parameters:
pitch_mm = 0.05
t_min_us = -0.25
t_max_us = 4.745
rise_time_us = 0.06
thickness_mm = 2

params_al_nmo = nmo_scan_param_set(b_scan_array_2d=b_scan_ds_V_raw,
                                   pitch_mm=pitch_mm,
                                   t_min_us=t_min_us,
                                   t_max_us=t_max_us,
                                   rise_time_us=rise_time_us,
                                   thickness_b_mm=thickness_mm)
