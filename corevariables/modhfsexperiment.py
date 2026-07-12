import numpy as np
from classdefs.modparamsets import nmo_scan_param_set


# Load the displacement data:
npz = np.load('displacement_data/displacement_data.npz')
displacements_exp_2_pm = npz['experiment_2_raw_pm']

# Pass NMO scan parameters:
pitch_mm = 0.05
t_min_us = -0.25
t_max_us = 4.745
rise_time_us = 0.06
thickness_mm = 2

params_al_nmo = nmo_scan_param_set(b_scan_array_2d_pm=displacements_exp_2_pm,
                                   pitch_mm=pitch_mm,
                                   t_min_us=t_min_us,
                                   t_max_us=t_max_us,
                                   rise_time_us=rise_time_us,
                                   thickness_b_mm=thickness_mm)
