import numpy as np
from classdefs.modparamsets import (material_param_set,
                                    sdh_mss_param_set)


# Load the displacement data:
npz = np.load('displacement_data/displacement_data.npz')
displacements_exp_1_MWS_filtered_pm = npz['experiment_1_MWS_filtered_pm']

params_al_sdh = sdh_mss_param_set(
    displacements_array_2d_pm=displacements_exp_1_MWS_filtered_pm,
    pitch_mm=0.2,
    t_min_us=-0.5,
    t_max_us=9.49,
    rise_time_us=0.06,
    x_sdh_mm=9.75,
    z_sdh_mm=5,
    radius_sdh_mm=0.5)

material_al_sdh = material_param_set(c_L_mpers=6475,
                                     c_T_mpers=3170)
