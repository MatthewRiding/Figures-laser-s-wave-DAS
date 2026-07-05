# %%
import numpy as np
import matplotlib.pyplot as plt

from functions.moddirectivitiesriding import get_detection_directivity_l_pilant
from classdefs.modparamsets import material_param_set


material_al = material_param_set(c_L_mpers=6475,
                                 c_T_mpers=3170)

theta_array_deg = np.linspace(0, 90, 90)

d_l_all_theta = get_detection_directivity_l_pilant(material_al.kappa,
                                                   np.deg2rad(theta_array_deg))

plt.plot(d_l_all_theta)

# %%
