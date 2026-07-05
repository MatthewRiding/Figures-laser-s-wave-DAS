# %%
import numpy as np
import matplotlib.pyplot as plt

from functions.moddirectivitiesriding import (get_generation_directivity_sv_riding,
                                              get_detection_directivity_sv_riding,
                                              get_reflection_coefficient_sv_sv_riding)
from functions.modreflectioncoefficients import (calculate_r_sv_sv_auld,
                                                 calculate_r_sv_l_auld,
                                                 calculate_theta_l)
from functions.moddectectiondirectivities import u_z_over_u_shear_dankbaar
from functions.modmyplotfuncs import (G_SV_plot,
                                      R_SV_SV_plot,
                                      R_SV_L_plot,
                                      D_SV_plot,
                                      axis_label_line_plot,
                                      theta_l_plot,
                                      G_D_line_plot,
                                      G_R_D_line_plot,
                                      angle_line_label_combined,
                                      invisible_plots_for_legend,
                                      set_mpl_fonts_Times_New_Roman,
                                      plot_inset_ray_diagram,
                                      plot_inset_ray_diagram_combined)
from corevariables.modjasa import (width_fig_one_column_jasa_in,
                                   fontsize_labels_pt)

# %%

# IPython magic:
%load_ext autoreload
%autoreload 2

# Define parameters:

# Tungsten:
# c_l_mpers = 5182
# c_t_mpers = 2870
# material_string = 'Tungsten'

# Aluminium:
c_l_mpers = 6373
c_t_mpers = 3130
material_string = 'Aluminium'

kappa = c_l_mpers / c_t_mpers

# Critical angle:
theta_crit_deg = np.rad2deg(np.asin(c_t_mpers/c_l_mpers))

# Array of angles to evaluate:
n_angles = (90 * 4) + 1
theta_array_deg = np.linspace(0, 90, n_angles)
theta_array_rad = np.deg2rad(theta_array_deg)

# Calculate thermoelastic T-wave directivity:
# d_T_thermoelastic:
G_SV_Riding = get_generation_directivity_sv_riding(kappa, theta_array_rad)

# Calculate reflection coefficients:
# R_SV_SV:
R_SV_SV_Riding = get_reflection_coefficient_sv_sv_riding(
    kappa, theta_array_rad)
R_SV_SV_Auld = calculate_r_sv_sv_auld(kappa, theta_array_rad)
# R_SV_L:
R_SV_L_Auld = calculate_r_sv_l_auld(kappa, theta_array_rad)

# Calculate theta L, the exit angle of the mode-converted L-wave,
# using Snell's law:
theta_L_rad_complex = calculate_theta_l(theta_array_rad, kappa)

# Calculate the 'detection directivity' for a shear vertical wave
# (normal component of displacement):
D_SV_Riding = get_detection_directivity_sv_riding(kappa, theta_array_rad)
D_SV_Dankbaar = u_z_over_u_shear_dankbaar(c_l_mpers, c_t_mpers,
                                          theta_array_rad)

# Combined effects:
# Generation & detection:
G_D = G_SV_Riding * D_SV_Riding
# Generation, relfection & detection:
G_R_D = G_SV_Riding * R_SV_SV_Auld * D_SV_Riding

# Plotting parameters:
fig_width_inches = width_fig_one_column_jasa_in
fig_height_inches = 6 * 0.9
fig_dpi = 300
set_mpl_fonts_Times_New_Roman()
dir_max_displacement = 4
r_tl_max = 2

# %%
# Create figure:
(fig,
 (ax_1,
  ax_2,
  ax_3,
  ax_4,
  ax_5)) = plt.subplots(5, 1, layout='constrained',
                        figsize=(fig_width_inches,
                                 fig_height_inches),
                        dpi=fig_dpi)

# Use plotting functions on axes:

# Plot generation directivity:
G_SV_plot(ax_1, G_SV_Riding, theta_array_deg, theta_crit_deg)
ax_1.set_ylim(-dir_max_displacement, dir_max_displacement)
axis_label_line_plot(ax_1, '(a)')

# Plot reflection coefficient:
R_SV_SV_plot(ax_2, R_SV_SV_Auld, theta_array_deg, theta_crit_deg,
             label_phase=False)
axis_label_line_plot(ax_2, '(b)')

# Plot detection directivity:
D_SV_plot(ax_3, D_SV_Riding, theta_array_deg, theta_crit_deg,
          label_phase=True)
axis_label_line_plot(ax_3, '(c)')

# Plot combined G*D:
G_D_line_plot(ax_4, G_D, theta_array_deg, theta_crit_deg)
axis_label_line_plot(ax_4, '(d)')

# Plot combined G*R*D:
G_R_D_line_plot(ax_5, G_R_D, theta_array_deg, theta_crit_deg)
axis_label_line_plot(ax_5, '(e)')
angle_line_label_combined(ax_5, theta_crit_deg, r'$\theta$*')
angle_line_label_combined(ax_5, 45, r'45°')


# Figure title:
mpers_string = r'$\mathrm{ms}^{-1}$'
title_string = (f'{material_string}: '
                + r'$c_\mathrm{L}=$' f'{c_l_mpers}'
                + r'$\,$' + mpers_string + ', '
                + r'$c_\mathrm{S}=$' f'{c_t_mpers}'
                + r'$\,$' + mpers_string)
(real_handle,
 imag_handle,
 abs_handle,
 phase_handle) = invisible_plots_for_legend(ax_1)

fig.suptitle(title_string, fontsize=fontsize_labels_pt, y=1.09)
fig.legend(loc=(0.13, 0.915), fontsize=fontsize_labels_pt,
           handles=[real_handle, imag_handle, abs_handle, phase_handle],
           ncols=4)
fig.supxlabel(r'Angle of SV wave incidence $\theta$ (°)',
              fontsize=fontsize_labels_pt, y=-0.025, x=0.54)

plot_inset_ray_diagram(ax_1, 'g')
plot_inset_ray_diagram(ax_2, 'r')
plot_inset_ray_diagram(ax_3, 'd')

plot_inset_ray_diagram_combined(ax_4, 'gd')
plot_inset_ray_diagram_combined(ax_5, 'grd')

# %%

# Saving:
# Directivities and Reflectivities figure:
# fig.savefig(r'Figure folders\Fig directivities and reflections\directivities_separate_and_combined_JASA.png',
#             format='png', dpi=200, bbox_inches='tight')
# fig.savefig(r'Figure folders\Fig directivities and reflections\directivities_separate_and_combined_JASA.pdf',
#             format='pdf', dpi=fig_dpi, bbox_inches='tight')

# %%
