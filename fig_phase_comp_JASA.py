import matplotlib.pyplot as plt

from corevariables.modjasa import (fontsize_labels_pt,
                                   width_fig_two_column_jasa_in)
from functions.modmyplotfuncs import set_mpl_fonts_Times_New_Roman
from functions.modplotphasecompsdh import phase_comp_plots_sdh
from functions.modplotphasecomphfs import phase_comp_plots_hfs


# IPython magic:
# %load_ext autoreload
# %autoreload 2

# Figure parameters:
fig_width_inches = width_fig_two_column_jasa_in * 0.8
fig_height_inches = 4/3 * 1.8
fig_dpi = 200
set_mpl_fonts_Times_New_Roman()

# Set displacement colormap limits:
# SDH:
v_min_SDH_nm = -0.03
v_max_SDH_nm = 0.03
cbar_ticks_SDH_pm = [-30, 0, 30]

# HFS:
v_min_HFS_nm = -0.2
v_max_HFS_nm = 0.2
cbar_ticks_HFS_pm = [-200, 0, 200]

# Set up mosaic figure:
kwargs_gridspec = dict(width_ratios=[1, 1.4])
fig, dict_axs = plt.subplot_mosaic([['t_left', 't_right'],
                                    ['b_left', 'b_right']],
                                   gridspec_kw=kwargs_gridspec,
                                   figsize=(fig_width_inches,
                                            fig_height_inches),
                                   layout="constrained",
                                   dpi=fig_dpi)

# Name axes:
ax_colormap_SDH = dict_axs['t_left']
ax_wiggles_SDH = dict_axs['t_right']
ax_colormap_HFS = dict_axs['b_left']
ax_wiggles_HFS = dict_axs['b_right']

# Plot SDH plots:
im_SDH = phase_comp_plots_sdh(ax_colormap_SDH, ax_wiggles_SDH,
                              v_min_SDH_nm, v_max_SDH_nm)

# Plot HFS plots:
im_HFS = phase_comp_plots_hfs(ax_colormap_HFS, ax_wiggles_HFS,
                              v_min_HFS_nm, v_max_HFS_nm)

# Add colorbars for both colormaps:
# SDH:
cbar_sdh = fig.colorbar(im_SDH, ax=ax_colormap_SDH, ticks=cbar_ticks_SDH_pm,
                        pad=0.01, aspect=10)
# cbar_sdh.set_label('Displacement $u_z$ (pm)', size=fontsize_labels_pt)
cbar_sdh.ax.tick_params(labelsize=fontsize_labels_pt)
# HFS:
cbar_hfs = fig.colorbar(im_HFS, ax=ax_colormap_HFS, ticks=cbar_ticks_HFS_pm,
                        pad=0.01, aspect=10)
# cbar_hfs.set_label('Displacement $u_z$ (pm)', size=fontsize_labels_pt)
cbar_hfs.ax.tick_params(labelsize=fontsize_labels_pt)
# Colorbar label above:
fig.text(0.435, 0.905, 'Displacement\n$u_z$ (pm)', fontsize=fontsize_labels_pt,
         ha='left')

# %% Saving:
# fig.savefig(r'Figure folders\Fig DAS phase comp\Fig_phase_comp_JASA.pdf',
#             format='pdf', dpi=600)

# fig.savefig(r'Figure folders\Fig DAS phase comp\Fig_phase_comp_JASA.png',
#             format='png', dpi=200)

# %%
