from classdefs.modwavetype import WaveType

from functions.delayfuncstfm.moddelayfuncstfm import (calculate_travel_times_tfm_l_direct_s,
                                                      calculate_travel_times_tfm_t_direct_s,
                                                      calculate_travel_times_tfm_head_wave_flat_surface_s,
                                                      calculate_travel_times_tfm_shear_hybrid_s)
from functions.delayfuncscheops.moddelayfunctionscheops import (calculate_travel_times_single_pixel_all_el_l_s,
                                                                calculate_travel_times_single_pixel_all_el_t_s,
                                                                calculate_travel_times_single_pixel_all_el_h_s,
                                                                calculate_travel_times_single_pixel_all_el_shear_hybrid_s)
from functions.delayfuncsisochron.moddelayfuncsisocron import (calculate_travel_times_all_pixels_single_el_l_direct_s,
                                                               calculate_travel_times_all_pixels_single_el_t_direct_s,
                                                               calculate_travel_times_all_pixels_single_el_head_wave_flat_surface_s,
                                                               calculate_travel_times_all_pixels_single_el_shear_hybrid_s)

# Define bulk longitudinal wave type, for use in both send and receive:
wave_type_l = WaveType('L',
                       calculate_travel_times_tfm_l_direct_s,
                       calculate_travel_times_single_pixel_all_el_l_s,
                       calculate_travel_times_all_pixels_single_el_l_direct_s)

# Define bulk transverse wave type, for use in both send and receive:
wave_type_t = WaveType('T',
                       calculate_travel_times_tfm_t_direct_s,
                       calculate_travel_times_single_pixel_all_el_t_s,
                       calculate_travel_times_all_pixels_single_el_t_direct_s)

# Define transverse head wave type, for use in send only:
wave_type_h = WaveType('H',
                       calculate_travel_times_tfm_head_wave_flat_surface_s,
                       calculate_travel_times_single_pixel_all_el_h_s,
                       calculate_travel_times_all_pixels_single_el_head_wave_flat_surface_s)

# Define shear hybrid wave type, for use in send only:
wave_type_shear_hybrid = WaveType('Shear hybrid',
                                  calculate_travel_times_tfm_shear_hybrid_s,
                                  calculate_travel_times_single_pixel_all_el_shear_hybrid_s,
                                  calculate_travel_times_all_pixels_single_el_shear_hybrid_s)

# Organise these WaveTypes into dictionaries, one for those that can be used on the 'send' leg, and one for those that
# can be used on the 'receive' leg:
dict_wave_types_send = {wave_type_l.string_name: wave_type_l,
                        wave_type_t.string_name: wave_type_t,
                        wave_type_h.string_name: wave_type_h,
                        wave_type_shear_hybrid.string_name: wave_type_shear_hybrid}
dict_wave_types_receive = {wave_type_l.string_name: wave_type_l,
                           wave_type_t.string_name: wave_type_t}
