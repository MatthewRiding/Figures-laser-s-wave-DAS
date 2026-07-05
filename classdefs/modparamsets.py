import numpy as np
from scipy.signal import detrend

from functions.modthetacrittrig import tan_theta_crit, calculate_theta_crit_deg
from functions.modgeneratetimevectorus import generate_time_vector_us
from functions.modgeneratexvectormm import generate_x_vector_mm
from functions.modextractmonostaticsaft import extract_mss_from_fmc2d
from functions.modfilterfmc3dbutter import filter_fmc3d_butter


class material_param_set():
    def __init__(self, c_L_mpers, c_T_mpers):
        self.c_L_mpers = c_L_mpers
        self.c_T_mpers = c_T_mpers
        self.theta_crit_deg = calculate_theta_crit_deg(c_L_mpers, c_T_mpers)
        self.tan_theta_crit = tan_theta_crit(c_L_mpers,
                                             c_T_mpers)
        self.kappa = c_L_mpers / c_T_mpers


class scan_param_set():
    def __init__(self, pitch_mm, t_min_us, t_max_us, n_samples, rise_time_us):
        self.pitch_mm = pitch_mm
        self.t_min_us = t_min_us
        self.t_max_us = t_max_us
        self.n_samples = n_samples
        self.rise_time_us = rise_time_us

        self.time_vector_us = generate_time_vector_us(t_min_us-rise_time_us,
                                                      t_max_us-rise_time_us,
                                                      n_samples)
        self.period_s = ((self.t_max_us - self.t_min_us) * 10**-6) / (
            self.n_samples-1)
        self.frequency_sampling_Hz = 1 / self.period_s


class fmc_param_set(scan_param_set):
    def __init__(self, fmc_2d, pitch_mm, t_min_us, t_max_us, rise_time_us):
        self.fmc_2d = fmc_2d
        n_samples = np.shape(self.fmc_2d)[0]

        super().__init__(pitch_mm, t_min_us, t_max_us, n_samples, rise_time_us)

        # Detrend each A-scan:
        self.fmc_2d_detrend = detrend(self.fmc_2d, axis=0, type='constant')

        # Create element x vector:
        self.n_tx = int(np.sqrt(np.shape(self.fmc_2d)[1]))
        self.aperture_mm = self.pitch_mm * (self.n_tx - 1)
        self.x_elements_mm = np.linspace(0, self.aperture_mm,
                                         self.n_tx)


class sdh_fmc_param_set(fmc_param_set):
    def __init__(self, fmc_2d, pitch_mm, t_min_us, t_max_us, rise_time_us,
                 x_sdh_mm, z_sdh_mm, radius_sdh_mm):
        super().__init__(fmc_2d, pitch_mm, t_min_us, t_max_us, rise_time_us)

        self.fmc_2d_detrend = detrend(fmc_2d, 0, 'constant')
        self.x_sdh_mm = x_sdh_mm
        self.z_sdh_mm = z_sdh_mm
        self.radius_sdh_mm = radius_sdh_mm

        # Get lateral offsets of gen points from SDH centre:
        self.x_offsets_from_sdh_mm = self.x_elements_mm - self.x_sdh_mm

        # Calculate vector of ray emission angles
        # (relative to top surface normal):
        (self.angles_gen_sdh_deg
         ) = np.rad2deg(np.arctan2(self.x_offsets_from_sdh_mm,
                                   self.z_sdh_mm)
                        )

        # Calculate vector of direct ray path lengths
        # from each element, to SDH surface, and back:
        (self.direct_reflected_path_lengths_mm
         ) = 2 * (np.sqrt(self.x_offsets_from_sdh_mm**2 +
                          self.z_sdh_mm**2
                          ) -
                  self.radius_sdh_mm)

        # Extract mono-static SAFT B-scan:
        self.mss_b_scan = extract_mss_from_fmc2d(self.fmc_2d_detrend)

    def get_mss_filtered(self, Wn_Hz=2*10**6):
        # Subtract mean waveform:
        mss_mws = (self.mss_b_scan -
                   np.atleast_2d(np.mean(self.mss_b_scan, axis=1)).T)

        # Filter:
        mss_filtered_mws = filter_fmc3d_butter(mss_mws,
                                               self.frequency_sampling_Hz,
                                               10, 'highpass', Wn_Hz)

        return mss_filtered_mws


class nmo_scan_param_set(scan_param_set):
    def __init__(self, b_scan_array_2d, pitch_mm, t_min_us, t_max_us,
                 rise_time_us, thickness_b_mm):
        self.thickness_b_mm = thickness_b_mm
        self.b_scan_array_2d = b_scan_array_2d
        self.b_scan_array_2d_detrend = detrend(b_scan_array_2d, 0, 'constant')
        (self.n_samples, self.n_elements) = np.shape(self.b_scan_array_2d)

        super().__init__(pitch_mm, t_min_us, t_max_us, self.n_samples,
                         rise_time_us)

        self.x_max_mm = (self.n_elements - 1) * self.pitch_mm
        self.x_vector_mm = generate_x_vector_mm(self.n_elements, self.pitch_mm)
        self.path_lengths_vector_mm = np.sqrt((2 * thickness_b_mm)**2 +
                                              self.x_vector_mm**2)
        self.theta_vector_deg = np.rad2deg(np.atan2(self.x_vector_mm,
                                                    2 * self.thickness_b_mm))
