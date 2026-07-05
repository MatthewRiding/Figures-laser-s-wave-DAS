from functions.modfilterfmc3dbutter import bandpass_fmc3d_butter


class FilterSpec:
    """A class to store the specifications for a SciPy Butter bandpass filter, to be used on the full matrix."""
    def __init__(self, butter_order, band_min_mhz, band_max_mhz):
        self.butter_order = butter_order
        self.band_min_mhz = band_min_mhz
        self.band_max_mhz = band_max_mhz

    def apply_to_fmc(self, full_matrix):
        fmc_3d_filtered = bandpass_fmc3d_butter(full_matrix.displacements_3d_nm, full_matrix.frequency_sampling_hz, self.butter_order,
                                                band_min_MHz=self.band_min_mhz, band_max_MHz=self.band_max_mhz)
        return fmc_3d_filtered
