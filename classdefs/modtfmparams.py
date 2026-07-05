class TFMParams:
    def __init__(self,
                 image_name_string,
                 pitch_mm,
                 v_l_mpers,
                 v_t_mpers,
                 wave_set_string,
                 grid_size_x_mm,
                 grid_size_z_mm,
                 n_pixels_z,
                 filter_tf,
                 butter_order,
                 band_min_MHz,
                 band_max_MHz,
                 mask_tf,
                 mask_angle_deg,
                 mask_behaviour_string
                 ):

        # Pin given values for variables to class instance as instance variables:
        filter_string = f' {band_min_MHz:g}-{band_max_MHz:g}MHz'
        self.image_name_string = image_name_string + ' ' + wave_set_string + filter_string
        self.pitch_mm = pitch_mm
        self.v_l_mpers = v_l_mpers
        self.v_t_mpers = v_t_mpers
        self.wave_set_string = wave_set_string
        self.grid_size_x_mm = grid_size_x_mm
        self.grid_size_z_mm = grid_size_z_mm
        self.n_pixels_z = n_pixels_z
        self.filter_tf = filter_tf
        self.butter_order = butter_order
        self.band_min_MHz = band_min_MHz
        self.band_max_MHz = band_max_MHz
        self.mask_tf = mask_tf
        self.mask_angle_deg = mask_angle_deg
        self.mask_behaviour_string = mask_behaviour_string
