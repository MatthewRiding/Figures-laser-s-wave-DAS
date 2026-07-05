class WaveType:
    """
    A valid wave type must have three functions:
    One computes travel times for this wave type to all pixels in an imaging grid, and for all elements in a linear
    periodic array, and is used in TFM imaging.
    The second function must compute travel times to a single pixel from all array elements for that wave type, and is
    used in computing Cheops surfaces.
    The third function must compute the travel time to all pixels in an imaging grid from/to a single element, and is
    used in the plotting of isocrons.
    Finally, the wave type must also have a characteristic name string, which will be used as a dictionary key for access
    and in comboboxes.
    """
    def __init__(self,
                 string_name,
                 func_calculate_travel_times_all_pixels_all_el_s,
                 func_calculate_delay_times_single_pixel_all_el_s,
                 func_calculate_travel_times_all_pixels_single_el_s):
        # Name:
        self.string_name = string_name
        # TFM function:
        self.func_calculate_travel_times_all_pixels_all_el_s = func_calculate_travel_times_all_pixels_all_el_s
        # Cheops function:
        self.func_calculate_delay_times_single_pixel_all_el_s = func_calculate_delay_times_single_pixel_all_el_s
        # Isochron function:
        self.func_calculate_travel_times_all_pixels_single_el_s = func_calculate_travel_times_all_pixels_single_el_s
