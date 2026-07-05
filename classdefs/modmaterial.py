from functions.modcalculatecriticalangle import (calculate_critical_angle_radians)


class Material:
    def __init__(self, c_l_mpers, c_t_mpers, c_lsaw_mpers=None):
        self.c_l_mpers = c_l_mpers
        self.c_t_mpers = c_t_mpers
        self.c_lsaw_mpers = c_lsaw_mpers

        # Calculate derived instance variables:
        self.critical_angle_radians = calculate_critical_angle_radians(self.c_l_mpers, self.c_t_mpers)
        self.kappa = c_l_mpers / c_t_mpers
