import numpy as np


from functions.modbuildxgenandxdetmatrices import build_x_gen_and_x_det_matrices_m


class ListedTFMImage:
    def __init__(self, worker_id, tfm_constructor, n_tx):

        # Define instance variables:
        self.worker_id = worker_id
        self.tfm_constructor = tfm_constructor

        # Make space for instance variables not assigned in constructor:
        self.image_complex_nm = None
        self.max_abs_nm = None
        self.fmc_3d_filtered = None
        self.progress_string = None
        self.complete = False
        self.x_gen_matrix_m = None
        self.x_det_matrix_m = None
        self.x_grid_m, self.z_grid_m = tfm_constructor.get_pixel_meshgrid_m()

        # Build instance variables derived from tfm_constructor:
        self.x_gen_matrix_m, self.x_det_matrix_m = build_x_gen_and_x_det_matrices_m(n_tx, self.tfm_constructor.pitch_mm)

    def completed(self):
        self.progress_string = ''
        self.complete = True

    def get_display_string(self):
        return self.tfm_constructor.image_name_string + self.progress_string

    def new_image_complex(self, image_complex_nm):
        self.image_complex_nm = image_complex_nm
        self.max_abs_nm = np.max(np.abs(self.image_complex_nm))
