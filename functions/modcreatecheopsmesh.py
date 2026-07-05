import numpy as np


def create_cheops_meshes(n_tx):
    indices = np.arange(n_tx)
    gen_indices_mesh, det_indices_mesh = np.meshgrid(indices, indices)
    return gen_indices_mesh, det_indices_mesh
