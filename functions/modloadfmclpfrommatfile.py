from scipy.io import loadmat

from functions.modgetvariablekeysfrommatcontentsdict import get_variable_keys_from_mat_contents_dict
from functions.modconvert2dto3dfmcformat import convert_2d_to_3d_fmclp_format


def load_fmclp_from_mat_file(file_path):
    # Load the contents of the .mat file as a dictionary using scipy:
    mat_contents_dict = loadmat(file_path)

    # Find the variables from the .mat file by examining the keys of mat_contents_dict:
    mat_variables_keys = get_variable_keys_from_mat_contents_dict(mat_contents_dict)

    # Allocate the variables found in the .mat file to their correct workspace objects:
    # Convert 2D array FMC format into 3D array format:
    # BAD ASSUMPTION: A-scan matrix is the only variable present in the .mat file.
    fmc_3d = convert_2d_to_3d_fmclp_format(mat_contents_dict[mat_variables_keys[0]])
    return fmc_3d
