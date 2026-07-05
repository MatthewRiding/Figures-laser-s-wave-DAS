def get_variable_keys_from_mat_contents_dict(mat_contents_dict=dict):
    # The contents dict will always have certain standard keys, created by the scipy.io.loadmat() function.
    keys = mat_contents_dict.keys()

    # Exclude the standard keys:
    standard_keys_list = ['__globals__', '__header__', '__version__']
    keys_variables = set(keys).difference(standard_keys_list)

    return list(keys_variables)
