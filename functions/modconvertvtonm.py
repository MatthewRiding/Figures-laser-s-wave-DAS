def convert_quartet_v_to_nm(measurement_v):
    # The quartet calibration in 'auto' mode is
    # 100mV per nanometre.
    # i.e. 10nm per volt.
    # To convert a measurement in volts into nm, we simply
    # multiply the volt values by 10:
    measurement_nm = measurement_v * 10
    return measurement_nm
