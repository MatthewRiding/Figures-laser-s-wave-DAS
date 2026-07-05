import numpy as np
from scipy.signal import hilbert

from functions.modsumgather import sum_gather, sum_gather_pos_and_neg_angles


def find_envelope_max_in_central_percent(
    sum_waveform_nm,
    central_percent
):
    # Get envelope function from analytic signal:
    envelope_nm = np.abs(hilbert(sum_waveform_nm))

    # Trim to central %:
    n_samples = len(envelope_nm)
    proportion_centre = central_percent / 100
    proportion_centre_start = 0.5 - (proportion_centre / 2)
    proportion_centre_stop = 0.5 + (proportion_centre / 2)
    i_centre_start = int(proportion_centre_start * n_samples)
    i_centre_stop = int(proportion_centre_stop * n_samples)
    envelope_central_percent_nm = envelope_nm[i_centre_start:i_centre_stop]

    # Calculate max:
    env_max_central_nm = np.max(envelope_central_percent_nm)

    return env_max_central_nm


def get_table_row(
    corr_array,
    theta_vector_deg,
    theta_crit_deg,
    theta_sep_deg,
    theta_max_deg,
    central_percent,
    sum_function
):
    """
    Calculate sum waveforms over different angular ranges.

    For each sum waveform, find the peak value of the envelope function
    within central region.

    Make % loss calculations.
    """
    # A1: 0-theta*
    sum_waveform_1_nm = sum_function(
        corr_array,
        theta_vector_deg,
        0,
        theta_crit_deg
    )
    A_1 = find_envelope_max_in_central_percent(
        sum_waveform_1_nm,
        central_percent
    )

    # A2: theta*-theta_sep
    sum_waveform_2_nm = sum_function(
        corr_array,
        theta_vector_deg,
        theta_crit_deg,
        theta_sep_deg
    )
    A_2 = find_envelope_max_in_central_percent(
        sum_waveform_2_nm,
        central_percent
    )

    # A3: theta_sep-theta_max
    sum_waveform_3_nm = sum_function(
        corr_array,
        theta_vector_deg,
        theta_sep_deg,
        theta_max_deg
    )
    A_3 = find_envelope_max_in_central_percent(
        sum_waveform_3_nm,
        central_percent
    )

    # A4: 0-theta* & theta_sep-theta_max
    sum_waveform_4_nm = (sum_waveform_1_nm + sum_waveform_3_nm)
    A_4 = find_envelope_max_in_central_percent(
        sum_waveform_4_nm,
        central_percent
    )

    # A5: 0-theta_max
    sum_waveform_5_nm = sum_function(
        corr_array,
        theta_vector_deg,
        0,
        theta_max_deg
    )
    A_5 = find_envelope_max_in_central_percent(
        sum_waveform_5_nm,
        central_percent
    )

    # A_1 + A_3:
    sum_of_A_1_3 = A_1 + A_3

    # A_1 + A_2 + A_3:
    sum_of_A_1_2_3 = A_1 + A_2 + A_3

    # % loss sub + super:
    percentage_loss_sub_plus_super = (
        (sum_of_A_1_3 - A_4) / sum_of_A_1_3) * 100

    # % loss 0-theta_max:
    percentage_loss_0_to_max = ((sum_of_A_1_2_3 - A_5) / sum_of_A_1_2_3) * 100

    return np.array(
        [A_1, A_2, A_3, A_4, A_5,
         theta_max_deg,
         sum_of_A_1_3, sum_of_A_1_2_3,
         percentage_loss_sub_plus_super, percentage_loss_0_to_max]
    )


def get_table_row_sdh(
    corr_array,
    theta_vector_deg,
    theta_crit_deg,
    theta_sep_deg,
    theta_max_deg,
    central_percent
):
    sum_function = sum_gather_pos_and_neg_angles

    table_row = get_table_row(
        corr_array,
        theta_vector_deg,
        theta_crit_deg,
        theta_sep_deg,
        theta_max_deg,
        central_percent,
        sum_function
    )

    return table_row


def get_table_row_hfs(
    corr_array,
    theta_vector_deg,
    theta_crit_deg,
    theta_sep_deg,
    theta_max_deg,
    central_percent
):
    sum_function = sum_gather

    table_row = get_table_row(
        corr_array,
        theta_vector_deg,
        theta_crit_deg,
        theta_sep_deg,
        theta_max_deg,
        central_percent,
        sum_function
    )

    return table_row
