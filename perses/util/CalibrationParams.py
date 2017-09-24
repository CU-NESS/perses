def _get_gain_params(gain_terms):
    return [('receiver_gain_a{}'.format(i)) for i in range(gain_terms)]
def _get_offset_params(offset_terms):
    return [('receiver_offset_a{}'.format(i)) for i in range(offset_terms)]
def _get_calibration_params(gain_terms, offset_terms):
    return _get_gain_params(gain_terms) + _get_offset_params(offset_terms)
