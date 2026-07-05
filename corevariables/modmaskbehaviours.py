import numpy as np

from classdefs.modmaskbehaviour import MaskBehaviour

# High or low-pass, governed by a single angle:
mask_behaviour_ignore_above = MaskBehaviour('Ignore above', np.ma.masked_greater)
mask_behaviour_ignore_below = MaskBehaviour('Ignore below', np.ma.masked_less)

# Band-based, governed by a 'start' and a 'stop' angle:
mask_behaviour_ignore_inside = MaskBehaviour('Ignore inside', np.ma.masked_inside)
mask_behaviour_ignore_outside = MaskBehaviour('Ignore outside', np.ma.masked_outside)

dict_mask_behaviours = {mask_behaviour_ignore_above.string_name: mask_behaviour_ignore_above,
                        mask_behaviour_ignore_below.string_name: mask_behaviour_ignore_below,
                        mask_behaviour_ignore_inside.string_name: mask_behaviour_ignore_inside,
                        mask_behaviour_ignore_outside.string_name: mask_behaviour_ignore_outside}
