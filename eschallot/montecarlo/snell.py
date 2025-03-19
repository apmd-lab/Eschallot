import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def snell(n1, n2, angle_inc):
    """ For absorbing materials, n1 & n2 should still be complex valued """
    temp_result = (n1*np.sin(angle_inc))/n2
    angle = np.arcsin(temp_result)

    return angle
