import numpy as np

# Numbers related to π

_π = 3.141592653589793238462643
_2π = 6.283185307179586476925284
_2πj = 6.283185307179586476925284j
_logπ = 1.144729885849400174143426
_log2π_2 = 0.91893853320467274178  # log(2π)/2
sqrt_2_π = 0.79788456080286535588  # √(2/π)
_2πe = 17.07946844534713413093

# Numbers related to the base of the natural logarithm (e)

_e = 2.718281828459045235360288

# Numbers related to the Euler-Mascheroni constant

_γ = 0.5772156649015328655494272

# Limits

_ε = 2.220446049250313e-16  # Machine epsilon
_root_ε = 1.490116119384765625e-8  # Square root of machine epsilon

# Largest float64 input to exp that doesn't overflow
_MAXEXP = 709.7827128933839731

# Largest float64 input to cosh that doesn't overflow. Note that sinh
# overflows when cosh does.
_MAXCOSH = 710.47586007394329499

# Smallest float64 before exp underflows
MINEXP = -745.13321910194110842
