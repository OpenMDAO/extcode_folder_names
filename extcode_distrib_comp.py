#!/usr/bin/env python
#
# usage: extcode_paraboloid.py input_filename output_filename
#
# Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
#
# Read the values of `x` and `y` from input file
# and write the value of `f_xy` to output file.

import numpy as np

if __name__ == '__main__':
    import sys

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(input_filename, 'rb') as input_file:
        invec = np.loadtxt(input_file)

    outvec = 2.0 * invec

    with open(output_filename, 'wb') as output_file:
        np.savetxt(output_file, outvec)
