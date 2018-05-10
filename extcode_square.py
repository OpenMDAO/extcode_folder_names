#!/usr/bin/env python
#
# usage: square_paraboloid.py input_filename output_filename
#
# Evaluates the equation f(x) = (x^2).
#
# Read the value of `x` from input file
# and write the value of `f_x` to output file.

if __name__ == '__main__':
    import sys

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(input_filename, 'r') as input_file:
        file_contents = input_file.readlines()

    x = [float(f) for f in file_contents]

    f_x = x[0]**2

    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % f_x)
