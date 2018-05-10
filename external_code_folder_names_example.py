'''
 This script shows how to have two external code components run in their own directories.
 Also shows how to pass the values for directories to subsystems.

 The external code components in this script wrap simple python scripts that are also included
 in this repo. Using simple python scripts was just an easy way to demonstrate this. Normally
 you wouldn't have to wrap Python with Python!

 In practice, you will like be running an executable that has been wrapped. Here are the two Python
 scripts that have been wrapped.

    extcode_paraboloid.py
    extcode_square.py

The script runs those scripts while inside specific directories for running the wrapped code.
'''

import os
import numpy as np
from openmdao.api import Problem, ExternalCodeComp, IndepVarComp, Group

# to get the directory of where this script exists
DIRECTORY = os.path.dirname((os.path.abspath(__file__)))

class ParaboloidSquaredGroup(Group):

    def initialize(self):
        # want to be able to pass in the top level directory into the constructor
        self.options.declare('toplevel_run_directory', types=str)

    def setup(self):

        # create and connect inputs
        self.add_subsystem('p1', IndepVarComp('x', 3.0))
        self.add_subsystem('p2', IndepVarComp('y', -4.0))
        # Notice how we pass in value for the top level directory to the subsystems
        self.add_subsystem('p',
                           ParaboloidExternalCodeComp(
                               toplevel_run_directory=self.options['toplevel_run_directory']))
        self.add_subsystem('s',
                           SquareExternalCodeComp(
                               toplevel_run_directory=self.options['toplevel_run_directory']))

        self.connect('p1.x', 'p.x')
        self.connect('p2.y', 'p.y')
        self.connect('p.f_xy', 's.x')

        # make the parent directory if needed
        try:
            os.mkdir(self.options['toplevel_run_directory'])
        except:
            pass

class ParaboloidExternalCodeComp(ExternalCodeComp):
    '''Wrap of code that computes the paraboloid function given x and y'''
    def initialize(self):
        super(ParaboloidExternalCodeComp, self).initialize()
        # want to be able to pass in the top level directory into the constructor
        self.options.declare('toplevel_run_directory', types=str)

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)

        self.input_file = 'paraboloid_input.dat'
        self.output_file = 'paraboloid_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['command'] = [
            'python',
            os.path.join(DIRECTORY, 'extcode_paraboloid.py'),
            self.input_file, self.output_file
        ]

        # At setup time set unique folder to run in and create it if need be
        self.run_directory = os.path.join(self.options['toplevel_run_directory'], 'paraboloid')
        try:
            os.mkdir(self.run_directory)
        except:
            pass

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        # save the directory so we can go back to it
        startdir = os.getcwd()

        # run in directory unique to this external code
        os.chdir(self.run_directory)

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n%.16f\n' % (x,y))

        # the parent compute function actually runs the external code
        super(ParaboloidExternalCodeComp, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'r') as output_file:
            f_xy = float(output_file.read())

        outputs['f_xy'] = f_xy

        # go back to the original directory
        os.chdir(startdir)

class SquareExternalCodeComp(ExternalCodeComp):
    '''Wrap of code that computes the square function given x'''
    def initialize(self):
        super(SquareExternalCodeComp, self).initialize()
        # want to be able to pass in the top level directory into the constructor
        self.options.declare('toplevel_run_directory', types=str)

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('f_x', val=0.0)

        self.input_file = 'square_input.dat'
        self.output_file = 'square_output.dat'

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['command'] = [
            'python',
            os.path.join(DIRECTORY, 'extcode_square.py'),
            self.input_file, self.output_file
        ]

        # At setup time set unique folder to run in and create it if need be
        self.run_directory = os.path.join(self.options['toplevel_run_directory'], 'squared')
        try:
            os.mkdir(self.run_directory)
        except:
            pass


    def compute(self, inputs, outputs):
        x = inputs['x']

        # save the directory so we can go back to it
        startdir = os.getcwd()

        # run in directory unique to this external code
        os.chdir(self.run_directory)

        # generate the input file for the square external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('%.16f\n' % (x))

        # the parent compute function actually runs the external code
        super(SquareExternalCodeComp, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_x
        with open(self.output_file, 'r') as output_file:
            f_x = float(output_file.read())

        outputs['f_x'] = f_x

        # go back to the original directory
        os.chdir(startdir)


prob = Problem(ParaboloidSquaredGroup(toplevel_run_directory='/tmp/paraboloid_squared'))

prob.setup()
prob.run_model()

np.testing.assert_almost_equal(prob['s.f_x'], 225.0)
