import os
import numpy as np

from openmdao.api import Problem, ExternalCodeComp, IndepVarComp, Group, ExplicitComponent
from openmdao.api import PETScVector

from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI

DIRECTORY = os.path.dirname((os.path.abspath(__file__)))

if not MPI:
    raise RuntimeError()

rank = MPI.COMM_WORLD.rank
size = 15

class SummerDistribGroup(Group):
    def initialize(self):
        self.options.declare('toplevel_run_directory', types=str)

    def setup(self):
        self.add_subsystem("indep", IndepVarComp('x', np.zeros(size)))
        self.add_subsystem("C2",
                           DistribExtCodeComp(size,
                                toplevel_run_directory=self.options['toplevel_run_directory']))
        self.add_subsystem("C3", Summer(size))

        self.connect('indep.x', 'C2.invec')
        self.connect('C2.outvec', 'C3.invec')

        # make the parent directory
        try:
            os.mkdir(self.options['toplevel_run_directory'])
        except:
            pass

class DistribExtCodeComp(ExternalCodeComp):
    def __init__(self, size, **kwargs):
        super(DistribExtCodeComp, self).__init__(**kwargs)
        self.size = size
        self.distributed = True

    def initialize(self):
        super(DistribExtCodeComp, self).initialize()

        self.options.declare('toplevel_run_directory', types=str)
        pass

    def setup(self):
        comm = self.comm
        rank = comm.rank

        # this results in 8 entries for proc 0 and 7 entries for proc 1 when using 2 processes.
        sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.add_input('invec', np.ones(sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('outvec', np.ones(sizes[rank], float))

        self.input_file = 'distrib_comp_input.dat'
        self.output_file = 'distrib_comp_output.dat'
        self.options['external_input_files'] = [self.input_file,]
        self.options['external_output_files'] = [self.output_file,]

        self.options['command'] = [
            'python',
            os.path.join(DIRECTORY, 'extcode_distrib_comp.py'),
            self.input_file, self.output_file
        ]

        # at setup time set unique folder to run in
        subdir_name = 'distrib_{}'.format(rank)
        self.run_directory = os.path.join(self.options['toplevel_run_directory'], subdir_name)
        try:
            os.mkdir(self.run_directory)
        except:
            pass

    def compute(self, inputs, outputs):
        invec = inputs['invec']

        startdir = os.getcwd()
        # # run in directory unique to this external code
        os.chdir(self.run_directory)

        # generate the input file for the paraboloid external code
        with open(self.input_file, 'wb') as input_file:
            np.savetxt(input_file, invec)

        # the parent compute function actually runs the external code
        super(DistribExtCodeComp, self).compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        with open(self.output_file, 'rb') as output_file:
            outvec = np.loadtxt(output_file)

        outputs['outvec'] = outvec
        os.chdir(startdir)

class Summer(ExplicitComponent):
    """Sums a distributed input."""

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size

    def setup(self):
        # this results in 8 entries for proc 0 and 7 entries for proc 1
        # when using 2 processes.
        sizes, offsets = evenly_distrib_idxs(self.comm.size, self.size)
        start = offsets[rank]
        end = start + sizes[rank]

        # NOTE: you must specify src_indices here for the input. Otherwise,
        #       you'll connect the input to [0:local_input_size] of the
        #       full distributed output!
        self.add_input('invec', np.ones(sizes[self.comm.rank], float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('out', 0.0)

    def compute(self, inputs, outputs):
        data = np.zeros(1)
        data[0] = np.sum(self._inputs['invec'])
        total = np.zeros(1)
        self.comm.Allreduce(data, total, op=MPI.SUM)
        self._outputs['out'] = total[0]

p = Problem(model=SummerDistribGroup(toplevel_run_directory='/tmp/summer_distrib'))
p.setup(vector_class=PETScVector)

p['indep.x'] = np.ones(size)

p.run_model()

np.testing.assert_almost_equal(p['C3.out'], 30.)
