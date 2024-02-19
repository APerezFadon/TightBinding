from TightBinding import TightBinding, Site, shapes
from shapes import shapes
import numpy as np

L = 4
dt = -0.5

sq = shapes["square"]
norbs = 4

tb = TightBinding(sq["shape"](L), sq["primitive"], sq["loc_sites"], norbs)
tb.initialise()
tb.plot_lattice(size = 0.2)

inner = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]])

outerx = np.array([[0, 1, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])

outery = np.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

tb.add_hopping_kind(np.array([0, 0]), (0, 0), (1 + dt) * inner)
tb.add_hopping_kind(np.array([1, 0]), (0, 0), (1 - dt) * outerx)
tb.add_hopping_kind(np.array([0, 1]), (0, 0), (1 - dt) * outery)

tb.make_hamiltonian()
tb.see_hamiltonian()

tb.eigh()
tb.plot_spectrum()

for i in range(tb.dim):
    tb.plot_eigenstate(i, size = 0.2)
