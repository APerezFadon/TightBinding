from TightBinding import TightBinding, Site, shapes
import numpy as np

def shape(pos):
    return (pos[0] >= L[0] and pos[0] < L[1]) and pos[1] == 0

L = [-0.5, 29.5]
lat_vecs = np.array([[1, 0], [0, 1]])
dt = 0.5

tb = TightBinding(shape, lat_vecs, [np.array([0, 0]), np.array([0.5, 0])])
tb.initialise()
tb.plot_lattice(size = 0.2)

tb.add_hopping_kind(np.array([0, 0]), (0, 1), 1 + dt)
tb.add_hopping_kind(np.array([1, 0]), (1, 0), 1 - dt)

tb.make_hamiltonian()
tb.see_hamiltonian()

tb.eigh()
tb.plot_spectrum()

tb.plot_eigenstate(tb.dim // 2 - 1, size = 0.2)
tb.plot_eigenstate(tb.dim // 2, size = 0.2)
