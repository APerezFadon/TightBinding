from TightBinding import TightBinding
from shapes import shapes
import numpy as np

PauliI = np.array([[1, 0], [0, 1]], dtype = np.complex128)
PauliX = np.array([[0, 1], [1, 0]], dtype = np.complex128)
PauliY = np.array([[0, -1j], [1j, 0]], dtype = np.complex128)
PauliZ = np.array([[1, 0], [0, -1]], dtype = np.complex128)

L = 10
dt = -0.5

sq = shapes["square"]
norbs = 4

tb = TightBinding(sq["shape"](L), sq["primitive"], sq["loc_sites"], norbs)
tb.initialise()
tb.plot_lattice(size = 0.2)

inner = np.kron(PauliZ, PauliY) + np.kron(PauliY, PauliI)

outerx = 0.5 * (np.kron(PauliZ, PauliY) - 1j * np.kron(PauliZ, PauliX))

outery = 0.5 * (np.kron(PauliY, PauliI) - 1j * np.kron(PauliX, PauliI))

tb.add_hopping_kind(np.array([0, 0]), (0, 0), (1 + dt) * inner)
tb.add_hopping_kind(np.array([1, 0]), (0, 0), (1 - dt) * outerx)
tb.add_hopping_kind(np.array([0, 1]), (0, 0), (1 - dt) * outery)

tb.make_hamiltonian()
tb.see_hamiltonian()

tb.eigh()
tb.plot_spectrum()

for i in range(tb.dim // 2 - 2, tb.dim):
    tb.plot_eigenstate(i, size = 0.2)
