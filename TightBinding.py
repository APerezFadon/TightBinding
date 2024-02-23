import numpy as np
from .Site import Site
from .shapes import shapes
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from .diagonaliser import diagonalise

class TightBinding:
    def __init__(self, 
                 shape: function, # returns True if a point (x, y) is within bounds
                 primitive: np.ndarray, # primitive unit vectors (vec1, vec2)
                 loc_sites: list[np.ndarray] = [np.array([0, 0])], # location of sites within unit cell
                 norbs: int = 1, # number of orbitals per site
                 start: np.ndarray = np.array([0, 0])): # position of unit cell that is inside shape
        
        self.shape = shape
        self.primitive = primitive
        self.loc_sites = loc_sites
        self.norbs = norbs
        self.start = start
    
    def get_pos(self, site: Site) -> np.ndarray:
        return site.unit @ self.primitive + self.loc_sites[site.sublattice]

    def initialise(self):
        self.dim = 0
        self.sites = []

        self.add_sites(self.start)
        self.dim *= self.norbs

        self.sites = np.sort(self.sites)

        self.hopping_kind = []

    def add_sites(self, current: np.ndarray):
        fully_out = True
        for i in range(len(self.loc_sites)):
            s = Site(current, i)
            if self.shape(self.get_pos(s)) and s not in self.sites:
                self.dim += 1
                self.sites.append(s)
                fully_out = False
        
        if not fully_out:
            self.add_sites(current + np.array([1, 0]))
            self.add_sites(current + np.array([0, 1]))
            self.add_sites(current + np.array([-1, 0]))
            self.add_sites(current + np.array([0, -1]))
    
    def plot_lattice(self, cols: list = None, title: str = None, size: list = None, show: bool = True):
        fig, ax = plt.subplots()
        patches = []

        if cols is None:
            cols = [0 for i in range(len(self.sites))]
        
        if size == None:
            size = 0.25

        for site in self.sites:
            patches.append(Circle(self.abs_pos(site), size))

        p = PatchCollection(patches, cmap = "hot")
        p.set_array(cols)
        ax.add_collection(p)
        ax.axis('scaled')

        if title == None:
            ax.set_title(f"Lattice")
        else:
            ax.set_title(title)

        ax.axis('equal')
        ax.set_facecolor((0.85, 0.85, 0.85))
        fig.colorbar(p, ax=ax)

        if show:
            plt.show()
        
    def site_to_vec(self, site: Site) -> np.ndarray:
        indx = np.where(self.sites == site)[0][0]
        vec = np.zeros((self.sites.shape[0], 1), dtype = np.complex128)
        vec[indx] = 1
        return vec
    
    def add_hopping_kind(self, rel_unit: np.ndarray, # relative vector between unit cells
                               sublattices: tuple, # (source sublattice, target sublattice)
                               value: np.ndarray): # matrix element
        if self.norbs != 1:
            assert(value.shape[0] == self.norbs)
            assert(value.shape[1] == self.norbs)
        self.hopping_kind.append([rel_unit, sublattices, value])

    def make_hamiltonian(self) -> np.ndarray:
        self.H = np.zeros((self.dim, self.dim), dtype = np.complex128)
        for i in range(self.sites.shape[0]):
            for j in range(len(self.hopping_kind)):
                if self.sites[i].sublattice == self.hopping_kind[j][1][0]:
                    s = Site(self.sites[i].unit + self.hopping_kind[j][0], self.hopping_kind[j][1][1])
                    new_indx = np.where(self.sites == s)[0]

                    if len(new_indx) > 0:
                        Imin = i * self.norbs
                        Imax = Imin + self.norbs
                        Jmin = new_indx[0] * self.norbs
                        Jmax = Jmin + self.norbs
                        self.H[Jmin: Jmax, Imin: Imax] = self.hopping_kind[j][2]
                        self.H[Imin: Imax, Jmin: Jmax] = np.conj(self.hopping_kind[j][2]).T
        return self.H
    
    def see_hamiltonian(self, show: float = True):
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("Real part")
        axs[0].imshow(np.real(self.H))
        axs[1].set_title("Imaginary part")
        axs[1].imshow(np.imag(self.H))
        plt.tight_layout()

        if show:
            plt.show()
    
    def eigh(self, symmetry: np.ndarray = None) -> tuple[np.ndarray]:
        if symmetry is None:
            eighsystem = np.linalg.eigh(self.H)
            eighvals = np.real(eighsystem[0])
            eighvecs = eighsystem[1].astype(np.complex128)
            arr1inds = eighvals.argsort()
            self.eighvals = eighvals[arr1inds]
            self.eighvecs = eighvecs[:, arr1inds]
            return self.eighvals, self.eighvecs
        else:
            eighsystem = diagonalise(self.H, symmetry)
            self.eighvals = np.real(eighsystem[0])
            self.eighvecs = eighsystem[1]
            return self.eighvals, self.eighvecs
    
    def plot_spectrum(self, show: bool = True):
        plt.figure()
        plt.title("Spectrum")
        plt.scatter([i for i in range(self.eighvals.shape[0])], self.eighvals, s = 5)
        plt.grid()
        plt.xlabel("Energy level")
        plt.ylabel("Energy")

        if show:
            plt.show()
    
    def plot_eigenstate(self, i: int, size: int = None, show = True):
        cols = np.zeros(self.sites.shape[0])
        for k in range(0, self.dim, self.norbs):
            for orb in range(self.norbs):
                cols[k // self.norbs] += np.abs(self.eighvecs[k + orb, i])**2

        self.plot_lattice(cols, f"Eigenstate {i} with Energy {self.eighvals[i]}", size, show)


if __name__ == "__main__":
    L = 4
    
    norbs = 4

    shape = shapes["square"]
    tb = TightBinding(shape["shape"](L), shape["primitive"], shape["loc_sites"], norbs)
    tb.initialise()
    #tb.plot_lattice()

    mat = np.array([[0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])
    
    tb.add_hopping_kind(np.array([0, 0]), (0, 0), mat)

    mat2 = 1j * np.array([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])
    
    tb.add_hopping_kind(np.array([1, 0]), (0, 0), mat2)

    mat3 = 1j * np.array([[0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 1, 0, 0]])
    
    tb.add_hopping_kind(np.array([0, 1]), (0, 0), mat3)

    tb.make_hamiltonian()
    tb.see_hamiltonian()
    tb.eigh()
    tb.plot_spectrum()
    tb.plot_eigenstate(0)
