# Construction of Real Space Tight-Binding Hamiltonians
This module allows an easy construction of real space tight-binding models. The information needed to create the model is a boolean function `shape` which takes an (x, y) position as an argument and returns `True` if (x, y) is within the space in which the model is to be created, and `False` otherwise. In addition, one must provide the primitive lattice vectors as a 2D array of shape `(2, 2)`, such that `primitive[i]` is the ith lattice vector. Further, `loc_sites` is a python list with the location of the sites on the unit cell, each expressed as a numpy array of shape `(2,)` (defaults to `[np.array([0, 0])]`). `norbs` is the number of orbitals per site (defaults to `1`), and `start` is a numpy array of shape `(2,)` representing a unit cell enclosed by `shape` in the basis of `primitive` (defauls to `np.array([0, 0])`). As an example, a simple square lattice would be defined as 
```
L = 10
square = lambda pos: (0 <= pos[0] and pos[0] < L) and (0 <= pos[1] and pos[1] < L)
prim = np.array([[1, 0], [0, 1]])
tb = TightBinding(square, prim)
```
Some common geometries have already been implemented, including `triangle`, `square`, `hexagon` and `circle`. These can be accessed through `shapes.py`. The next step is to initialise the model, done by `tb.initialise()`. This calculates which sites are enclosed by `shape`, and calculates the dimension of the Hilbert space. Once this is done, the lattice can be displayed by `tb.plot_lattice()`. Adding hoppings can be done in the following manner:
```
tb.add_hopping_kind(rel_unit, sublattices, value)
```
where `rel_unit` is a numpy array of shape `(2,)` with the relative position between the source and target unit cells in the basis of `primitive`, `sublattices` is a tuple of length 2 where the first element is the index of the site within the source unit cell and the second is the site index for the target. Lastly, `value` is the matrix element associated with that hopping (it must be a numpy array of shape `(tb.norbs, tb.norbs)`). Once all hoopings have been defined, the Hamiltonian is constructed with 
```
tb.make_hamiltonian()
```
The Hamiltonian can be diagonalised with
```
tb.eigh(symmetry)
```
where `symmetry` is an optional argument which is an expected symmetry of the Hamiltonian (this is NOT checked). Then, in the case of degeneracies, the returned eigenstates will be chosen to also be eigensates of `symmetry`. After the Hamiltonian has been diagonalised, the spectrum can be displayed with
```
tb.plot_spectrum()
```
Lastly, a particular eigenstate can be displayed with 
```
tb.plot_eigenstate(i)
```
where `i` is the index of the eigenstate (ordered by increasing energy). For further guidance, please check the `examples` folder.

## Further Comments
The only dependency is `numpy`, and is compatible with most versions (you would have to go back to a really old version for this code to be incompatibe). While I am not making any claims about performance or good coding practise such as checking datatypes or using sparse matrices (yet), this module does offer an easy way to quickly define models and obtain a Hamiltonian with a consistent ordering. The chosen ordering is as follows: a site is defined to be "lower" than another site if the `y` coordinate of the unit cell (in the basis of `primitive`) is lower. If they are the same, then the `x` component is compated. If these are also the same, then the index of the site within the unit cell is compared. The sites are then in "increasing" order. If `norbs` > 1, then all orbitals for each site are grouped together. I am happy to accept any feedback/pull requests. I hope this helps!

