import numpy as np

def list_repeated_energies(energies, threshold=1e-8, shift=0):
	energies_last = -np.inf
	degen_list = []
	temp_degen_list = [0]
	degen_last = False

	for i, energy_current in enumerate(energies):

		if abs(energy_current - energies_last) <= threshold:
			degen_current = True
		else:
			degen_current = False

		if degen_current is True and i != energies.__len__() - 1:
			temp_degen_list.append(i + shift)
		elif degen_current is True and i == energies.__len__() - 1:
			temp_degen_list.append(i + shift)
			degen_list.append(tuple(temp_degen_list))
		elif degen_current is False and degen_last is True:
			degen_list.append(tuple(temp_degen_list))
			temp_degen_list = [i + shift]
		else:
			temp_degen_list = [i + shift]

		degen_last = degen_current
		energies_last = energy_current

	return degen_list


def diagonalise(H, P):
	eighsystem = np.linalg.eigh(H)
	eighvals = np.real(eighsystem[0])
	eighvecs = eighsystem[1].astype(np.complex128)
	arr1inds = eighvals.argsort()
	eighvals = eighvals[arr1inds]
	eighvecs = eighvecs[:, arr1inds]

	degen_list = list_repeated_energies(eighvals)
	for i, deg_set in enumerate(degen_list):
		degen = eighvecs[:, deg_set]
		Pmat = np.conj(degen).T @ P @ degen

		_, Peighvecs = np.linalg.eig(Pmat)
		eighvecs[:, deg_set] = degen @ Peighvecs
	return eighvals, eighvecs

if __name__ == "__main__":
	H = np.array([[2, 0, 0],
				  [0, 1, 0],
				  [0, 0, 1]], dtype=np.float32)

	P = np.array([[0, 0, 0],
				  [0, 0, -1],
				  [0, 1, 0]], dtype=np.float32)


	print(np.round(diagonalise(H, P)[1], 3))
