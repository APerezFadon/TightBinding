import numpy as np

class Site:
    def __init__(self, 
                 unit, # numpy array with index of unit cell in primitive basis
                 sublattice = 0): # Index of the sublattice within the unit cell
        
        self.unit = unit
        self.sublattice = sublattice

    def __eq__(self, other):
        return (np.sum(np.abs(self.unit - other.unit)) < 1e-6) and (self.sublattice == other.sublattice)
    
    def __str__(self):
        return f"unit: {self.unit}, sublattice: {self.sublattice}"
    
    def __lt__(self, other):
        if self.unit[1] != other.unit[1]:
            return self.unit[1] < other.unit[1]
        elif self.unit[0] != other.unit[0]:
            return self.unit[0] < other.unit[0]
        else:
            return self.sublattice < other.sublattice

if __name__ == "__main__":
    s1 = Site(np.array([0, 0]), 0)
    s2 = Site(np.array([-1, 0]), 1)
    s3 = Site(np.array([3, 0]), 1)
    print(s1 < s2)

    arr = [s1, s2]
    sorted_arr = np.sort(arr)
    print(sorted_arr[0])
    print(sorted_arr[1])
    print(type(sorted_arr))
    print(np.where(sorted_arr == s3)[0][0])
