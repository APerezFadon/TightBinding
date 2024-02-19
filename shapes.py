import numpy as np

def get_square_function(L):
    def square(pos):
        return pos[0] >= 0 and pos[0] < L and pos[1] >= 0 and pos[1] < L
    return square

def get_hexagon_function(L):
    def hexagon(pos):
        pos[1] += np.cos(np.pi / 6) + 1 / (2 * np.sqrt(3))
        sixty_d = np.pi / 3
        theta = np.arctan2(pos[1], pos[0])
        r2 = pos[0] * pos[0] + pos[1] * pos[1]
        return np.sqrt(r2) <= L * np.sqrt(3) / (2 * np.sin(theta + sixty_d - sixty_d * np.floor(theta / sixty_d)))
    return hexagon

def get_circle_function(L):
    def circle(pos):
        return pos[0] * pos[0] + pos[1] * pos[1] <= L * L
    return circle

shapes = {"square": {"shape": get_square_function,
                     "primitive": np.array([[1, 0], [0, 1]]),
                     "loc_sites": [np.array([0, 0])]},
          "triangle": {"shape": "TODO",
                       "primitive": "TODO",
                       "loc_sties": "TODO"},
          "hexagon": {"shape": get_hexagon_function,
                      "primitive": np.array([[1, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)]]),
                      "loc_sites": [np.array([0, 0]), np.array([np.cos(np.pi / 3), 1 / (2 * np.sqrt(3))])]},
          "circle": {"shape": get_circle_function,
                     "primitive": np.array([[1, 0], [0, 1]]),
                     "loc_sites": [np.array([0, 0])]}}


