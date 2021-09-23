import numpy as np


def inversek(coordinates, LinkLength):

    if coordinates[0] == 0 and coordinates[1] == 0:
        print("Singularity")
        theta = [0, np.pi/2, 0]
    else:
        theta1 = np.arctan(coordinates[1]/coordinates[0])

        s = coordinates[2] - LinkLength[1]
        r = (coordinates[0]**2 + coordinates[1]**2)**0.5

        if r == 0:
            theta2 = 0
        else:
            theta2 = np.arctan(s/r)

        d3 = ((r**2 + s**2)**0.5) - LinkLength[1]

        theta = [theta1, theta2, d3]

    return theta


Coordinates = [1, 1, 1]

LinkLength = [1, 1]


a = inversek(Coordinates, LinkLength)

print(a)