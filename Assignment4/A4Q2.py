import numpy as np


def inversek(coordinates, LinkLength):

    theta2 = np.arccos((coordinates[0]**2 + coordinates[1]**2 - LinkLength[0]**2 - LinkLength[1]**2)/(2*LinkLength[0]*LinkLength[1]))

    theta1 = np.arctan(coordinates[1]/coordinates[0]) - np.arctan((LinkLength[1]*np.sin(theta2))/(LinkLength[0] + LinkLength[1]*np.cos(theta2)))
    d3 = coordinates[2]

    theta = [theta1, theta2, d3]

    return theta


Coordinates = [1, 1, 1]

LinkLength = [1, 1]


a = inversek(Coordinates, LinkLength)

print(a)