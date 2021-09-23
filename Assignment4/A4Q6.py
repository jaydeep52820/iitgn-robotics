import numpy as np

U = [[0.3535534, -0.3535534,  0.8660254],
     [0.9185587,  0.3061862, -0.2500000],
     [-0.1767767,  0.8838835,  0.4330127]]


def inversek(U):
    if (U[0][2] != 0 or U[1][2] != 0) and (U[2][2] == 1 or U[2][2] == -1):
        print("Invalid Matrix")
        return
    if U[0][2] == 0 or U[1][2] == 0:
        print("theta = 0 and Many solution for phi and psi")
        return

    theta = np.arctan(((1 - U[2][2]**2)**0.5)/U[2][2])
    phi = np.arctan(U[1][2]/U[0][2])
    psi = np.arctan(-U[2][1]/U[2][0])


    eangles = [theta, phi, psi]
    return eangles

a = inversek(U)
print(a)