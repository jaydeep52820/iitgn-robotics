import numpy as np


def cp(a, b):
    c = np.array([[0, a[2], a[1]], [-a[2], 0, a[0]], [-a[1], -a[0], 0]])
    return np.matmul(c, b)


# Link Parameters:    [a, alpha, d, theta]
# For Multiple Links: [[a, alpha, d, theta]
#                      [a, alpha, d, theta]
#                      [a, alpha, d, theta]]

def atransformation(LinkParameters):
    a = LinkParameters[0]
    d = LinkParameters[2]
    alpha = LinkParameters[1]
    theta = LinkParameters[3]
    A = np.array([[[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.cos(alpha), a * np.cos(theta)],
                   [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                   [0, np.sin(alpha), np.cos(alpha), d],
                   [0, 0, 0, 1]]])
    return (A)


def dhconvention(n, LinkParameters):
    A = np.zeros([n, 4, 4])
    for x in range(n):
        A[x-1] = atransformation(LinkParameters[n-1])
    T = np.zeros([n, 4, 4])
    for x in range(n):
        B = np.identity(4)
        for i in range(x+1):
            B = np.matmul(B, A[i])
        T[x] = B
    return T

# Optional argument to define linear joints, can be added as :
# [0, 0, 0, 1, 1]
# [R, R, R, L, L]

def jacobian(n, T, type=0):
    o = np.zeros([n, 4])
    for i in range(n):
        o[i] = np.matmul(T[i], np.array([0, 0, 0, 1]))

    o = np.delete(o, 3, 1)
    z = np.zeros([n, 4])
    for j in range(n):
        z[j] = np.matmul(T[j], np.array([0, 0, 1, 0]))

    z = np.delete(z, 3, 1)

    J = np.zeros([6, n])
    for k in range(n):
        if type == 0:
            c1 = cp(z[k], (o[n - 1] - o[k]))
            c2 = z[k]
        else:
            if type[k] == 1:
                c1 = z[k]
                c2 = [0, 0, 0]
            else:
                c1 = cp(z[k], (o[n - 1] - o[k]))
                c2 = z[k]

        J[:, k] = np.hstack((c1, c2))
    return J


# DH Parameter for 3D Printer

#           a       alpha   d    theta
# Link 1    0       0       d1    0
# Link 2    d2      0       0     0
# Link 3    d3      0       0     -pi/2


RRPParameters = [[0, 0, 1, 0], [2, 0, 0, 0], [3, 0, 0, -np.pi/2]]

a = dhconvention(3, RRPParameters)
b = jacobian(3, a, [1, 1, 1])
print(b, a[2])
