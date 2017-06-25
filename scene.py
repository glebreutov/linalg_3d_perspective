import numpy as np
from math import cos, sin

X = 0
Y = 1
Z = 2
W = 3


def translation(origin, direction):
    thX = np.array([[1, 0, 0], [0, cos(direction[X]), sin(direction[X])], [0, -sin(direction[X]), cos(direction[X])]])
    thY = np.array([[cos(direction[Y]), 0, -sin(direction[Y])], [0, 1, 0], [sin(direction[Y]), 0, cos(direction[Y])]])
    thZ = np.array([[cos(direction[Z]), sin(direction[Z]), 0], [-sin(direction[Z]), cos(direction[Z]), 0], [0, 0, 1]])
    rotation = thX.dot(thY).dot(thZ)
    rotation = rotation.transpose()
    cam = -rotation.dot(origin.reshape(3, 1))
    trans = np.hstack([rotation, cam])
    trans = np.vstack([trans, np.zeros(4)])
    trans[3, 3] = 1
    return trans


def intrinsic(focal, aspect_ratio=0.1, skew=0):
    return np.array([[focal, skew,  aspect_ratio],
                     [0,     focal, aspect_ratio],
                     [0,     0,     1]])


def projection(trans_matr, intrins_matr, point):
    homo_point = np.append(point, 1).reshape(4, 1)
    translated = trans_matr.dot(homo_point)
    proj = intrins_matr.dot(translated[0:3])
    proj /= proj[2]

    # homo_point = np.append(intrins_matr.dot(point), 1)
    # translated = trans_matr.dot(homo_point)
    reshape = proj[0:2].astype(np.int32).reshape(2)
    print(reshape)
    return reshape
