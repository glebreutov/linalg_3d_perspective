from math import cos, sin

import numpy as np
from PIL import ImageColor, Image


def draw_line_norm(pic, v1, v2, color):
    crgb = ImageColor.getcolor(color, 'RGB')
    n = norm_vector(v1, v2)
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    # print("n", n)
    if dx == 0:
        slope = int(dy / abs(dy))
        for y in range(int(v1[Y]), int(v2[Y]), slope):
            if 0 < y < 1000 and 0 < v1[0] < 1000:
                pic.putpixel((int(y), int(v1[0])), crgb)
    elif dy == 0:
        slope = int(dx / abs(dx))
        for x in range(int(v1[X]), int(v2[X]), slope):
            if 0 < x < 1000 and 0 < v1[1] < 1000:
                pic.putpixel((int(v1[1]), int(x)), crgb)
    else:
        # slope = int(dy / abs(dy))
        # for y in range(v1[Y], v2[Y], slope):
        #     x = (v1[0] * n[0] - y * n[1] + v1[1] * n[1]) / n[1]
        #     # if y in range(0, 1000):
        #     # print(y)
        #     if 0 < y < 1000:
        #         pic.putpixel((int(y), int(x)), 0)
        slope = int(dx / abs(dx))
        for x in range(int(v1[X])*10, int(v2[X]) * 10, slope):
            # y = (-x * n[X] + v1[X]*n[X] + v1[Y] * n[Y]) / n[Y]
            x = x / 10
            dn = n[X] / n[Y]
            y = -x * dn + v1[X]*dn + v1[Y]
            # if y in range(0, 1000):
            #print(x, y)
            if 0 < y < 1000 and 0 < x < 1000:
                pic.putpixel((int(y), int(x)), crgb)


def norm_vector(v1, v2):
    flipud = np.flipud(v2 - v1)
    flipud[0] = -flipud[0]
    return flipud


X = 0
Y = 1
Z = 2
W = 3


def project_dot2d(d1, viewer, cam, theta):
    thX = np.array([[1, 0, 0], [0, cos(theta[X]), sin(theta[X])], [0, -sin(theta[X]), cos(theta[X])]])
    thY = np.array([[cos(theta[Y]), 0, -sin(theta[Y])], [0, 1, 0], [sin(theta[Y]), 0, cos(theta[Y])]])
    thZ = np.array([[cos(theta[Z]), sin(theta[Z]), 0], [-sin(theta[Z]), cos(theta[Z]), 0], [0, 0, 1]])
    m = np.array([
        [1, 0, -(viewer[X] / viewer[Z]), 0],
        [0, 1, -(viewer[Y] / viewer[Z]), 0],
        [0, 0, 1, 0],
        [0, 0, 1 / viewer[Z], 0]
    ])

    d = d1 - cam
    d = thX.dot(thY).dot(thZ).dot(d)

    d = np.append(d, 1).reshape(4, 1)

    dot = m.dot(d)
    res = np.array([dot[X][0] / dot[W][0], dot[Y][0] / dot[W][0]]).astype(np.int32)
    # print("res")
    # print(d1, "=>", res)
    return res


def calc_3d_from_2d(point, viewer, dz):
    # d = (d1 - cam).dot(m)
    # d /= d[W][0]
    pass


def draw_line3d(pic, d1, d2, cam, theta, viewer, color='blue'):
    p1 = project_dot2d(d1, viewer, cam, theta)
    p2 = project_dot2d(d2, viewer, cam, theta)
    # print(str(d1) + " => "+str(p1))
    # print(str(d2) + " => " + str(p2))
    print(p1, p2)
    draw_line_norm(pic, p1, p2, color)
    # n = norm_vector(p1, p2)
    # nplus = (n + n * 0.01).astype(np.int32)
    # print(n)
    # print(nplus)
    # draw_line_norm(pic, p1, nplus)


def new_canva(size=1000):
    return Image.new('RGB', (size, size), ImageColor.getcolor('white', 'RGB'))