from PIL import Image
import numpy as np
from PIL.ImageDraw import Draw
from math import cos, sin

canva = Image.new('1', (1000, 1000), 255)

X = 0
Y = 1
Z = 2
W = 3


def draw_line_norm(pic, v1, v2):
    n = norm_vector(v1, v2)
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    print("n", n)
    if dx == 0:
        slope = int(dy / abs(dy))
        for y in range(int(v1[Y]), int(v2[Y]), slope):
            if 0 < y < 1000 and 0 < v1[0] < 1000:
                pic.putpixel((int(y), int(v1[0])), 0)
    elif dy == 0:
        slope = int(dx / abs(dx))
        for x in range(int(v1[X]), int(v2[X]), slope):
            if 0 < x < 1000 and 0 < v1[1] < 1000:
                pic.putpixel((int(v1[1]), int(x)), 0)
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
            dn = n[Y] / n[X]
            y = -x * dn + v1[X]*dn + v1[Y]
            # if y in range(0, 1000):
            #print(x, y)
            if 0 < y < 1000 and 0 < x < 1000:
                pic.putpixel((int(y), int(x)), 0)


def norm_vector(v1, v2):
    n = (v2 - v1) / min(v2 - v1)
    n[1] *= -1
    return n


def project_dot2d(d1, viewer, cam, theta):
    d = d1 - cam

    thX = np.array([[1, 0, 0], [0, cos(theta[X]), sin(theta[X])], [0, -sin(theta[X]), cos(theta[X])]])
    thY = np.array([[cos(theta[Y]), 0, -sin(theta[Y])], [0, 1, 0], [sin(theta[Y]), 0, cos(theta[Y])]])
    thZ = np.array([[cos(theta[Z]), sin(theta[Z]), 0], [-sin(theta[Z]), cos(theta[Z]), 0], [0, 0, 1]])

    print(thX)
    print(thY)
    print(thZ)
    print(d)
    d = thX.dot(thY).dot(thZ).dot(d)
    m = [
        [1, 0, -(viewer[X] / viewer[Z]), 0],
        [0, 1, -(viewer[Y] / viewer[Z]), 0],
        [0, 0, 1, 0],
        [0, 0, 1 / viewer[Z], 0]
    ]
    # m = [
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 1, 0]
    # ]

    # print(d)
    d = np.append(d, 1)
    d = d.reshape(4, 1)
    dot = np.array(m).dot(d)
    # print("dot")
    # print(dot)
    res = np.array([dot[X][0] / dot[W][0], dot[Y][0] / dot[W][0]])
    # print("res")
    # print(res)
    return res


def draw_line3d(pic, d1, d2, cam, theta, viewer):
    p1 = project_dot2d(d1, viewer, cam, theta)
    p2 = project_dot2d(d2, viewer, cam, theta)
    print(str(d1) + " => "+str(p1))
    print(str(d2) + " => " + str(p2))
    draw_line_norm(pic, p1, p2)
    # n = norm_vector(p1, p2)
    # nplus = (n + n * 0.01).astype(np.int32)
    # print(n)
    # print(nplus)
    # draw_line_norm(pic, p1, nplus)



# draw_line_norm(txt, np.array((10, 10)), np.array((10, 245)))
# draw_line_norm(txt, np.array((10, 245)), np.array((245, 245)))
# draw_line_norm(txt, np.array((245, 245)), np.array((245, 10)))
# draw_line_norm(txt, np.array((245, 10)), np.array((10, 10)))

cam = np.array((0, 0, 0))
# cam = np.array((122, 122, 1))
theta = np.array((0, 0, 0))
viewer = np.array((0, 0, 2))
x0 = 100
y0 = 245
dz0 = 1
dz = dz0 + .5
draw_line3d(canva, np.array((x0, x0, dz0)), np.array((x0, y0, dz0)), cam, theta, viewer)
draw_line3d(canva, np.array((x0, y0, dz0)), np.array((y0, y0, dz0)), cam, theta, viewer)
draw_line3d(canva, np.array((y0, y0, dz0)), np.array((y0, x0, dz0)), cam, theta, viewer)
draw_line3d(canva, np.array((y0, x0, dz0)), np.array((x0, x0, dz0)), cam, theta, viewer)

draw_line3d(canva, np.array((x0, x0, dz)), np.array((x0, y0, dz)), cam, theta, viewer)
draw_line3d(canva, np.array((x0, y0, dz)), np.array((y0, y0, dz)), cam, theta, viewer)
draw_line3d(canva, np.array((y0, y0, dz)), np.array((y0, x0, dz)), cam, theta, viewer)
draw_line3d(canva, np.array((y0, x0, dz)), np.array((x0, x0, dz)), cam, theta, viewer)

draw_line3d(canva, np.array((x0, x0, dz0)), np.array((x0, x0, dz)), cam, theta, viewer)
draw_line3d(canva, np.array((y0, y0, dz0)), np.array((y0, y0, dz)), cam, theta, viewer)
draw_line3d(canva, np.array((y0, x0, dz)), np.array((y0, x0, dz0)), cam, theta, viewer)
draw_line3d(canva, np.array((x0, y0, dz)), np.array((x0, y0, dz0)), cam, theta, viewer)




canva.show()
