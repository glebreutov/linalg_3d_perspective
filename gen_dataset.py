from random import randrange

import numpy as np

from draw_func import project_dot2d




def scene():
    viewer = [0, 0, -10]
    theta = (0, 0, 0)
    cam1 = [50, 75, 1]
    cam2 = [50, 25, 1]
    return cam1, cam2, viewer, theta


def data_gen(examples):

    # cam = np.array((122, 122, 1))

    for x in range(0, examples):
        while True:
            d1, res = gen_frame()

            if 1000 < max(res[0:4]) or min(res[0:4]) <= 0:
                continue
            else:
                yield res, d1
                break


def gen_frame():

    # cam1 = [randrange(1, 10), randrange(1, 10), randrange(1, 9)]
    # cam2 = [randrange(1, 10), randrange(1, 10), randrange(1, 9)]
    #seed()
    cam1, cam2, viewer, theta = scene()
    d1 = [randrange(10, 100), randrange(10, 100), randrange(50, 100)]
    d2 = [d1[0]+5, d1[1], d1[2]]

    proj1 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam1), np.array(theta))
    proj2 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam2), np.array(theta))

    proj3 = project_dot2d(np.array(d2), np.array(viewer), np.array(cam1), np.array(theta))
    proj4 = project_dot2d(np.array(d2), np.array(viewer), np.array(cam2), np.array(theta))

    res = list(proj1) + list(proj2) + list(proj3) + list(proj4)
    return d1, res


def data_gen_batch(batch_size):
    inp = []
    out = []
    for x, y in data_gen(batch_size):
        inp.append(x)
        out.append(y)

    return np.array(inp).astype(np.float32) / 100, np.array(out).astype(np.float32) / 100

# for i in range(0, 100):
#     print(gen_frame())