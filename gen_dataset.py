from random import randrange

import numpy as np

from draw_func import project_dot2d



def scene():
    viewer = [0, 0, 5]
    theta = (0, 0, 0)
    cam1 = [200, 250, 2]
    cam2 = [300, 250, 2]
    return cam1, cam2, viewer, theta


def data_gen(examples):

    # cam = np.array((122, 122, 1))

    for x in range(0, examples):
        while True:
            d1, res = gen_frame()

            if 1000 <= max(res[0:4]) or min(res[0:4]) <= 0:
                continue
            else:
                yield res, d1
                break


def gen_frame(max_rand=1000):

    # cam1 = [randrange(1, 10), randrange(1, 10), randrange(1, 9)]
    # cam2 = [randrange(1, 10), randrange(1, 10), randrange(1, 9)]
    #seed()
    cam1, cam2, viewer, theta = scene()

    d1 = [randrange(1, max_rand), randrange(1, max_rand), randrange(50, max_rand)]

    proj1 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam1), np.array((0.4, 0, 0)))
    proj2 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam2), np.array((-0.4, 0, 0)))
    # d2 = [d1[0] + 10, d1[1], d1[2]]
    # proj3 = project_dot2d(np.array(d2), np.array(viewer), np.array(cam1), np.array(theta))
    # proj4 = project_dot2d(np.array(d2), np.array(viewer), np.array(cam2), np.array(theta))
    res = list(proj1) + list(proj2) + list(cam1) + list(cam2)
    return d1, res


def data_gen_batch(batch_size):
    inp = []
    out = []
    for x, y in data_gen(batch_size):
        inp.append(x)
        out.append(y)

    return np.array(inp).astype(np.float32) / 1000, np.array(out).astype(np.float32) / 1000

# for i in range(0, 100):
#     print(gen_frame())