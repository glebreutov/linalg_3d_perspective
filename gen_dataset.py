from random import randrange

import numpy as np

from draw_func import project_dot2d


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
    viewer = [0, 0, -2]
    theta = (0, 0, 0)
    # cam1 = [randrange(1, 10), randrange(1, 10), randrange(1, 9)]
    # cam2 = [randrange(1, 10), randrange(1, 10), randrange(1, 9)]
    cam1 = [500, 500, 1]
    cam2 = [500, 250, 1]
    d1 = [randrange(1, 100), randrange(1, 100), randrange(11, 100)]
    proj1 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam1), np.array(theta))
    proj2 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam2), np.array(theta))
    res = list(proj1) + list(proj2)
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