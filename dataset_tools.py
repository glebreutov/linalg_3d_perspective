from random import randrange

import numpy as np

from scene import projection


def gen_dot(max_rand):
    d1 = [randrange(1, max_rand), randrange(1, max_rand), randrange(1, max_rand)]
    return d1


def gen_example(dot, intr, extr1, extr2):
    p1 = projection(extr1, intr, dot)
    p2 = projection(extr2, intr, dot)

    return np.array(dot), p1, p2


def gen_batch(intr, extr1, extr2, stdev, examples):
    x = []
    y = []
    while len(x) < examples:
        dot = gen_dot(stdev)
        d, p1, p2 = gen_example(dot, intr, extr1, extr2)

        concat = np.append(p1, p2)
        if min(concat) > 0 and max(concat) < 1000:
            x.append(concat / stdev)
            y.append(d / stdev)
    return x, y