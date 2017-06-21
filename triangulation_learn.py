import tensorflow as tf
import numpy as np
from draw_func import project_dot2d
from random import randrange

def data_gen(examples):
    cam1 = [randrange(1, 1000), randrange(1, 1000), randrange(1, 9)]
    cam2 = [randrange(1, 1000), randrange(1, 1000), randrange(1, 9)]
    # cam = np.array((122, 122, 1))
    viewer = [0, 0, -2]
    theta = (0, 0, 0)

    for x in range(0, examples):
        d1 = [randrange(1, 1000), randrange(1, 1000), randrange(11, 1000)]
        proj1 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam1), np.array(theta))
        proj2 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam2), np.array(theta))

        res = list(proj1) + list(proj2) + cam1 + cam2 + viewer[2:]
        yield np.array(res).astype(np.float32), np.array(d1).astype(np.float32)


FEATURES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


def input_func():
    gen = data_gen(10)
    cols = {f: [] for f in FEATURES}
    ycol = [[], [], []]

    for x, y in gen:
        for n in range(0, len(FEATURES)):
            cols[FEATURES[n]].append(x[n])
        ycol[0].append(y[0])
        ycol[1].append(y[1])
        ycol[2].append(y[2])

    feature_cols = {k: tf.constant(cols[k]) for k in FEATURES}

    #answer_cols = {k: tf.constant(ycol[k]) for k in ("X", "Y", "Z")}
    answer_cols = tf.constant(ycol)

    return feature_cols, answer_cols



tf.logging.set_verbosity(tf.logging.INFO)

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]


regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[11, 45])

regressor.fit(input_fn=input_func, steps=2000)
# xa, ya = input_func()
# regressor.fit(x=xa, y=ya, steps=2000)