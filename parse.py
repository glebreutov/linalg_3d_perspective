import random

import functools
import numpy as np
from PIL import Image
from math import sqrt

from draw import cube_edges, cam, viewer, theta, project
from draw_func import norm_vector, draw_line_norm, new_canva


def load_img(path):
    return np.array(Image.open(path))


def get_pixels(img: Image):
    pix = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            coords = img[x, y]
            if not coords:
                pix.append(np.array([x, y]))
    return pix


def find_line(pixels):
    # get two random dots
    v1 = pixels[random.randrange(len(pixels))]
    v2 = pixels[random.randrange(len(pixels))]
    # find n
    n = norm_vector(v1, v2)
    # find all points that (v-v0) * n = 0

    possible_line = []
    for x in pixels:
        vec_x_norm = (x - v1).dot(n)
        if abs(vec_x_norm) < 0.5:
            possible_line.append(x)
    return possible_line


def np_contains(val, arr):
    for x in arr:
        if np.array_equal(x, val):
            return True

    return False


def filter_line(line, min_dots=10, min_dist_bw_points=10):
    if len(line) < min_dots:
        return ()

    # line.sort(key=functools.cmp_to_key(lambda x, y: sqrt(sum((x - y) * (x - y)))))
    # for x in range(1, len(line)):
    #     print(sqrt(sum((line[x-1] - line[x]) * (line[x-1] - line[x]))))

    def min_dist(point):
        md = 99999999
        for x in line:
            if np.array_equal(x, point):
                continue
            md = min(md, sqrt(sum((x - point)*(x - point))))
            if md <= 1:
                break
        #print("md "+str(md))
        return md

    filtered_line = [pnt for pnt in line if min_dist(pnt) <= min_dist_bw_points]

    if len(filtered_line) < min_dots:
        return ()
    # print("max dist", max([min_dist(pnt) for pnt in filtered_line]))
    return max(filtered_line, key=lambda x: x[0] + x[1]), min(filtered_line, key=lambda x: x[0] + x[1])


def find_lines(pixels):
    lines = []
    while len(pixels) > 0:
        line = find_line(pixels)
        # print(line)
        fline = filter_line(line)
        if len(fline) == 2:
            lines.append(fline)
        #lines.append(line)
        pixels = [x for x in pixels if not np_contains(x, line)]
        #print(len(pixels))
    return lines


def validate_results(val_set, act_set):
    for x in act_set:
        mv = 10E5,
        similar = None
        for y in val_set:
            similarity = (abs(sum(x[0] - y[0])) + abs(sum(x[1] - y[1])))
            if similarity < mv:
                mv = similarity
                similar = y

        print(str(x) + " is similar to " + str(similar) + " dist is mv " + str(mv))

validation_set = project(cube_edges(), viewer, cam, theta)
# print(validation_set)


img = load_img('cube.bmp')
pixels = get_pixels(img)
print(len(pixels))
lines = find_lines(pixels)
#print(lines)

canva = new_canva()

#print(lines)
validate_results(validation_set, lines)
for x in lines:
    print(x)
    draw_line_norm(canva, x[0], x[1], 'red')

canva.show()