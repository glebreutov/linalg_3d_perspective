import random
import numpy as np
from PIL import Image

from draw_func import norm_vector, draw_line_norm


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
        if abs(vec_x_norm) < 1:
            possible_line.append(x)
    return possible_line


def np_contains(val, arr):
    for x in arr:
        if np.array_equal(x, val):
            return True

    return False


def filter_line(line, min_dots=20, min_dist_bw_points=3):
    def min_dist(point):
        md = 99999999
        pva = (point[0] + point[1]) /2
        for x in line:
            dist_val = (x[0] + x[1]) / 2
            md = min(md, abs(pva-dist_val))

        return md

    if len(line) < min_dots:
        return ()
    filtered_line = [x for x in line if min_dist(x) <= min_dist_bw_points]
    return max(filtered_line, key=lambda x: x[0] + x[1]), min(filtered_line, key=lambda x: x[0] + x[1])


def find_lines(pixels):
    lines = []
    while len(pixels) > 0:
        line = find_line(pixels)
        fline = filter_line(line)
        if len(fline) == 2:
            lines.append(fline)
        #lines.append(line)
        pixels = [x for x in pixels if not np_contains(x, line)]
        #print(len(pixels))
    return lines


img = load_img('cube.bmp')
pixels = get_pixels(img)
print(len(pixels))
lines = find_lines(pixels)
#print(lines)

canva = Image.new('1', (1000, 1000), 255)
for x in lines:
    draw_line_norm(canva, x[0], x[1])

canva.show()