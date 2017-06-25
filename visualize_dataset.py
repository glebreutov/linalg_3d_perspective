import numpy as np
from PIL import ImageColor

from draw_func import new_canva, project_dot2d
from gen_dataset import data_gen, scene

top_canva = new_canva()
proj1canva = new_canva()
proj2canva = new_canva()
front_canva = new_canva()
cam1, cam2, viewer, theta = scene()

blue = ImageColor.getcolor("blue", 'RGB')
top_canva.putpixel(cam1[1:], blue)
red = ImageColor.getcolor("red", 'RGB')
top_canva.putpixel(cam2[1:], red)


def bold_pixel(canv, px):
    for i in range(-4, 4):
        for j in range(-4, 4):
            try:
                canv.putpixel(list(reversed([px[0] + i, px[1] + j])), 0)
            except IndexError:
                print("broken pixel")


for x, y in data_gen(100):
    bold_pixel(top_canva, y[0:2])
    bold_pixel(proj1canva, x[0:2])
    bold_pixel(proj2canva, x[2:])
    bold_pixel(front_canva, y[1:])


top_canva.show("top view")
front_canva.show("top view")
# proj1canva.show("cam1")
# proj2canva.show("cam2")