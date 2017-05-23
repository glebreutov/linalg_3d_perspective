import numpy as np
from PIL import Image

from draw_func import draw_line3d

canva = Image.new('1', (1000, 1000), 255)

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



#canva.save("cube.bmp")
canva.show()
