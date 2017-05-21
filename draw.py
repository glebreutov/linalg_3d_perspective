from PIL import Image
import numpy as np
txt = Image.new('1', (500, 500), 255)

txt.putpixel((255, 255), 0)


def draw_line_norm(pic, v1, v2):
    n = norm_vector(v1, v2)
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    if dx == 0:
        slope = int(dy / abs(dy))
        for y in range(v1[1], v2[1], slope):
            pic.putpixel((int(v1[0]), int(y)), 0)
    elif dy == 0:
        slope = int(dx / abs(dx))
        for x in range(v1[0], v2[0], slope):
            pic.putpixel((int(x), int(v1[1])), 0)
    else:
        slope = int(dx / abs(dx))
        for x in range(v1[0], v2[0], slope):
            c = -v1[0] * n[0] - v1[1]*n[1]
            y = (-x * n[0])/n[1] - c * n[1]
            pic.putpixel((int(x), int(y)), 0)



def draw_line(pic, v1, v2):
    v = v2-v1
    for t in range(1, (v2[0]-v1[0]) * 100):
        dv = v1 + t * v / 100
        # x = v1[0] + t*(v2[0] - v1[0])
        # y = int(v1[1] + t*(v2[1] - v1[1]))
        dv = dv.astype(np.int32)
        pic.putpixel(dv, 0)


def norm_vector(v1, v2):
    n = v2 - v1
    n[1] *= -1
    return n


# draw_line(txt, np.array((10, 10)), np.array((50, 10)))
# draw_line_norm(txt, np.array((10, 10)), np.array((50, 50)))
draw_line_norm(txt, np.array((10, 10)), np.array((10, 50)))
draw_line_norm(txt, np.array((10, 50)), np.array((50, 50)))
draw_line_norm(txt, np.array((50, 50)), np.array((50, 10)))
draw_line_norm(txt, np.array((50, 10)), np.array((10, 10)))
draw_line_norm(txt, np.array((10, 10)), np.array((50, 50)))
draw_line_norm(txt, np.array((50, 10)), np.array((10, 50)))

#draw_line_norm(txt, np.array((50, 50)), np.array((10, 50)))
# draw_line(txt, np.array((10, 10)), np.array((70, 30)))
# draw_line(txt, np.array((50, 10)), np.array((50, 50)))
#draw_line(txt, np.array((10, 50)), np.array((50, 100)))
txt.show()
