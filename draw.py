import numpy as np

from draw_func import draw_line3d, new_canva, project_dot2d, draw_line_norm
from scene import translation, intrinsic, projection

canva = new_canva()

# draw_line_norm(txt, np.array((10, 10)), np.array((10, 245)))
# draw_line_norm(txt, np.array((10, 245)), np.array((245, 245)))
# draw_line_norm(txt, np.array((245, 245)), np.array((245, 10)))
# draw_line_norm(txt, np.array((245, 10)), np.array((10, 10)))

def cube_edges():
    x0 = 800 - 145/2
    y0 = x0 + 145
    dz0 = 500
    dz = dz0 + 145
    edges3d = [
        [np.array((x0, x0, dz0)), np.array((x0, y0, dz0))],
        [np.array((x0, y0, dz0)), np.array((y0, y0, dz0))],
        [np.array((y0, y0, dz0)), np.array((y0, x0, dz0))],
        [np.array((y0, x0, dz0)), np.array((x0, x0, dz0))],
        [np.array((x0, x0, dz)), np.array((x0, y0, dz))],
        [np.array((x0, y0, dz)), np.array((y0, y0, dz))],
        [np.array((y0, y0, dz)), np.array((y0, x0, dz))],
        [np.array((y0, x0, dz)), np.array((x0, x0, dz))],
        [np.array((x0, x0, dz0)), np.array((x0, x0, dz))],
        [np.array((y0, y0, dz0)), np.array((y0, y0, dz))],
        [np.array((y0, x0, dz)), np.array((y0, x0, dz0))],
        [np.array((x0, y0, dz)), np.array((x0, y0, dz0))]
    ]
    return edges3d


def draw_cube(canva, color, trans, intr):
    for x1, x2 in cube_edges():
        l1 = projection(trans, intr, np.array(x1))
        l2 = projection(trans, intr, np.array(x2))
        draw_line_norm(canva, l1, l2, color)


if __name__ == "__main__":
    intr = intrinsic(focal=200, aspect_ratio=300, skew=0)

    cam = np.array((475, 500, 2))
    theta = np.array((0.1, 0, 0))
    trans1 = translation(cam, theta)

    draw_cube(canva, 'blue', trans1, intr)

    cam = np.array((525, 500, 2))
    theta = np.array((-0.1, 0, 0))
    trans2 = translation(cam, theta)


    draw_cube(canva, 'red', trans2, intr)

    # test_point = cube_edges()[3][0]
    # print(test_point, "cam1", project_dot2d(test_point, viewer, np.array((500, 500, 1)), theta))
    # print(test_point, "cam2", project_dot2d(test_point, viewer, np.array((400, 400, 2)), theta))
    # draw_cube(canva, 'red', viewer, np.array((400, 400, 2)), theta)
    # print(cube_edges(viewer, cam, theta))
    # canva.save("cube.bmp")
    canva.show()
