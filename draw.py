import numpy as np

from draw_func import draw_line3d, new_canva, project_dot2d, draw_line_norm

canva = new_canva()

# draw_line_norm(txt, np.array((10, 10)), np.array((10, 245)))
# draw_line_norm(txt, np.array((10, 245)), np.array((245, 245)))
# draw_line_norm(txt, np.array((245, 245)), np.array((245, 10)))
# draw_line_norm(txt, np.array((245, 10)), np.array((10, 10)))

cam = np.array((500, 500, 2))
# cam = np.array((122, 122, 1))
viewer = np.array((0, 0, -2))
theta = np.array((0, 0, 0))
# viewer = np.array((0, 0, 2))


def cube_edges():
    x0 = 100
    y0 = 245
    dz0 = 5
    dz = dz0 + .5
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


def project(values, viewer, cam, theta):
    return [(project_dot2d(l1, viewer, cam, theta), project_dot2d(l2, viewer, cam, theta)) for l1, l2 in values]


def draw_cube(canva, color, viewer, cam, theta):
    for l1, l2 in project(cube_edges(), viewer, cam, theta):
        draw_line_norm(canva, l1, l2, color)


if __name__ == "__main__":
    test_point = cube_edges()[3][0]
    print(test_point, "cam1", project_dot2d(test_point, viewer, np.array((500, 500, 1)), theta))
    print(test_point, "cam2", project_dot2d(test_point, viewer, np.array((400, 400, 2)), theta))
    draw_cube(canva, 'blue', viewer, np.array((500, 500, 1)), theta)
    draw_cube(canva, 'red', viewer, np.array((400, 400, 2)), theta)
    # print(cube_edges(viewer, cam, theta))
    # canva.save("cube.bmp")
    canva.show()
