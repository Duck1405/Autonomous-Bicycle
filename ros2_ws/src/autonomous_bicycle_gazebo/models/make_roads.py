import math
from pathlib import Path

ROAD_WIDTH = 7.0
EDGE_LINE_WIDTH = 0.12
CENTER_LINE_WIDTH = 0.12
Z_ROAD = 0.02
Z_LINE = 0.035

OUT = Path("curved_road")
MESH = OUT / "meshes"
MESH.mkdir(parents=True, exist_ok=True)


def sample_centerline():
    points = []

    # Straight section along +x
    for i in range(41):
        x = -40.0 + i * 1.0
        y = 0.0
        points.append((x, y))

    # 90-degree left turn
    # Start at (0, 0), tangent initially +x
    # Turn radius = 20 m
    radius = 20.0
    cx = 0.0
    cy = radius

    for i in range(1, 91):
        theta = -math.pi / 2.0 + (math.pi / 2.0) * (i / 90.0)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        points.append((x, y))

    # Straight section after turn, now along +y
    last_x, last_y = points[-1]
    for i in range(1, 41):
        x = last_x
        y = last_y + i * 1.0
        points.append((x, y))

    return points


def compute_normals(points):
    normals = []

    for i in range(len(points)):
        if i == 0:
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
        elif i == len(points) - 1:
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
        else:
            x0, y0 = points[i - 1]
            x1, y1 = points[i + 1]

        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)

        if length < 1e-9:
            normals.append((0.0, 1.0))
            continue

        tx = dx / length
        ty = dy / length

        # Left normal
        nx = -ty
        ny = tx
        normals.append((nx, ny))

    return normals


def make_strip(points, width, z):
    normals = compute_normals(points)
    left = []
    right = []

    half = width / 2.0

    for (x, y), (nx, ny) in zip(points, normals):
        left.append((x + nx * half, y + ny * half, z))
        right.append((x - nx * half, y - ny * half, z))

    return left, right


def write_obj(path, left, right):
    vertices = left + right

    with path.open("w") as f:
        f.write("# generated curved road mesh\n")

        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        n = len(left)

        for i in range(n - 1):
            a = i + 1
            b = i + 2
            c = n + i + 2
            d = n + i + 1

            f.write(f"f {a} {b} {c}\n")
            f.write(f"f {a} {c} {d}\n")


def write_centerline_csv(path, points):
    with path.open("w") as f:
        f.write("x,y,z\n")
        for x, y in points:
            f.write(f"{x:.6f},{y:.6f},{Z_LINE:.6f}\n")


centerline = sample_centerline()

road_left, road_right = make_strip(centerline, ROAD_WIDTH, Z_ROAD)
write_obj(MESH / "road.obj", road_left, road_right)

left_edge_center = []
right_edge_center = []
normals = compute_normals(centerline)

for (x, y), (nx, ny) in zip(centerline, normals):
    left_edge_center.append((x + nx * (ROAD_WIDTH / 2.0 - 0.45), y + ny * (ROAD_WIDTH / 2.0 - 0.45)))
    right_edge_center.append((x - nx * (ROAD_WIDTH / 2.0 - 0.45), y - ny * (ROAD_WIDTH / 2.0 - 0.45)))

left_line_l, left_line_r = make_strip(left_edge_center, EDGE_LINE_WIDTH, Z_LINE)
right_line_l, right_line_r = make_strip(right_edge_center, EDGE_LINE_WIDTH, Z_LINE)
center_line_l, center_line_r = make_strip(centerline, CENTER_LINE_WIDTH, Z_LINE)

write_obj(MESH / "left_edge.obj", left_line_l, left_line_r)
write_obj(MESH / "right_edge.obj", right_line_l, right_line_r)
write_obj(MESH / "centerline.obj", center_line_l, center_line_r)
write_centerline_csv(MESH / "centerline.csv", centerline)

print(f"Wrote road files to {MESH.resolve()}")