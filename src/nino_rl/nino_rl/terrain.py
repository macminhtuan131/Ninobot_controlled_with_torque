"""Deterministic hall terrain geometry, independent of ROS/Gazebo imports.

Coordinates are world metres. Permanent end pads cover [-2, 2] and [29, 32];
this replaceable floor covers [2, 29] x [-2, 2], with real recessed potholes.
"""
from math import cos, pi, sin, isfinite
import xml.etree.ElementTree as ET

TERRAIN_REVISION = 1


def mixed_layout(config, rng):
    """Disjoint longitudinal slots prevent overlapping obstacle footprints."""
    cfg = config["mixed_obstacles"]
    kinds = [kind for kind in ("cable", "pothole", "speed_bump")
             for _ in range(_count(cfg, kind + "_count"))]
    if not kinds or len(kinds) > 12:
        raise ValueError("Mixed terrain requires between 1 and 12 obstacles")
    rng.shuffle(kinds)
    spacing = 23.0 / len(kinds)
    layout = []
    for i, kind in enumerate(kinds):
        x = 4.0 + (i + 0.5) * spacing + float(rng.uniform(-0.10, 0.10))
        item = {"kind": kind, "x": x, "y": 0.0}
        if kind == "cable":
            item.update(radius=_draw(cfg, "cable_diameter_range_m", rng, .006, .044) / 2,
                        angle=float(rng.uniform(-pi / 18, pi / 18)))
        elif kind == "pothole":
            item.update(length=_draw(cfg, "pothole_length_range_m", rng, .2, .7),
                        width=_draw(cfg, "pothole_width_range_m", rng, .3, 1.2),
                        depth=_draw(cfg, "pothole_depth_range_m", rng, .005, .04),
                        y=float(rng.uniform(-.15, .15)))
        else:
            item.update(length=_draw(cfg, "speed_bump_length_range_m", rng, .3, .8),
                        width=3.8,
                        height=_draw(cfg, "speed_bump_height_range_m", rng, .005, .04))
        layout.append(item)
    return layout


def _count(cfg, key):
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def _draw(cfg, key, rng, minimum, maximum):
    low, high = map(float, cfg[key])
    if not (minimum <= low <= high <= maximum):
        raise ValueError(f"{key} must be ordered within [{minimum}, {maximum}]")
    return float(rng.uniform(low, high))


def cable_layout(cables):
    return [dict(kind="cable", x=float(x), y=0.0, radius=float(r), angle=float(a))
            for x, r, a in cables]


def floor_boxes(layout):
    """Tile around rectangular apertures; no collision spans a pothole mouth."""
    holes = sorted((o for o in layout if o["kind"] == "pothole"), key=lambda o: o["x"])
    boxes = []
    cursor = 2.0

    def tile(x0, x1, y0, y1, top=0.0):
        if x1 > x0 and y1 > y0:
            boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, (top - .1) / 2,
                          x1 - x0, y1 - y0, top + .1))

    for hole in holes:
        x0, x1 = hole["x"] - hole["length"] / 2, hole["x"] + hole["length"] / 2
        y0, y1 = hole["y"] - hole["width"] / 2, hole["y"] + hole["width"] / 2
        if not (cursor <= x0 < x1 <= 29 and -2 < y0 < y1 < 2 and 0 < hole["depth"] < .1):
            raise ValueError("Potholes must fit the replaceable floor and have disjoint X spans")
        tile(cursor, x0, -2, 2)
        tile(x0, x1, -2, y0)
        tile(x0, x1, y1, 2)
        tile(x0, x1, y0, y1, -hole["depth"])
        cursor = x1
    tile(cursor, 29, -2, 2)
    return boxes


def _pair(link, name, pose, shape, dimensions, color):
    for tag in ("collision", "visual"):
        node = ET.SubElement(link, tag, name=f"{name}_{tag}")
        ET.SubElement(node, "pose").text = " ".join(f"{v:.9f}" for v in pose)
        geom = ET.SubElement(ET.SubElement(node, "geometry"), shape)
        for key, values in dimensions.items():
            ET.SubElement(geom, key).text = " ".join(f"{v:.9f}" for v in values)
        if tag == "visual":
            material = ET.SubElement(node, "material")
            for key in ("ambient", "diffuse"):
                ET.SubElement(material, key).text = color
        else:
            ode = ET.SubElement(ET.SubElement(ET.SubElement(node, "surface"), "friction"), "ode")
            ET.SubElement(ode, "mu").text = "1.0"
            ET.SubElement(ode, "mu2").text = "1.0"


def terrain_model(layout):
    for item in layout:
        if item["kind"] not in ("cable", "pothole", "speed_bump"):
            raise ValueError("Unknown terrain obstacle")
        if any(not isfinite(float(v)) for k, v in item.items() if k != "kind"):
            raise ValueError("Terrain dimensions must be finite")
    model = ET.Element("model", name="training_terrain")
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", name="terrain")
    for i, (x, y, z, length, width, height) in enumerate(floor_boxes(layout)):
        _pair(link, f"floor_{i}", (x, y, z, 0, 0, 0), "box",
              {"size": (length, width, height)}, "0.42 0.44 0.47 1")
    for i, o in enumerate(layout):
        if o["kind"] == "cable":
            # Endpoints stay inside the side walls even at an angle.
            length = (3.96 - 2 * o["radius"]) / cos(o["angle"])
            _pair(link, f"cable_{i}", (o["x"], o["y"], o["radius"], pi/2, 0, o["angle"]),
                  "cylinder", {"radius": (o["radius"],), "length": (length,)}, "0.06 0.06 0.06 1")
        elif o["kind"] == "speed_bump":
            # 24 thin boxes approximate a rounded cosine profile. The largest
            # vertical step is < 5.3 mm even at the allowed 40 mm maximum.
            segments = 24
            dx = o["length"] / segments
            for j in range(segments):
                height = o["height"] * sin(pi * (j + .5) / segments) ** 2
                x = o["x"] - o["length"] / 2 + (j + .5) * dx
                _pair(link, f"speed_bump_{i}_{j}", (x, o["y"], height/2, 0, 0, 0),
                      "box", {"size": (dx, o["width"], height)},
                      "0.9 0.65 0.05 1" if j % 6 < 3 else "0.08 0.08 0.08 1")
    return model


def terrain_sdf(layout):
    root = ET.Element("sdf", version="1.9")
    root.append(terrain_model(layout))
    return ET.tostring(root, encoding="unicode")
