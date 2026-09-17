"""Collision geometry, repeatable curricula and reset transport regressions."""
import ast
from copy import deepcopy
from math import cos, sin
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from nino_rl.terrain import mixed_layout, floor_boxes, terrain_sdf
from nino_rl.evaluation import compare_summaries, summarize, METRICS

ROOT = Path(__file__).parents[1]
CONFIG = yaml.safe_load((ROOT / "config/ppo.yaml").read_text())["terrain_curriculum"]


def test_repeatable_mixed_layout_and_separated_footprints():
    for seed in range(100):
        layout = mixed_layout(CONFIG, np.random.default_rng(seed))
        assert layout == mixed_layout(CONFIG, np.random.default_rng(seed))
        assert {k: sum(o["kind"] == k for o in layout)
                for k in ("cable", "pothole", "speed_bump")} == {
                    "cable": 4, "pothole": 3, "speed_bump": 3}
        previous_right = 2.0
        for o in sorted(layout, key=lambda o: o["x"]):
            if o["kind"] == "cable":
                length = (3.96 - 2 * o["radius"]) / cos(o["angle"])
                half_x = abs(sin(o["angle"])) * length/2 + o["radius"]
                assert length/2 * cos(o["angle"]) + o["radius"] < 2
            else:
                half_x = o["length"]/2
                assert abs(o["y"]) + o["width"]/2 < 2
            assert o["x"] - half_x > previous_right
            previous_right = o["x"] + half_x
        assert previous_right < 29
        ET.fromstring(terrain_sdf(layout))


def test_potholes_are_recesses_and_floor_has_no_gaps_or_overlaps():
    layout = mixed_layout(CONFIG, np.random.default_rng(42))
    boxes = floor_boxes(layout)
    holes = [o for o in layout if o["kind"] == "pothole"]
    # Every arbitrary XY point has exactly one supporting box, including holes.
    points = [(o["x"], o["y"]) for o in holes]
    points += list(zip(np.random.default_rng(1).uniform(2, 29, 1000),
                       np.random.default_rng(2).uniform(-2, 2, 1000)))
    for x, y in points:
        support = [b for b in boxes if abs(x-b[0]) < b[3]/2 and abs(y-b[1]) < b[4]/2]
        assert len(support) == 1
        hole = next((o for o in holes if abs(x-o["x"]) < o["length"]/2
                     and abs(y-o["y"]) < o["width"]/2), None)
        assert support[0][2] + support[0][5]/2 == pytest.approx(-hole["depth"] if hole else 0)
    assert sum(b[3]*b[4] for b in boxes) == pytest.approx(27*4)
    assert floor_boxes([]) == [(15.5, 0.0, -.05, 27.0, 4, .1)]


def test_preview_floor_has_real_holes_and_permanent_reset_support():
    world = ET.parse(ROOT.parent / "nino_description/worlds/long_hall.sdf").getroot()
    assert world.find(".//collision[@name='floor_collision']") is None
    pads = world.findall(".//model[@name='terrain_floor_v1']//collision")
    assert len(pads) == 2
    assert world.find(".//model[@name='training_terrain']") is not None
    collisions = world.findall(".//collision")
    for x, depth in [(6., .015), (16.4, .02), (21.4, .03)]:
        tops = []
        for c in collisions:
            size = c.findtext("geometry/box/size")
            if size is None:
                continue
            px, py, pz, *_ = map(float, c.findtext("pose").split())
            sx, sy, sz = map(float, size.split())
            if abs(x-px) < sx/2 and abs(py) < sy/2 and pz < 1:
                tops.append(pz+sz/2)
        assert tops == pytest.approx([-depth])


@pytest.mark.parametrize("key,value", [("pothole_depth_range_m", [.02, .2]),
    ("cable_count", -1), ("pothole_count", 2.5), ("speed_bump_count", 20)])
def test_invalid_configuration_rejected_before_scene_changes(key, value):
    cfg = deepcopy(CONFIG)
    cfg["mixed_obstacles"][key] = value
    with pytest.raises(ValueError):
        mixed_layout(cfg, np.random.default_rng(42))


def test_reset_removes_all_old_terrain_and_spawns_one_identity_model():
    # Execute the real transport methods with service fakes; ROS is not needed.
    source = ast.parse((ROOT / "nino_rl/ros_interface.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RosRobotInterface")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("configure_training_terrain", "_terrain_spawn_request")]
    class Service:
        srv_name = "fake"
        def __init__(self): self.calls = []
        def wait_for_service(self, **kw): return True
        def call_async(self, req): self.calls.append(req); return NS(success=True)
    def request():
        return NS(entity=NS(), entity_factory=NS(pose=NS(orientation=NS(w=0))))
    import re, os
    scene_names = ["nino", "terrain_floor_v1", "enclosed_hall", "training_terrain",
                   "cable_bumps", "training_cable_12", "unrelated_model"]
    scene = NS(returncode=0, stdout="".join('model {\n  name: "'+n+'"\n}\n' for n in scene_names))
    scope = dict(re=re, os=os, terrain_sdf=terrain_sdf,
                 subprocess=NS(run=lambda *a, **kw: scene),
                 DeleteEntity=NS(Request=request), SpawnEntity=NS(Request=request), Entity=NS(MODEL=2))
    wrapper = ast.ClassDef(name="Transport", bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "transport", "exec"), scope)
    transport = scope["Transport"]()
    transport.world_name = "long_hall"
    transport.delete_entity = Service()
    transport.spawn_entity = Service()
    transport._wait_future = lambda response, timeout: response
    transport.configure_training_terrain([])
    assert {r.entity.name for r in transport.delete_entity.calls} == {
        "training_terrain", "cable_bumps", "training_cable_12"}
    req, = transport.spawn_entity.calls
    assert req.entity_factory.pose.orientation.w == 1
    assert req.entity_factory.allow_renaming is False
    root = ET.fromstring(req.entity_factory.sdf)
    assert len(root.findall(".//collision")) == 1  # Phase 1 is a flat floor.
    scene.stdout = scene.stdout.replace('terrain_floor_v1', 'old_floor')
    with pytest.raises(RuntimeError, match="rebuild"):
        transport.configure_training_terrain([])
    assert len(transport.spawn_entity.calls) == 1


def test_comparison_rejects_different_actual_terrain():
    row = dict.fromkeys(METRICS, 1.)
    row.update(success=True, finished_within_target_time=True, termination="success")
    a = summarize([row], dict(complete=True, terrain_layouts_sha256="a"))
    b = deepcopy(a)
    b["terrain_layouts_sha256"] = "b"
    with pytest.raises(ValueError, match="terrain_layouts"):
        compare_summaries(a, b)
