#!/usr/bin/env python3
"""Drop a Unitree Go2 into a sim-ready scene and check it stands on the ground.

Headless Isaac Sim / Isaac Lab. The robot is spawned 0.5 m above the ground,
held at its default joint targets for --hold seconds, then given a forward
shove. Pass criteria:

  * it lands and settles (base height stable, ground_z + 0.25..0.45 m),
  * it does not sink through the collision mesh,
  * it stays upright (|roll|, |pitch| < 15 deg),
  * the shove decays (friction from ground contact), it does not glide off.

    python scripts/physics_check_go2.py --usd /path/to/World0.usd --headless
    python scripts/physics_check_go2.py --usd .../export_version.usd --spawn -717,490,0.0 --headless
"""
import argparse
import math
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--usd", required=True)
parser.add_argument("--spawn", type=str, default=None,
                    help="x,y,ground_z to drop the robot at. Default: centre of the first 'Walkable' mesh.")
parser.add_argument("--hold", type=float, default=3.0, help="Seconds to stand before the shove.")
parser.add_argument("--dt", type=float, default=1 / 200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
_GLB_EXT = "--enable omni.kit.asset_converter"  # CraftBench payloads .glb (see render_scene_preview.py)
args.kit_args = f"{args.kit_args} {_GLB_EXT}".strip() if args.kit_args else _GLB_EXT

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # noqa: E402


def find_spawn(stage: Usd.Stage):
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and "Walkable" in str(prim.GetPath()) and "NonWalkable" not in str(prim.GetPath()):
            b = bc.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = b.GetMin(), b.GetMax()
            return ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mx[2]), str(prim.GetPath())
    raise SystemExit("no 'Walkable' mesh found - pass --spawn x,y,ground_z")


def main() -> None:
    ctx = omni.usd.get_context()
    if not ctx.open_stage(args.usd):
        raise SystemExit(f"could not open {args.usd}")
    stage = ctx.get_stage()
    for _ in range(4):
        simulation_app.update()

    if args.spawn:
        x, y, gz = (float(v) for v in args.spawn.split(","))
        where = "--spawn"
    else:
        (x, y, gz), where = find_spawn(stage)
    print(f"[go2] spawn over ({x:.1f}, {y:.1f}), ground z={gz:.2f}  ({where})")

    sim = SimulationContext(SimulationCfg(dt=args.dt, device="cuda:0", enable_scene_query_support=True))
    sim_utils.DomeLightCfg(intensity=1000.0).func("/World/PhysicsCheckLight", sim_utils.DomeLightCfg(intensity=1000.0))

    cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Go2")
    cfg.init_state.pos = (x, y, gz + 5.0)  # parked high; moved onto the probed ground below
    robot = Articulation(cfg)
    sim.reset()
    robot.reset()
    default_q = robot.data.default_joint_pos.clone()

    # Probe the real collision surface with a PhysX raycast (bbox tops lie for
    # curved lanes / curbs): straight down from 3 m above the guess.
    from omni.physx import get_physx_scene_query_interface
    sq = get_physx_scene_query_interface()
    profile = []
    for dy in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        h = sq.raycast_closest([x, y + dy, gz + 3.0], [0.0, 0.0, -1.0], 50.0)
        profile.append(f"{dy:+.0f}m:{h['position'][2]:.2f}" if h["hit"] else f"{dy:+.0f}m:none")
    print("[go2] ground profile across y:", "  ".join(profile))
    hit = sq.raycast_closest([x, y, gz + 3.0], [0.0, 0.0, -1.0], 50.0)
    if hit["hit"]:
        gz = hit["position"][2]
        print(f"[go2] raycast ground z={gz:.3f} on {hit['rigidBody']}")
    else:
        print("[go2] raycast found NO collision surface below the spawn point")
    pose = robot.data.root_pose_w.clone()
    pose[0, :3] = torch.tensor([x, y, gz + 0.5], device=pose.device)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))

    def base():
        pos = robot.data.root_pos_w[0]
        r, p, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        r = (r[0].item() + math.pi) % (2 * math.pi) - math.pi
        p = (p[0].item() + math.pi) % (2 * math.pi) - math.pi
        return pos[2].item() - gz, math.degrees(r), math.degrees(p), robot.data.root_lin_vel_w[0].clone()

    def step(n):
        for _ in range(n):
            robot.set_joint_position_target(default_q)
            robot.write_data_to_sim()
            sim.step()
            robot.update(args.dt)

    hold_steps = int(args.hold / args.dt)
    min_h, trace = 9e9, []
    for i in range(hold_steps):
        step(1)
        h, r, p, v = base()
        min_h = min(min_h, h)
        if i % int(0.5 / args.dt) == 0:
            trace.append(h)
            print(f"[go2] t={i * args.dt:4.1f}s  height={h:6.3f} m  roll={r:6.1f}  pitch={p:6.1f}  |v|={v.norm().item():.3f}")
    h_end, r_end, p_end, v_end = base()
    settled = abs(trace[-1] - trace[-2]) < 0.01 if len(trace) >= 2 else False

    # shove: 0.8 m/s forward, see whether friction kills it
    vel = robot.data.root_vel_w.clone()
    vel[0, 0] += 0.8
    robot.write_root_velocity_to_sim(vel)
    x0 = robot.data.root_pos_w[0, 0].item()
    step(int(1.0 / args.dt))
    h2, r2, p2, v2 = base()
    slid = robot.data.root_pos_w[0, 0].item() - x0
    print(f"[go2] after shove: height={h2:.3f} roll={r2:.1f} pitch={p2:.1f} |v|={v2.norm().item():.3f} slid={slid:.2f} m")

    checks = {
        "landed on ground (0.25 < h < 0.45 m)": 0.25 < h_end < 0.45,
        "never sank below ground (min h > 0.05)": min_h > 0.05,
        "settled (height change < 1 cm / 0.5 s)": settled,
        "upright (|roll|,|pitch| < 15 deg)": abs(r_end) < 15 and abs(p_end) < 15,
        "shove decayed by friction (|v| < 0.2 after 1 s)": v2.norm().item() < 0.2,
        "still upright after shove": abs(r2) < 15 and abs(p2) < 15 and 0.2 < h2 < 0.5,
    }
    for k, ok in checks.items():
        print(f"[go2] {'PASS' if ok else 'FAIL'}  {k}")
    print("[go2] RESULT:", "PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()
    simulation_app.close()
