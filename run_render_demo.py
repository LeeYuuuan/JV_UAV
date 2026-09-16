"""Generate deterministic diagnostic figures, without training an RL policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from jv_uav import build_smoke_test_runner, load_config
from jv_uav.joint_runner import LowestBatteryRequestPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "render_demo")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--gif", action="store_true", help="Also save a frame-boundary animation (requires Pillow).")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["time"]["episode_upper_frames"] = args.frames
    runner = build_smoke_test_runner(cfg)
    env = runner.upper_env
    obs, _ = env.reset()
    # Use fixed map-derived waypoints for diagnostics only; no hidden state is
    # added to the learned observation contract.
    distance = np.linalg.norm(env.scene.sensor_pos - env.scene.airship_pos, axis=1)
    targets = env.scene.sensor_pos[np.argsort(distance)[:env.scene.num_uavs]].copy()

    def lower(observation):
        ids = observation["active_uav_ids"]
        delta = targets[ids] - observation["active_uav_features"][:, :2]
        norm = np.linalg.norm(delta, axis=1)
        return delta / np.maximum(norm[:, None], env.scene.max_move_distance_m)

    env.low_runner.low_policy = lower
    upper = LowestBatteryRequestPolicy(requests_per_frame=4)
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    if args.gif:
        from PIL import Image
    for _ in range(args.frames):
        obs, _, terminated, truncated, _ = env.step(upper(obs))
        if args.gif:
            fig = env.render(show=False, dpi=85)
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
        if terminated or truncated:
            break
    env.render(save_path=args.output / "dashboard.png", show=args.show)
    env.render(save_path=args.output / "heatmaps.png", show=args.show, view="heatmaps")
    if images:
        images[0].save(args.output / "episode.gif", save_all=True, append_images=images[1:], duration=650, loop=0)
        for image in images:
            image.close()
    (args.output / "trace.json").write_text(json.dumps(env.trace.to_serializable(), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Rendered {len(env.trace.frames)} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
