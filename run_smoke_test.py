from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from jv_uav import build_smoke_test_runner, load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    # Keep the example short; the configured training/evaluation horizon remains 100.
    cfg["time"]["episode_upper_frames"] = 3
    runner = build_smoke_test_runner(cfg)
    trace = runner.run_episode(reset_rng=True)
    summary = {
        "frames": len(trace.frames),
        "low_samples_plus_initial": len(trace.backlog_max_post_timeline),
        "last_system_max_backlog": trace.backlog_max_post_timeline[-1],
        "sensor_visits": int(trace.sensor_visit_count.sum()),
        "collected_packets": float(trace.sensor_collected_packets.sum()),
        "coverage_grid_visits": int(trace.coverage_grid_step_count.sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
