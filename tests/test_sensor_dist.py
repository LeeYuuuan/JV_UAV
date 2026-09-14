import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# Configuration
# ============================================================

MAP_SIZE = 2400.0

NUM_SENSORS = 320
NUM_CLUSTERS = 9

CLUSTER_RATIO = 0.9
CLUSTER_STD = 120.0

CENTER_MARGIN = 300.0
MIN_CENTER_DISTANCE = 400.0

COVERAGE_RADIUS = 86.6

MAP_SEED = 42


# ============================================================
# Generate hotspot centers
# ============================================================

def generate_cluster_centers(
    rng,
    num_clusters,
    map_size,
    margin,
    min_distance,
    max_attempts=10000,
):
    """
    Generate hotspot centers.

    Requirements:
    1. Each center is at least `margin` away from the map boundary.
    2. The distance between two centers is at least `min_distance`.
    """
    centers = []

    for _ in range(max_attempts):
        candidate = rng.uniform(
            low=margin,
            high=map_size - margin,
            size=2,
        )

        valid = all(
            np.linalg.norm(candidate - center) >= min_distance
            for center in centers
        )

        if valid:
            centers.append(candidate)

        if len(centers) == num_clusters:
            return np.asarray(centers, dtype=np.float32)

    raise RuntimeError(
        "Could not generate enough cluster centers. "
        "Try reducing MIN_CENTER_DISTANCE."
    )


# ============================================================
# Generate sensors inside one hotspot
# ============================================================

def generate_points_for_cluster(
    rng,
    center,
    num_points,
    std,
    map_size,
):
    """
    Generate sensor positions using a 2D Gaussian distribution.

    Points outside the map are discarded and regenerated.
    """
    points = []

    while len(points) < num_points:
        remaining = num_points - len(points)

        # Generate extra candidates to reduce the number of loops.
        candidates = rng.normal(
            loc=center,
            scale=std,
            size=(remaining * 2, 2),
        )

        inside_map = (
            (candidates[:, 0] >= 0.0)
            & (candidates[:, 0] <= map_size)
            & (candidates[:, 1] >= 0.0)
            & (candidates[:, 1] <= map_size)
        )

        valid_candidates = candidates[inside_map]

        for point in valid_candidates[:remaining]:
            points.append(point)

    return np.asarray(points, dtype=np.float32)


# ============================================================
# Generate all sensors
# ============================================================

def generate_sensor_positions():
    rng = np.random.default_rng(MAP_SEED)

    num_clustered = int(round(NUM_SENSORS * CLUSTER_RATIO))
    num_uniform = NUM_SENSORS - num_clustered

    cluster_centers = generate_cluster_centers(
        rng=rng,
        num_clusters=NUM_CLUSTERS,
        map_size=MAP_SIZE,
        margin=CENTER_MARGIN,
        min_distance=MIN_CENTER_DISTANCE,
    )

    # Distribute clustered sensors equally among hotspots.
    cluster_counts = np.full(
        NUM_CLUSTERS,
        num_clustered // NUM_CLUSTERS,
        dtype=int,
    )

    # Assign the remainder to the first few hotspots.
    cluster_counts[: num_clustered % NUM_CLUSTERS] += 1

    clustered_parts = []

    for center, count in zip(cluster_centers, cluster_counts):
        cluster_points = generate_points_for_cluster(
            rng=rng,
            center=center,
            num_points=count,
            std=CLUSTER_STD,
            map_size=MAP_SIZE,
        )

        clustered_parts.append(cluster_points)

    clustered_positions = np.vstack(clustered_parts)

    # Generate the remaining uniformly distributed sensors.
    uniform_positions = rng.uniform(
        low=0.0,
        high=MAP_SIZE,
        size=(num_uniform, 2),
    ).astype(np.float32)

    # Combine and shuffle all sensors.
    sensor_positions = np.vstack(
        [
            clustered_positions,
            uniform_positions,
        ]
    ).astype(np.float32)

    rng.shuffle(sensor_positions)

    return (
        sensor_positions,
        cluster_centers,
        clustered_positions,
        uniform_positions,
        cluster_counts,
    )


# ============================================================
# Generate example UAV positions
# ============================================================

def generate_example_uav_positions(cluster_centers):
    """
    Place several example UAVs near hotspot centers.

    These positions are only used to visualize the UAV coverage radius.
    """
    offsets = np.asarray(
        [
            [0.0, 0.0],
            [60.0, 0.0],
            [-50.0, 50.0],
            [80.0, -40.0],
            [-70.0, -50.0],
        ],
        dtype=np.float32,
    )

    num_demo_uavs = min(len(offsets), len(cluster_centers))

    uav_positions = (
        cluster_centers[:num_demo_uavs]
        + offsets[:num_demo_uavs]
    )

    return np.clip(
        uav_positions,
        0.0,
        MAP_SIZE,
    ).astype(np.float32)


# ============================================================
# Plot
# ============================================================

def plot_sensor_distribution(
    clustered_positions,
    uniform_positions,
    cluster_centers,
    uav_positions,
):
    fig, ax = plt.subplots(figsize=(9, 9))

    # Clustered sensors
    ax.scatter(
        clustered_positions[:, 0],
        clustered_positions[:, 1],
        s=18,
        alpha=0.75,
        color="tab:blue",
        label="Clustered sensors",
        zorder=3,
    )

    # Uniform sensors
    ax.scatter(
        uniform_positions[:, 0],
        uniform_positions[:, 1],
        s=22,
        alpha=0.85,
        color="tab:orange",
        label="Uniform sensors",
        zorder=3,
    )

    # Hotspot centers
    ax.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        s=120,
        marker="x",
        linewidths=2.5,
        color="red",
        label="Hotspot centers",
        zorder=6,
    )

    # Airship position
    airship_position = np.asarray(
        [MAP_SIZE / 2.0, MAP_SIZE / 2.0]
    )

    ax.scatter(
        airship_position[0],
        airship_position[1],
        s=200,
        marker="*",
        color="black",
        edgecolor="white",
        linewidth=0.8,
        label="Airship",
        zorder=8,
    )

    # Colors for example UAVs
    uav_colors = [
        "tab:green",
        "tab:purple",
        "tab:red",
        "tab:brown",
        "tab:pink",
    ]

    # UAV positions and coverage circles
    for uav_id, (position, color) in enumerate(
        zip(uav_positions, uav_colors)
    ):
        coverage_circle = Circle(
            xy=position,
            radius=COVERAGE_RADIUS,
            facecolor=color,
            edgecolor=color,
            alpha=0.20,
            linewidth=2.0,
            zorder=2,
        )

        ax.add_patch(coverage_circle)

        # Draw UAV
        ax.scatter(
            position[0],
            position[1],
            s=90,
            marker="^",
            color=color,
            edgecolor="black",
            linewidth=0.9,
            label=(
                "UAV and coverage area"
                if uav_id == 0
                else None
            ),
            zorder=7,
        )

        # UAV label
        ax.text(
            position[0] + 25.0,
            position[1] + 25.0,
            f"UAV {uav_id}",
            fontsize=9,
            color=color,
            weight="bold",
            zorder=9,
        )

    ax.set_xlim(0.0, MAP_SIZE)
    ax.set_ylim(0.0, MAP_SIZE)
    ax.set_aspect("equal")

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")

    ax.set_title(
        f"Sensor Distribution on a "
        f"{MAP_SIZE:.0f} m × {MAP_SIZE:.0f} m Map\n"
        f"{NUM_SENSORS} sensors, "
        f"{NUM_CLUSTERS} hotspots, "
        f"cluster std = {CLUSTER_STD:.0f} m, "
        f"coverage radius = {COVERAGE_RADIUS:.1f} m"
    )

    ax.grid(
        visible=True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    ax.legend(
        loc="upper right",
        framealpha=0.95,
    )

    plt.tight_layout()

    output_path = "sensor_distribution_with_uav_coverage.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Figure saved to: {output_path}")

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    (
        sensor_positions,
        cluster_centers,
        clustered_positions,
        uniform_positions,
        cluster_counts,
    ) = generate_sensor_positions()

    uav_positions = generate_example_uav_positions(
        cluster_centers
    )

    print("Map size:", MAP_SIZE, "m ×", MAP_SIZE, "m")
    print("All sensors:", sensor_positions.shape)
    print("Clustered sensors:", clustered_positions.shape)
    print("Uniform sensors:", uniform_positions.shape)
    print("Cluster centers:", cluster_centers.shape)
    print("Sensors per cluster:", cluster_counts)
    print("Example UAV positions:")
    print(uav_positions)

    plot_sensor_distribution(
        clustered_positions=clustered_positions,
        uniform_positions=uniform_positions,
        cluster_centers=cluster_centers,
        uav_positions=uav_positions,
    )