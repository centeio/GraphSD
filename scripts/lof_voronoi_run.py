import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from graphsd.outlier import OutlierSDMining, voronoi_finite_polygons_2d

from graphsd.datasets import load_data
from graphsd.utils import make_bins
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler


# Main test routine
def main():
    position_data, social_df = load_data("playground_a")
    position_data["time"] = pd.to_datetime(position_data["time"])
    social_df = make_bins(social_df, n_bins=3, strategy="quantile")

    miner = OutlierSDMining()
    area_df = miner.get_voronoi_areas(position_data, social_df, time_step=1)  # internally converted to Timedelta
    lof_df = OutlierSDMining.get_lof(position_data, social_df, time_step=1)  # internally converted to Timedelta

    merged = pd.merge(lof_df, area_df, on=["id", "timestamp"], how="outer")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot: mean Voronoi area and LOF per individual
    means = merged.groupby("id")[["area", "lof"]].mean().reset_index()
    print(means)
    scaler = MinMaxScaler()

    means[["area_norm", "lof_norm"]] = scaler.fit_transform(means[["area", "lof"]])
    means = means.sort_values(by="lof_norm", ascending=False).reset_index(drop=True)
    print(means)
    width = 0.35
    x = np.arange(len(means["id"]))

    axes[0].bar(x - width / 2, means["area_norm"], width, alpha=0.6, label="Mean Voronoi Area (normalised)")
    axes[0].bar(x + width / 2, means["lof_norm"], width, alpha=0.6, label="Mean LOF (normalised)")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(means["id"])
    axes[0].set_title("Mean Voronoi Area and LOF per ID")
    axes[0].set_xlabel("ID")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    # Snapshot of random time point showing positions
    random_ts = position_data["time"].sample(1, random_state=0).values[0]
    snapshot = position_data[position_data["time"] == random_ts]

    axes[1].scatter(snapshot["x"], snapshot["y"], alpha=0.7, color="darkblue", edgecolor="white", s=60)

    # Compute and overlay Voronoi diagram
    try:
        points = snapshot[["x", "y"]].values
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        areas = miner.compute_voronoi_areas(points)
        norm = colors.Normalize(vmin=min(areas), vmax=max(areas))
        cmap = plt.get_cmap("viridis")

        for i, region in enumerate(regions):
            polygon = vertices[region]
            poly = Polygon(np.append(polygon, [polygon[0]], axis=0))
            if not poly.is_valid:
                continue
            x, y = poly.exterior.xy
            facecolor = cmap(norm(areas[i]))
            axes[1].fill(x, y, alpha=0.6, edgecolor="black", facecolor=facecolor)
    except Exception as e:
        print(f"Could not overlay Voronoi diagram: {e}")
    axes[1].set_title(f"Position snapshot at t={random_ts}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
