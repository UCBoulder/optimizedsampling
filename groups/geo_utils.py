import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.geometry import box, Point
from shapely.ops import nearest_points

def get_nearest_polygon_index(point, gdf, buffer_degrees=0.5):
    assert gdf.crs.to_string() == "EPSG:4326"
    assert isinstance(point, Point)
    lon, lat = point.x, point.y
    bbox = box(lon - buffer_degrees, lat - buffer_degrees, lon + buffer_degrees, lat + buffer_degrees)
    candidates = gdf[gdf.intersects(bbox)]
    if candidates.empty:
        candidates = gdf
    distances = candidates.geometry.apply(
        lambda poly: geodesic(
            (point.y, point.x),
            (nearest_points(point, poly)[1].y, nearest_points(point, poly)[1].x)
        ).meters
    )
    return distances.idxmin()

def counts_per_division(gdf, division_col):
    counts = gdf[division_col].value_counts()
    print(f"\n=== {division_col} ===")
    print(f"Average: {counts.mean():.2f}")
    print(f"Median:  {counts.median()}")
    print(f"Min:     {counts.min()}")
    print(f"Max:     {counts.max()}")
    return counts

def plot_points_distribution(label, counts, division_type, division_col, log_scale=False):
    plt.figure(figsize=(10, 6))
    counts.plot(kind='hist', bins=100, alpha=0.7, color='skyblue')
    plt.title(f'Distribution of Number of Points per {division_col}', fontsize=14)
    plt.xlabel('Number of Points')
    plt.ylabel('Number of Divisions')
    plt.yscale('log')
    if log_scale:
        plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{label}/plots/{division_type}_hist.png", dpi=300)
