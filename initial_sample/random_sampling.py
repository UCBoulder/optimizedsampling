import os
import argparse
import dill
import geopandas as gpd
from plotting_utils import plot_sampled_points_on_map

class RandomSampler:
    def __init__(self, gdf_points, id_col):
        self.gdf_points = gdf_points.copy()
        if self.gdf_points.crs != "EPSG:4326":
            self.gdf_points = self.gdf_points.to_crs("EPSG:4326")
        self.id_col = id_col
        self.sampled_gdf = None
        self.sampled_ids = None
        self.sample_size = None
        self.seed = None
        self.out_dir = None

    def sample(self, total_sample_size, seed=None):
        self.sampled_gdf = self.gdf_points.sample(n=total_sample_size, random_state=seed)
        self.sampled_ids = self.sampled_gdf[self.id_col].tolist()
        self.sample_size = len(self.sampled_ids)
        self.seed = seed
        return self.sampled_ids

    def reset_sample(self):
        self.sampled_gdf = None
        self.sampled_ids = None
        self.sample_size = None
        self.seed = None
        self.out_dir = None

    def save_sampled_ids(self, out_path):
        self.out_dir = out_path
        os.makedirs(out_path, exist_ok=True)
        file_name = f"random_sample_{self.sample_size}_points_seed_{self.seed}.pkl"
        self.out_path = os.path.join(out_path, file_name)
        with open(self.out_path, "wb") as f:
            dill.dump(self.sampled_ids, f)

    def plot(self, country_shape_file, country_name=None, exclude_names=None):
        plot_dir = os.path.join(self.out_dir, 'plots')
        save_path = os.path.join(plot_dir, f"random_sample_{self.sample_size}_seed_{self.seed}.png")
        plot_sampled_points_on_map(
            self.gdf_points, self.sampled_gdf, self.sample_size, country_shape_file,
            title=f'Random Sampling ({self.sample_size} points)', save_path=save_path,
            country_name=country_name, exclude_names=exclude_names, sampled_color='#2ca02c',
        )

def main():
    parser = argparse.ArgumentParser(description="Run random sampling over USAVars labels.")
    parser.add_argument("--labels", type=str, nargs="+", default=["population", "treecover"])
    parser.add_argument("--data_dir", type=str, default="../../0_data")
    parser.add_argument("--country_shape_file", type=str,
                        default="../../0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp")
    parser.add_argument("--exclude_names", type=str, nargs="+", default=["Alaska", "Hawaii", "Puerto Rico"])
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=list(range(1100, 3001, 100)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415])
    args = parser.parse_args()

    for label in args.labels:
        gdf = gpd.read_file(f"{args.data_dir}/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson")
        out_path = f"{args.data_dir}/initial_samples/usavars/{label}/random_sampling"

        sampler = RandomSampler(gdf, id_col="id")
        for total_sample_size in args.sample_sizes:
            for seed in args.seeds:
                sampler.sample(total_sample_size=total_sample_size, seed=seed)
                sampler.save_sampled_ids(out_path)
                sampler.plot(args.country_shape_file, exclude_names=args.exclude_names)
                sampler.reset_sample()

if __name__ == '__main__':
    main()
