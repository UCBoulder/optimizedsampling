import os
import argparse
import dill
from load_points import DATASET_DEFAULTS, load_gdf
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
    parser = argparse.ArgumentParser(description="Run random sampling.")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS, default="usavars")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--country_shape_file", type=str, default=None)
    parser.add_argument("--country_name", type=str, default=None)
    parser.add_argument("--exclude_names", type=str, nargs="+", default=None)
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=list(range(1100, 3001, 100)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415])
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    labels = args.labels or defaults["labels"]
    data_dir = args.data_dir or defaults["data_dir"]
    id_col = args.id_col or defaults["id_col"]
    country_shape_file = args.country_shape_file or defaults["country_shape_file"]
    country_name = args.country_name or defaults["country_name"]
    exclude_names = args.exclude_names or defaults["exclude_names"]

    for label in labels:
        gdf = load_gdf(args.dataset, data_dir, label)
        out_path = f"{data_dir}/initial_samples/{args.dataset}/{label}/random_sampling"

        sampler = RandomSampler(gdf, id_col=id_col)
        for total_sample_size in args.sample_sizes:
            for seed in args.seeds:
                sampler.sample(total_sample_size=total_sample_size, seed=seed)
                sampler.save_sampled_ids(out_path)
                sampler.plot(country_shape_file, country_name=country_name, exclude_names=exclude_names)
                sampler.reset_sample()

if __name__ == '__main__':
    main()
