import os
import pickle
import geopandas as gpd
from collections import Counter
import numpy as np

base_dir = "/home/libe2152/optimizedsampling/"

def cost_of_cluster(sizes, points_per_cluster):
    return points_per_cluster + 0.1*np.sqrt(sizes)

if __name__ == "__main__":
    for label in ["income", "population", "treecover"]:
        for points_per_cluster in [2,4,6,8,10,12,14,16,18,20]:
            assignment_path = os.path.join(base_dir, "region_assignments", label, "counties.pkl")
            output_path = os.path.join(base_dir, "cost", label, f"county_costs_{points_per_cluster}_points_per_cluster.pkl")

            with open(assignment_path, "rb") as f:
                data = pickle.load(f)

            unique_counties, county_counts = np.unique(data['assignments'], return_counts=True)

            county_costs = cost_of_cluster(county_counts, points_per_cluster)
            county_costs_dict = {str(unique_counties[i]): int(county_costs[i]) for i in range(len(unique_counties))}

            with open(output_path, "wb") as f:
                pickle.dump(county_costs_dict, f)

            print(f"Saved cost mapping for {len(county_costs_dict)} counties to {output_path}")