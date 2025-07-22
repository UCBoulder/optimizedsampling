import pickle
import geopandas as gpd
import pandas as pd

def make_group_assignment(df, columns, id_col):
    """
    Create a dict mapping from row ID to a combined group ID like 'COUNTY_NAME_COUNTYFP'.

    Parameters:
        df (pd.DataFrame): DataFrame with 'id' column and grouping columns
        columns (list of str): Columns to join for the group ID

    Returns:
        dict: Mapping from df['id'] to joined string from selected columns
    """
    group_ids = df[columns].astype(str).agg("_".join, axis=1)
    return dict(zip(df[id_col], group_ids))

def save_dict_to_pkl(d, filepath):
    """
    Save a dictionary to a pickle file.

    Parameters:
        d (dict): Dictionary to save
        filepath (str): Full path to the output .pkl file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(d, f)

if __name__ == "__main__":
    from IPython import embed; embed()
    # for label in ['population', 'treecover']:
    #     gdf = gpd.read_file(f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson")

    #     county_dict = make_group_assignment(gdf, ['combined_county_id'], 'id')
    #     save_dict_to_pkl(county_dict, f"/home/libe2152/optimizedsampling/0_data/groups/usavars/{label}/county_assignments_dict.pkl")

    #     def make_state_dict(county_dict):
    #         """
    #         Takes a county_dict with keys like 'id' and values like 'County_FP_State_Name_FP'
    #         Returns a dict mapping each full county string to its corresponding state string.
    #         """
    #         state_dict = {}
    #         for (id, full_county) in county_dict.items():
    #             # Example: 'Montgomery_097_05_Arkansas'
    #             parts = full_county.split('_')
    #             if len(parts) >= 4:
    #                 county_name, county_fp, state_fp, state_name = parts
    #                 state_key = f"{state_name}_{state_fp}"
    #                 state_dict[id] = state_key
    #         return state_dict

    #     state_dict = make_state_dict(county_dict)
    #     save_dict_to_pkl(state_dict, f"/home/libe2152/optimizedsampling/0_data/groups/usavars/{label}/state_assignments_dict.pkl")