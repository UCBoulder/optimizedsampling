import torch
import dill
import pandas as pd
from geoclip import LocationEncoder

def get_embeddings(latlons):
    gps_encoder = LocationEncoder()
    if not isinstance(latlons, torch.Tensor):
        latlons = torch.from_numpy(latlons).float()

    with torch.no_grad():
        #Tensor nx256, where n is number of lat lon pairs
        embeddings = gps_encoder(latlons)
    print(embeddings.shape)

    return embeddings

def featurize_and_save(latlon_path, out_fpath):
    #Retrieve data
    with open(latlon_path, "rb") as f:
        arrs = dill.load(f)
        
    # get latlons
    latlons = pd.DataFrame(arrs["latlon"], index=arrs["ids_X"], columns=["lat", "lon"])
    lonlats = latlons[["lon", "lat"]]

    # sort
    latlons = latlons.sort_values(["lat", "lon"], ascending=[False, True])
    ids = lonlats.index.to_numpy()

    #Convert to numpy array
    latlons = latlons.values

    emb = get_embeddings(lonlats)

    if isinstance(lonlats, torch.Tensor):
        latlons = latlons.numpy()
    # save
    with open(out_fpath, "wb") as f:
        dill.dump(
            {"emb": emb.numpy(), "ids": ids, "latlon": latlons},
            f,
            protocol=4,
        )

latlon_path = "optimizedsampling/data/int/feature_matrices/CONTUS_UAR.pkl"
out_fpath = "optimizedsampling/data/int/feature_matrices/geoclip_embeddings.pkl"
featurize_and_save(latlon_path, out_fpath)