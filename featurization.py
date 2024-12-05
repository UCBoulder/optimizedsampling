from pathlib import Path
import torch
import dill
import numpy as np

from torchgeo.models import RCF
from USAVars import USAVars

train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
test = USAVars(root="/share/usavars", split="test", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
val = USAVars(root="/share/usavars", split="val", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

total_num_images = train.__len__() + test.__len__() + val.__len__()
img_height = 256
img_width = 256
num_channels = 4

image_folder = Path("/share/usavars/uar")
#image_folder = Path("test_image")

# def mosaiks_featurization():
#     out_fpath = Path(c.features_dir) / f"{image_folder.name}_mosaiks.pkl"

#     featurize_and_save(image_folder, out_fpath, c)

def torchgeo_featurization(num_features):
    out_fpath = "data/int/feature_matrices/CONTUS_UAR_torchgeo4096.pkl"

    #imgs, ids, latlons = format_data(train, val, test)
    ids, latlons = format_ids_latlons(train, val, test)

    #Torchgeo Random Convolutional Feature Implementation
    #Patch = Kernel = kernel_size x kernel_size over all bands
    rcf = RCF(
        in_channels=4, 
        features=num_features,
        kernel_size=4, #maybe change; patch in mosaiks is 3
        bias=-1.0, 
        seed=42, 
        mode='empirical',
        dataset=train
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcf = rcf.to(device)

    #Forward pass of TorchGeo RCF
    print("Featurizing images...")
    from IPython import embed; embed()

    featurized_imgs = np.empty((total_num_images, num_features), dtype=np.float32)

    for i in range(total_num_images):
        print("Featurizing image ", i)

        if (i< len(train)):
            img = train[i]['image'].to(device)
        if ((i>=len(train)) & ((i-len(train))<len(val))):
            img = val[i-len(train)]['image'].to(device)
        if (i>=(len(train)+len(val))):
            img = test[i-len(train)-len(val)]['image'].to(device)

        featurized_imgs[i] = rcf.forward(img).cpu().numpy()

    with open(out_fpath, "wb") as f:
        dill.dump(
            {"X": featurized_imgs, "ids_X": ids, "latlon": latlons},
            f,
            protocol=4,
        )

'''
    Retrieve images, ids, and latlons from dataset
'''
def retrieve_data(dataset):
    print("Retrieving ", dataset)
    imgs = np.array([dataset[i]['image'] for i in range(len(dataset))])
    ids = np.array([dataset[i]['name'].replace('tile_', '').replace('.tif', '') for i in range(len(dataset))])
    latlons = np.array([[dataset[i]['centroid_lat'].item(), dataset[i]['centroid_lon'].item()] for i in range(len(dataset))])
    return imgs, ids, latlons

def retrieve_ids_latlons(dataset):
    print("Retrieving ", dataset)

    print("Retrieving ids...")
    ids = np.array([dataset[i]['name'].replace('tile_', '').replace('.tif', '') for i in range(len(dataset))])

    print("Retrieving latlons...")
    latlons = np.array([[dataset[i]['centroid_lat'].item(), dataset[i]['centroid_lon'].item()] for i in range(len(dataset))])
    return ids, latlons

'''
    Append train, val, and test data
'''
def format_data(*args):
    print("Formatting data...")
    combined_imgs = np.empty((total_num_images, num_channels, img_height, img_width), dtype=np.float32)
    combined_ids = np.empty((total_num_images,), dtype='U{}'.format(15))
    combined_latlons = np.empty((total_num_images, 2), dtype=np.float32)

    data_idx = 0

    for arg in args:
        imgs, ids, latlons = retrieve_data(arg)

        for i in range(len(imgs)):
            combined_imgs[data_idx] = imgs[i]
            combined_ids[data_idx] = ids[i]
            combined_latlons[data_idx] = latlons[i]
            data_idx += 1

    print("Done adding the data to combined list")

    return combined_imgs, combined_ids, combined_latlons

def format_ids_latlons(*args):
    print("Formatting ids and latlons...")
    combined_ids = np.empty((total_num_images,), dtype='U{}'.format(15))
    print(combined_ids)
    combined_latlons = np.empty((total_num_images, 2), dtype=np.float32)
    print(combined_latlons)

    data_idx = 0

    for arg in args:
        ids, latlons = retrieve_ids_latlons(arg)

        for i in range(len(ids)):
            combined_ids[data_idx] = ids[i]
            combined_latlons[data_idx] = latlons[i]
            data_idx += 1

    print("Done adding the data to combined list")

    return combined_ids, combined_latlons

torchgeo_featurization(4096)