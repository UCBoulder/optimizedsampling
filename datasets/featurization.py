import argparse
import importlib
from pathlib import Path

import torch
import dill
import numpy as np

from torchgeo.models import RCF, swin_v2_t, swin_v2_b
from USAVars import USAVars

DATASET_CLASSES = {
    "USAVars": USAVars,
}


def load_dataset(dataset_name, data_root, split, labels):
    if dataset_name not in DATASET_CLASSES:
        raise ValueError(f"Dataset '{dataset_name}' is not in the allowed list: {list(DATASET_CLASSES.keys())}")

    dataset_class = DATASET_CLASSES[dataset_name]

    return dataset_class(
        root=data_root,
        split=split,
        labels=labels,
        transforms=None,
        download=False,
        checksum=False,
    )

def transformer_featurization(train, val, test, total_num_images):
    num_features=1024
    out_fpath = f"data/int/feature_matrices/CONTUS_UAR_swin_v2_b.pkl"

    # ids, latlons extraction
    ids, latlons = format_ids_latlons(total_num_images, train, val, test)

    # Load SwinV2-B pretrained on NAIP MI-SATLAS
    model = swin_v2_b(weights="NAIP_RGB_MI_SATLAS")
    model.eval()
    model.head = torch.nn.Identity()  # remove classification head

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Featurizing images...")

    featurized_imgs = np.empty((total_num_images, num_features), dtype=np.float32)

    for i in range(total_num_images):
        print(f"Featurizing image {i}/{total_num_images}")

        if i < len(train):
            img = train[i]['image']
        elif i < len(train) + len(val):
            img = val[i - len(train)]['image']
        else:
            img = test[i - len(train) - len(val)]['image']

        # Ensure batch dimension and move to device
        img = img.unsqueeze(0).to(device)  # shape (1, C, H, W)

        with torch.no_grad():
            feats = model(img)  # shape (1, D)
            featurized_imgs[i] = feats.squeeze(0).cpu().numpy()

    # Save features
    with open(out_fpath, "wb") as f:
        dill.dump(
            {"X": featurized_imgs, "ids_X": ids, "latlon": latlons},
            f,
            protocol=4,
        )


def torchgeo_featurization(train, val, test, num_features, total_num_images):
    out_fpath = f"data/int/feature_matrices/CONTUS_UAR_torchgeo{num_features}.pkl"

    #imgs, ids, latlons = format_data(train, val, test)
    ids, latlons = format_ids_latlons(total_num_images, train, val, test)

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

    featurized_imgs = np.empty((total_num_images, num_features), dtype=np.float32)

    for i in range(total_num_images):
        print("Featurizing image ", i)
        if i < len(train):
            img = train[i]['image'].to(device)
        elif i < len(train) + len(val):
            img = val[i - len(train)]['image'].to(device)
        else:
            img = test[i - len(train) - len(val)]['image'].to(device)

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
def format_data(total_num_images, num_channels, img_height, img_width, *args):
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


def format_ids_latlons(total_num_images, *args):
    print("Formatting ids and latlons...")
    combined_ids = np.empty((total_num_images,), dtype='U{}'.format(15))
    combined_latlons = np.empty((total_num_images, 2), dtype=np.float32)

    data_idx = 0

    for arg in args:
        ids, latlons = retrieve_ids_latlons(arg)

        for i in range(len(ids)):
            combined_ids[data_idx] = ids[i]
            combined_latlons[data_idx] = latlons[i]
            data_idx += 1

    print("Done adding the data to combined list")

    return combined_ids, combined_latlons

def main(args):
    data_root = Path(args.data_root).expanduser()
    labels = tuple(args.labels.split(","))

    print(f"Using dataset: {args.dataset_name}")
    train = load_dataset(args.dataset_name, data_root, "train", labels)
    val = load_dataset(args.dataset_name, data_root, "val", labels)
    test = load_dataset(args.dataset_name, data_root, "test", labels)

    total_num_images = len(train) + len(val) + len(test)
    print(f"Loaded {total_num_images} total images.")

    print(f"Image dimensions: {args.image_size}x{args.image_size}, Channels: {args.num_channels}")

    if args.feat_type == 'RCF':
        torchgeo_featurization(train, val, test, args.num_features, total_num_images)
    else:
        transformer_featurization(train, val, test, total_num_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run featurization on USAVars dataset.")

    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=DATASET_CLASSES.keys(),
                        help=f"Name of dataset. Allowed: {list(DATASET_CLASSES.keys())}")
    parser.add_argument("--data_root", type=str, default="/share/usavars",
                        help="Root directory for dataset.")
    parser.add_argument("--labels", type=str, default="treecover,elevation,population",
                        help="Comma-separated labels to include.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Height and width of input images.")
    parser.add_argument("--num_channels", type=int, default=4,
                        help="Number of channels per image.")
    parser.add_argument("--num_features", type=int, default=4096,
                        help="Number of features.")
    parser.add_argument("--feat_type", type=str, default="RCF",
                        help="type of features")

    args = parser.parse_args()
    main(args)