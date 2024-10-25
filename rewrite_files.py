import rasterio
import matplotlib.pyplot as plt
from USAVars import USAVars

train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
test = USAVars(root="/share/usavars", split="test", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
val = USAVars(root="/share/usavars", split="val", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

def change_metadata(file_path, lat, lon):
    with rasterio.open(file_path, 'r+') as data:
        current_meta = data.meta
        updated_meta = current_meta.copy()
        updated_meta['centroid_lat'] = lat
        updated_meta['centroid_lon'] = lon
        data.update_tags(**updated_meta)

def rewrite(dataset):
    for num in range(0, dataset.__len__()):
        sample = dataset.__getitem__(num)
        name = sample['name']
        centroid_lat = sample['centroid_lat']
        centroid_lon = sample['centroid_lon']
        file_path = f'/share/usavars/uar/{name}'
        change_metadata(file_path, centroid_lat, centroid_lon)

print("hi")