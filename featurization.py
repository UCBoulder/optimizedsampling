from pathlib import Path

from mosaiks.code.mosaiks.featurization import featurize_and_save
from mosaiks.code.mosaiks import config as c

image_folder = "/share/usavars/uar"
out_fpath = Path(c.features_dir) / f"{image_folder.name}.pkl"

featurize_and_save(image_folder, out_fpath, c)