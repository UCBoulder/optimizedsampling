from pathlib import Path
from mosaiks.code.mosaiks import config as c
from mosaiks.code.mosaiks.featurization import featurize_and_save

image_folder = Path("/share/usavars/uar")
#image_folder = Path("test_image")
out_fpath = Path(c.features_dir) / f"{image_folder.name}.pkl"

featurize_and_save(image_folder, out_fpath, c)