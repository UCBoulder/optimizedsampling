import rasterio
import matplotlib.pyplot as plt

# Replace 'your_file.tif' with your TIFF file path
file_path = 'test_image/tile_460,1372.tif'

with rasterio.open(file_path) as dataset:
    image_data = dataset.read(1)
    plt.imshow(image_data, cmap='gray')
    plt.savefig("test.png")