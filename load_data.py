# from USAVars import USAVars

import random

#Visualize
import rasterio
import matplotlib.pyplot as plt
import numpy as np

train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=True, checksum=False)
test = USAVars(root="/share/usavars", split="test", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
val = USAVars(root="/share/usavars", split="val", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

def create_sample_plot(dataset, s, t):
    fig, ax = plt.subplots(3,2, figsize=(10, 15)) #Subplot

    i = 0
    j = 0
    for num in random.sample(list(range(dataset.__len__())), 6):
        sample = dataset.__getitem__(num)
        sub_fig = dataset.plot(sample, suptitle=None) #function from USAVars.py
        ax_of_sub_fig = sub_fig.gca() 
        image_from_ax = ax_of_sub_fig.get_images()[0].get_array()

        if (j > 1):
            j = 0
            i = i + 1

        ax[i,j].imshow(image_from_ax)
        ax[i,j].set_title(ax_of_sub_fig.get_title(), fontsize=15)
        j = j+1

    fig.subplots_adjust(hspace = 0.8)

    name = s + " " + t + " " + "Sample Plot"
    fig.suptitle(name, fontsize = 20)
    fig.savefig(name)
    plt.close()

create_sample_plot(train, "USAVars", "Train")
create_sample_plot(test, "USAVars", "Test")
create_sample_plot(val, "USAVars", "Val")
