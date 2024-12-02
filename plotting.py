import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np

df_srs = pd.read_csv("results/TestSetPerformanceRandom.csv", index_col=0)
# df_img = pd.read_csv("results/TestSetPerformanceImage.csv", index_col=0)
# df_satclip = pd.read_csv("results/TestSetPerformanceSatCLIP.csv", index_col=0)

def format_dataframe(df):
    # Split the index into 'label' and 'size_with_prefix' parts
    split = pd.DataFrame(df.index.str.split(';').tolist(), index=df.index, columns=['label', 'size_with_prefix'])
    df['label'] = split['label']
    df['size_with_prefix'] = split['size_with_prefix']

    # Handle 'sizeNone' by assigning it a placeholder value (e.g., float('inf') for whole dataset)
    df['size_of_subset'] = df.apply(
        lambda row: float(54340) if (row['size_with_prefix'] == 'sizeNone' and row['label'] == 'population') 
        else (float(97875) if (row['size_with_prefix'] == 'sizeNone' and (row['label'] in ['elevation', 'treecover'] ))
            else int(row['size_with_prefix'].replace('size', ''))),
        axis=1
    )

    # Set 'label' and 'size_of_subset' as a MultiIndex and drop the old index columns
    df.set_index(['label', 'size_of_subset'], inplace=True)

    df.drop(columns='size_with_prefix', inplace=True)

def plot_results(*dfs):
    for df in dfs:
        format_dataframe(df)

    #Plot on the same plot:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = dfs[0].index.get_level_values('label').unique().tolist()
    for label in labels:
        j = 0
        colors = ["orangered", "steelblue", "green"]
        labels = ["srs", "image", 'satclip'] #FIX HARDCODING
        for df in dfs:
            # Filter rows for the specific label (e.g., "population")
            filtered_df = df.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            filtered_df = filtered_df.sort_index()

            # Plot
            axs[i].plot(filtered_df.index, filtered_df["Test R2"], marker='o', linestyle='-', label=labels[j], color=colors[j])
            j += 1

        # Customize the plot
        axs[i].set_xlabel("Size of Subset")
        axs[i].set_ylabel("$R^2$ Score")
        if label=="population":
            axs[i].set_title("Population")
        if label=="elevation":
            axs[i].set_title("Elevation")
        if label=="treecover":
            axs[i].set_title("Treecover")
        axs[i].legend()
        i = i+1

    fig.subplots_adjust(wspace=0.4)   
    fig.suptitle('Number of samples vs R^2')
    fig.savefig("Num of samples vs R^2.png")

def plot_results_with_cost(*dfs):
    for df in dfs:
        format_dataframe(df)

    #Plot on the same plot:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = dfs[0].index.get_level_values('label').unique().tolist()
    for label in labels:
        j = 0
        labels = ["srs", "image", 'satclip'] #FIX HARDCODING
        for df in dfs:
            # Filter rows for the specific label (e.g., "population")
            filtered_df = df.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            filtered_df = filtered_df.sort_index()
            x = filtered_df.index
            y = filtered_df["Test R2"]
            c = filtered_df["Cost"] #parameter of color bar

            # Create segments of the line
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a LineCollection
            norm = plt.Normalize(c.min(), c.max())  # Normalize the color values
            lc = LineCollection(segments, cmap='viridis', norm=norm, label=labels[j])
            j += 1
            lc.set_array(c)  # Assign the color values to the LineCollection
            lc.set_linewidth(2)  # Set line width

            # Plot
            axs[i].add_collection(lc)
            axs[i].autoscale()  # Adjust axis limits to the data
            axs[i].set_xlim(x.min(), x.max())
            axs[i].set_ylim(y.min(), y.max())

            # Add a color bar
            cb = plt.colorbar(lc, ax=axs[i], extend='neither')
            cb.set_label('Cost')

        # Customize the plot
        axs[i].set_xlabel("Size of Subset")
        axs[i].set_ylabel("$R^2$ Score")
        if label=="population":
            axs[i].set_title("Population")
        if label=="elevation":
            axs[i].set_title("Elevation")
        if label=="treecover":
            axs[i].set_title("Treecover")
        axs[i].legend()
        i = i+1

    fig.subplots_adjust(wspace=0.4)   
    fig.suptitle('Number of samples vs R^2')
    fig.savefig("Num of samples vs R^2.png")

plot_results_with_cost(df_srs)