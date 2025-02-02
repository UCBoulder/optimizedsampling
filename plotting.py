import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from format_csv import *

def plot_r2_num_samples(methods, *dfs):
    #Plot on the same plot:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = dfs[0].index.get_level_values('label').unique().tolist()
    for label in labels:
        j = 0
        colors = ["orangered", "steelblue", "green"]
        for df in dfs:
            df = df.reset_index()
            df = df.set_index(['label', 'size_of_subset'])
            # Filter rows for the specific label (e.g., "population")
            filtered_df = df.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            filtered_df = filtered_df.sort_index()

            # Plot
            axs[i].plot(filtered_df.index, filtered_df["Test R2"], marker='o', linestyle='-', label=methods[j], color=colors[j])
            j += 1

        # Customize the plot
        axs[i].set_xlabel("Size of Subset")
        axs[i].set_ylabel("$R^2$")
        axs[i].set_ylim(0,1)
        if label=="population":
            axs[i].set_title("Population")
        if label=="elevation":
            axs[i].set_title("Elevation")
        if label=="treecover":
            axs[i].set_title("Treecover")
        axs[i].legend()
        i = i+1

    fig.subplots_adjust(wspace=0.4)   
    fig.suptitle('$R^2$ of ridge regression trained on subsets of fixed size')
    return fig

def plot_r2_cost(methods, dfs, title):
    #Plot on the same plot:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = dfs[0].index.get_level_values('label').unique().tolist()
    for label in ["population", "treecover", "elevation"]:
        j = 0
        colors = ["#FF6347", "#4682B4", "#000000", "#008000", "#FFD700"]
        for df in dfs:
            df = df.reset_index()
            df = df.set_index(['label'])
            # Filter rows for the specific label (e.g., "population")
            filtered_df = df.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            #filtered_df = filtered_df.sort_index()
            filtered_df = filtered_df.sort_values(by="Cost")

            #If R2 is averaged, plot error bars
            if filtered_df.get('Test Avg R2.') is not None:
                if filtered_df.get('Test Std R2') is not None:
                    axs[i].errorbar(filtered_df["Budget"], 
                                    filtered_df["Test Avg R2"], 
                                    yerr=filtered_df['Test Std R2'], 
                                    fmt='o', 
                                    linestyle='', 
                                    label=methods[j], 
                                    color=colors[j], 
                                    markersize=5,
                                    capsize=5, 
                                    alpha=0.6)
            else:
                axs[i].plot(filtered_df["Cost"], 
                            filtered_df["Test R2"], 
                            marker='.', 
                            linestyle='', 
                            label=methods[j], 
                            color=colors[j], 
                            markersize=4, 
                            alpha=0.5)
            j += 1

        # Customize the plot
        axs[i].set_xlabel("Cost")
        axs[i].set_ylabel("$R^2$")
        axs[i].set_ylim(0,1)
        if label=="population":
            axs[i].set_title("Population")
        if label=="elevation":
            axs[i].set_title("Elevation")
        if label=="treecover":
            axs[i].set_title("Treecover")
        axs[i].legend()
        i = i+1

    fig.subplots_adjust(wspace=0.4)   
    fig.suptitle(title)
    return fig

def plot_r2_num_samples_with_cost(methods, *dfs):
    #Plot on the same plot:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = dfs[0].index.get_level_values('label').unique().tolist()
    for label in labels:
        j = 0
        for df in dfs:
            df = df.reset_index()
            df = df.set_index(['label', 'size_of_subset'])
            # Filter rows for the specific label (e.g., "population")
            filtered_df = df.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            filtered_df = filtered_df.sort_index()
            x = filtered_df.index.to_numpy()
            y = filtered_df["Test R2"].to_numpy()
            c = filtered_df["Cost"] #parameter of color bar

            # Create segments of the line
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a LineCollection
            norm = plt.Normalize(c.min(), c.max())  # Normalize the color values
            lc = LineCollection(segments, cmap='Spectral', norm=norm, label=methods[j])
            axs[i].text(x[-3], y[-3] + 0.03, methods[j], fontsize=8, color='black', verticalalignment='bottom')
            j += 1
            lc.set_array(c)  # Assign the color values to the LineCollection
            lc.set_linewidth(2)  # Set line width

            # Plot
            axs[i].add_collection(lc)
            axs[i].autoscale()  # Adjust axis limits to the data
            axs[i].set_xlim(x.min(), x.max())
            axs[i].set_ylim(y.min(), y.max())
            #axs[i].text(segments[-1][0], segments[-1][1], methods[j], fontsize=10, ha='center', va='center')
            

        # Add a color bar
        cb = plt.colorbar(lc, ax=axs[i], extend='neither')
        cb.set_label('Cost')

        # Customize the plot
        axs[i].set_xlabel("Size of Subset")
        axs[i].set_ylabel("$R^2$")
        axs[i].set_ylim(0,1)
        if label=="population":
            axs[i].set_title("Population")
        if label=="elevation":
            axs[i].set_title("Elevation")
        if label=="treecover":
            axs[i].set_title("Treecover")
        #axs[i].legend()
        i = i+1

    fig.subplots_adjust(wspace=0.4)   
    fig.suptitle('Number of samples vs $R^2$ with cost')
    return fig

if __name__ == '__main__':
    dfs = []
    df_random = pd.read_csv(f"results/Torchgeo4096_random_cost_cluster_NLCD_percentages_1347_formatted.csv", index_col=0)
    dfs.append(df_random)

    df_clusters = pd.read_csv(f"results/Torchgeo4096_clusters_cost_cluster_NLCD_percentages_1347_formatted.csv", index_col=0)
    dfs.append(df_clusters)

    df_inv = pd.read_csv(f"results/Torchgeo4096_invsize_cost_cluster_NLCD_percentages_1347_lambda_0.5_formatted.csv", index_col=0)
    dfs.append(df_inv)

    #df_propstrat = pd.read_csv(f"results/Torchgeo4096_invsize_cost_cluster_NLCD_percentages_1347_lambda_1_formatted.csv", index_col=0)
    #dfs.append(df_propstrat)

    df_lowcost = pd.read_csv(f"results/Torchgeo4096_greedycost_cost_cluster_NLCD_percentages_1347_formatted.csv", index_col=0)
    dfs.append(df_lowcost)

    fig = plot_r2_cost(["srs", "clusters", "invsize", "lowcost"], dfs, title=f'$R^2$ vs Cost of Collection for NLCD percentages clusters')
    fig.savefig(f"test.png")