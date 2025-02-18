import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
import seaborn as sns

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

def plot_r2_cost(methods, dfs, title=None):
    #Plot on the same plot:
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = ['population', 'treecover']
    colors = ['#909090', "#a559aa", "#2B6A4B", "#A32020", "#154683"]
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    if len(dfs) == 2:
        colors = ['#909090', "#154683"]
        linestyles = ['-', (0, (3, 1, 1, 1))]

    if len(dfs) == 3:
        colors = ['#909090', "#a559aa", "#154683"]
        linestyles = ['-', '--', (0, (3, 1, 1, 1))]

    for label in labels:
        j = 0

        for df in dfs:
            df = df.reset_index()
            df = df.set_index(['label'])
            # Filter rows for the specific label (e.g., "population")
            filtered_df = df.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            #filtered_df = filtered_df.sort_index()
            filtered_df = filtered_df.sort_values(by="Cost")

            #If R2 is averaged, plot error bars
            if filtered_df.get('Test Avg R2') is not None:
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
                            marker='o', 
                            linestyle='', 
                            #label=methods[j], 
                            color=colors[j], 
                            markersize=4, 
                            alpha=0.1)

            mask = np.isfinite(filtered_df["Test R2"]) & np.isfinite(filtered_df["Cost"])
            filtered_df = filtered_df[mask]
            frac = 0.3

            if label == 'population':
                if j == 2:
                    frac = 0.4
            if label == 'treecover':
                if j == 2:
                    frac = 0.5
            # Perform LOWESS smoothing
            smoothed = sm.nonparametric.lowess(
                endog=filtered_df["Test R2"],  # Dependent variable (Y)
                exog=filtered_df["Cost"],  # Independent variable (X)
                frac=frac,  # Smoothing parameter (adjust between 0.1 and 0.5)
                it=3  # Number of robustifying iterations
            )

            # Extract smoothed values
            x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]

            # Plot LOWESS trendline
            axs[i].plot(x_smooth, y_smooth, 
                        linestyle=linestyles[j], 
                        color=colors[j], 
                        alpha=0.8, 
                        label=f"{methods[j]}")

            j += 1

        # Customize the plot
        axs[i].set_xlabel("Cost")
        axs[i].set_ylabel("$R^2$")
        if label=="population":
            axs[i].set_ylim(bottom=0.25)
            axs[i].set_title("Population")
        if label=="elevation":
            axs[i].set_ylim(bottom=0.0)
            axs[i].set_title("Elevation")
        if label=="treecover":
            axs[i].set_ylim(bottom=0.65)
            axs[i].set_title("Treecover")
        axs[i].legend()
        i = i+1

    #fig.subplots_adjust(wspace=0.4)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
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
    # #Plot 10
    # dfs = []
    # df_random = pd.read_csv(f"results/NLCD_percentages_8_four_groups_10/final_random_cost_cluster_NLCD_percentages.csv", index_col=0)
    # dfs.append(df_random)

    # df_clusters = pd.read_csv(f"results/NLCD_percentages_8_four_groups_10/final_clusters_cost_cluster_NLCD_percentages.csv", index_col=0)
    # dfs.append(df_clusters)

    # df_l0 = pd.read_csv(f"results/NLCD_percentages_8_four_groups_10/final_greedycost_cost_cluster_NLCD_percentages.csv", index_col=0)
    # dfs.append(df_l0)

    # df_l005 = pd.read_csv(f"results/NLCD_percentages_8_four_groups_10/final_invsize_cost_cluster_NLCD_percentages_lambda_0.05.csv", index_col=0)
    # dfs.append(df_l005)

    # df_l1 = pd.read_csv(f"results/NLCD_percentages_8_four_groups_10/final_invsize_cost_cluster_NLCD_percentages_lambda_1.0.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", "StRS", "OPT($\lambda=0$)", "OPT($\lambda=0.05$)", "OPT($\lambda=1$)"], dfs)
    # fig.savefig("Plot10.png", dpi=300, bbox_inches='tight')

    # #Plot 100
    # dfs = []
    # df_random = pd.read_csv(f"results/NLCD_percentages_8_four_groups_100/100_final_random_cost_cluster_NLCD_percentages_formatted.csv", index_col=0)
    # dfs.append(df_random)

    # df_clusters = pd.read_csv(f"results/NLCD_percentages_8_four_groups_100/100_final_clusters_cost_cluster_NLCD_percentages_formatted.csv", index_col=0)
    # dfs.append(df_clusters)

    # df_l0 = pd.read_csv(f"results/NLCD_percentages_8_four_groups_100/100_final_greedycost_cost_cluster_NLCD_percentages_formatted.csv", index_col=0)
    # dfs.append(df_l0)

    # df_l05 = pd.read_csv(f"results/NLCD_percentages_8_four_groups_100/100_final_invsize_cost_cluster_NLCD_percentages_lambda_0.5_formatted.csv", index_col=0)
    # dfs.append(df_l05)

    # df_l1 = pd.read_csv(f"results/NLCD_percentages_8_four_groups_100/100_final_invsize_cost_cluster_NLCD_percentages_lambda_1.0_formatted.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", "StRS", "OPT($\lambda=0$)", "OPT($\lambda=0.5$)", "OPT($\lambda=1$)"], dfs)
    # fig.savefig("Plot100.png", dpi=300, bbox_inches='tight')

    # #Plot 50
    # dfs = []
    # df_random = pd.read_csv(f"results/NLCD_percentages_8_two_groups_50/50_final_random_cost_cluster_NLCD_percentages.csv", index_col=0)
    # dfs.append(df_random)

    # df_clusters = pd.read_csv(f"results/NLCD_percentages_8_two_groups_50/50_final_clusters_cost_cluster_NLCD_percentages.csv", index_col=0)
    # dfs.append(df_clusters)

    # df_l0 = pd.read_csv(f"results/NLCD_percentages_8_two_groups_50/50_final_greedycost_cost_cluster_NLCD_percentages.csv", index_col=0)
    # dfs.append(df_l0)

    # df_l005 = pd.read_csv(f"results/NLCD_percentages_8_two_groups_50/final_invsize_NLCD_percentages_cost_cluster_NLCD_percentages_two_groups_50_lambda_0.05.csv", index_col=0)
    # dfs.append(df_l005)

    # df_l1 = pd.read_csv(f"results/NLCD_percentages_8_two_groups_50/50_final_invsize_cost_cluster_NLCD_percentages_lambda_1.0.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", "StRS", "OPT($\lambda=0$)", "OPT($\lambda=0.05$)", "OPT($\lambda=1$)"], dfs)
    # fig.savefig("Plot50.png", dpi=300, bbox_inches='tight')

    #Plot 3 (East)
    dfs = []

    df_random = pd.read_csv(f"results/train_East_test_US_NLCD_percentages_8_four_groups_10/final_random_State_East_formatted.csv", index_col=0)
    dfs.append(df_random)

    df_clusters = pd.read_csv("results/train_East_test_US_NLCD_percentages_8_four_groups_10/final_clusters_State_East_formatted.csv", index_col=0)
    dfs.append(df_clusters)

    df_l1 = pd.read_csv(f"results/train_East_test_US_NLCD_percentages_8_four_groups_10/final_invsize_State_East_lambda_1.0_formatted.csv", index_col=0)
    dfs.append(df_l1)

    fig = plot_r2_cost(["SRS", " StRS", "OPT($\lambda=1$)"], dfs)
    fig.savefig("Plot_East.png", dpi=300, bbox_inches='tight')

    #Plot 3 (West)
    dfs = []

    df_random = pd.read_csv(f"results/train_West_test_US_NLCD_percentages_8_four_groups_10/final_random_State_West_formatted.csv", index_col=0)
    dfs.append(df_random)

    df_clusters = pd.read_csv("results/train_West_test_US_NLCD_percentages_8_four_groups_10/final_clusters_State_West_formatted.csv", index_col=0)
    dfs.append(df_clusters)

    df_l1 = pd.read_csv(f"results/train_West_test_US_NLCD_percentages_8_four_groups_10/final_invsize_State_West_lambda_1.0_formatted.csv", index_col=0)
    dfs.append(df_l1)

    fig = plot_r2_cost(["SRS", " StRS", "OPT($\lambda=1$)"], dfs)
    fig.savefig("Plot_West.png", dpi=300, bbox_inches='tight')

    #Plot 4 (North)
    # dfs = []

    # df_random = pd.read_csv(f"results/train_North_test_US_NLCD_percentages_8_four_groups_10/final_random_State_North_1.csv", index_col=0)
    # dfs.append(df_random)

    # df_clusters = pd.read_csv("results/train_North_test_US_NLCD_percentages_8_four_groups_10/final_clusters_State_North_1.csv", index_col=0)
    # dfs.append(df_clusters)

    # df_l1 = pd.read_csv(f"results/train_North_test_US_NLCD_percentages_8_four_groups_10/final_invsize_State_North_1_lambda_1.0.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", " StRS", "OPT($\lambda=1$)"], dfs, title='Train on North, Test on US')
    # fig.savefig("Plot_North.png", dpi=300, bbox_inches='tight')

    # #Plot 4 (South)
    # dfs = []

    # df_random = pd.read_csv(f"results/train_South_test_US_NLCD_percentages_8_four_groups_10/final_random_State_South_1.csv", index_col=0)
    # dfs.append(df_random)

    # df_clusters = pd.read_csv("results/train_South_test_US_NLCD_percentages_8_four_groups_10/final_clusters_State_South_1.csv", index_col=0)
    # dfs.append(df_clusters)

    # df_l1 = pd.read_csv(f"results/train_South_test_US_NLCD_percentages_8_four_groups_10/final_invsize_State_South_1_lambda_1.0.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", " StRS", "OPT($\lambda=1$)"], dfs, title='Train on South, Test on US')
    # fig.savefig("Plot_South.png", dpi=300, bbox_inches='tight')

    #Plot 5 (train on West, test on East; NLCD_percentages 4)
    # dfs = []

    # df_random = pd.read_csv(f"results/train_West_test_East/final_random_State_West_1_test_East.csv", index_col=0)
    # dfs.append(df_random)

    # # df_clusters = pd.read_csv("results/train_South_test_US_NLCD_percentages_8_four_groups_10/final_clusters_State_South_1.csv", index_col=0)
    # # dfs.append(df_clusters)

    # df_l0 = pd.read_csv("results/train_West_test_East/final_greedycost_State_West_1_test_East.csv", index_col=0)
    # dfs.append(df_l0)

    # df_l05 = pd.read_csv("results/train_West_test_East/final_invsize_NLCD_percentages_4_State_West_1_lambda_0.5_test_East.csv", index_col=0)
    # dfs.append(df_l05)

    # df_l1 = pd.read_csv(f"results/train_West_test_East/final_invsize_NLCD_percentages_4_State_West_1_lambda_1.0_test_East.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", "OPT($\lambda=0$)", "OPT($\lambda=0.5$)", "OPT($\lambda=1$)"], dfs, title='Train on West, Test on East, with NLCD_percentages 4')
    # fig.savefig("Plot_West_test_East_NLCD_percentages_4.png", dpi=300, bbox_inches='tight')

    #Plot 6 (train on North, test on Sotuh; NLCD_percentages)
    # dfs = []

    # df_random = pd.read_csv(f"results/train_North_test_South/final_random_State_North_1_test_South.csv", index_col=0)
    # dfs.append(df_random)

    # # df_clusters = pd.read_csv("results/train_South_test_US_NLCD_percentages_8_four_groups_10/final_clusters_State_South_1.csv", index_col=0)
    # # dfs.append(df_clusters)

    # # df_l0 = pd.read_csv("results/train_South_test_North/final_greedycost_State_South_1_test_North.csv", index_col=0)
    # # dfs.append(df_l0)

    # df_l1 = pd.read_csv(f"results/train_North_test_South/final_invsize_NLCD_percentages_State_North_1_lambda_1.0_test_South.csv", index_col=0)
    # dfs.append(df_l1)

    # fig = plot_r2_cost(["SRS", "OPT($\lambda=1$)"], dfs, title='Train on North, Test on South')
    # fig.savefig("Plot_North_test_South.png", dpi=300, bbox_inches='tight')