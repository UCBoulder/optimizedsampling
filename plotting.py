import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/TestSetPerformance.csv", index_col=0)
df_vopt = pd.read_csv("results/TestSetPerformanceVOptimality.csv", index_col=0)
df_satclip = pd.read_csv("results/TestSetPerformanceVOptimalitySatCLIP.csv", index_col=0)

def plot_results(*dataframes):
    for d in dataframes:
        d.drop('Cross-validation R2', axis=1, errors='ignore', inplace=True)

        # Split the index into 'label' and 'size_with_prefix' parts
        split = pd.DataFrame(d.index.str.split(';').tolist(), index=d.index, columns=['label', 'size_with_prefix'])
        d['label'] = split['label']
        d['size_with_prefix'] = split['size_with_prefix']

        # Handle 'sizeNone' by assigning it a placeholder value (e.g., float('inf') for whole dataset)
        d['size_of_subset'] = d['size_with_prefix'].apply(lambda x: float('inf') if x == 'sizeNone' else int(x.replace('size', '')))

        # Set 'label' and 'size_of_subset' as a MultiIndex and drop the old index columns
        d.set_index(['label', 'size_of_subset'], inplace=True)

        d.drop(columns='size_with_prefix', inplace=True)

    #Plot on the same plot:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    i=0
    labels = df.index.get_level_values('label').unique().tolist()
    for label in labels:
        j = 0
        colors = ["orangered", "steelblue", "green"]
        labels = ["srs", "image", 'satclip'] #FIX HARDCODING
        for d in [df, df_vopt, df_satclip]:
            # Filter rows for the specific label (e.g., "population")
            filtered = d.loc[label]

            # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
            filtered = filtered.sort_index()

            # Plot
            axs[i].plot(filtered.index, filtered["Test R2"], marker='o', linestyle='-', label=labels[j], color=colors[j])
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

    plt.subplots_adjust(wspace=0.4)   
    plt.suptitle('Number of samples vs R^2')
    plt.savefig("Num of samples vs R^2.png")

plot_results(df, df_vopt, df_satclip)