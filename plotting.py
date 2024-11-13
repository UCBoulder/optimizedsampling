import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/TestSetPerformance.csv", index_col=0)
df_vopt = pd.read_csv("results/TestSetPerformanceVOptimality.csv", index_col=0)
df_satclip = pd.read_csv("results/TestSetPerformanceSatCLIPVOptimality.csv", index_col=0)

df.drop('Cross-validation R2', axis=1, errors='ignore', inplace=True)
df_vopt.drop('Cross-validation R2', axis=1, errors='ignore', inplace=True)
df_satclip.drop('Cross-validation R2', axis=1, errors='ignore', inplace=True)

# Split the index into 'label' and 'size_with_prefix' parts
split_df = pd.DataFrame(df.index.str.split(';').tolist(), index=df.index, columns=['label', 'size_with_prefix'])
df['label'] = split_df['label']
df['size_with_prefix'] = split_df['size_with_prefix']

split_df = pd.DataFrame(df_vopt.index.str.split(';').tolist(), index=df_vopt.index, columns=['label', 'size_with_prefix'])
df_vopt['label'] = split_df['label']
df_vopt['size_with_prefix'] = split_df['size_with_prefix']

split_df = pd.DataFrame(df_satclip.index.str.split(';').tolist(), index=df_satclip.index, columns=['label', 'size_with_prefix'])
df_satclip['label'] = split_df['label']
df_satclip['size_with_prefix'] = split_df['size_with_prefix']

# Handle 'sizeNone' by assigning it a placeholder value (e.g., float('inf') for whole dataset)
df['size_of_subset'] = df['size_with_prefix'].apply(lambda x: float('inf') if x == 'sizeNone' else int(x.replace('size', '')))
df_vopt['size_of_subset'] = df_vopt['size_with_prefix'].apply(lambda x: float('inf') if x == 'sizeNone' else int(x.replace('size', '')))
df_satclip['size_of_subset'] = df_satclip['size_with_prefix'].apply(lambda x: float('inf') if x == 'sizeNone' else int(x.replace('size', '')))

# Set 'label' and 'size_of_subset' as a MultiIndex and drop the old index columns
df.set_index(['label', 'size_of_subset'], inplace=True)
df_vopt.set_index(['label', 'size_of_subset'], inplace=True)
df_satclip.set_index(['label', 'size_of_subset'], inplace=True)

df.drop(columns='size_with_prefix', inplace=True)
df_vopt.drop(columns='size_with_prefix', inplace=True)
df_satclip.drop(columns='size_with_prefix', inplace=True)

#Plot on the same plot:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

i=0
labels = df.index.get_level_values('label').unique().tolist()
for label in labels:
    # Filter rows for the specific label (e.g., "population")
    filtered_df = df.loc[label]
    filtered_df_vopt = df_vopt.loc[label]
    filtered_df_satclip = df_satclip.loc[label]

    # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
    filtered_df = filtered_df.sort_index()
    filtered_df_vopt = filtered_df_vopt.sort_index()
    filtered_df_satclip = filtered_df_satclip.sort_index()

    # Plot
    axs[i].plot(filtered_df.index, filtered_df["Test R2"], marker='o', linestyle='-', label="Spatial-only (SRS)", color='orangered')
    axs[i].plot(filtered_df_vopt.index, filtered_df_vopt["Test R2"], marker='o', linestyle='-', label="Image-only", color='steelblue')
    axs[i].plot(filtered_df_satclip.index, filtered_df_satclip["Test R2"], marker='o', linestyle='-', label="SatCLIP embeddings-based", color='green')

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
plt.savefig("Num of samples vs R^2.png")