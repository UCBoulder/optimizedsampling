import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/TestSetPerformance.csv", index_col=0)
df.drop('Cross-validation R2', axis=1, errors='ignore', inplace=True)

# Split the index into 'label' and 'size_with_prefix' parts
df['label'], df['size_with_prefix'] = df.index.str.split(';').str

# Handle 'sizeNone' by assigning it a placeholder value (e.g., float('inf') for whole dataset)
df['size_of_subset'] = df['size_with_prefix'].apply(lambda x: float('inf') if x == 'sizeNone' else int(x.replace('size', '')))

# Set 'label' and 'size_of_subset' as a MultiIndex and drop the old index columns
df.set_index(['label', 'size_of_subset'], inplace=True)
df.drop(columns='size_with_prefix', inplace=True)

labels = df.index.get_level_values('label').unique().tolist()
for label in labels:
    # Filter rows for the specific label (e.g., "population")
    filtered_df = df.loc[label]

    # Sort the filtered DataFrame by 'size_of_subset' for accurate plotting
    filtered_df = filtered_df.sort_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df.index, filtered_df["Test R2"], marker='o', linestyle='-', label=f"Label: {label}")

    # Customize the plot
    plt.xlabel("Size of Subset")
    plt.ylabel("R2 Score")
    plt.title(f"R2 Score vs Size of Subset for Label '{label}'")
    plt.legend()
    plt.grid(True)
    plt.savefig(label+" Number of Samples vs Test R2.png")