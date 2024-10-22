import pickle

# Path to your .pkl file
file_path = 'CONTUS_UAR.pkl'

# Load the feature vectors from the .pkl file
with open(file_path, 'rb') as file:
    feature_vectors = pickle.load(file)

# Now you can use feature_vectors as needed
print(feature_vectors)
