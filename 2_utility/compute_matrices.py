import torch
import numpy as np

def try_cosine_similarity_matrix(X_train, X_test):
    """
    Try computing cosine similarity on GPU in one shot.
    If OOM occurs, fallback to block-wise computation.
    """
    try:
        return cosine_similarity_matrix(X_train, X_test)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM — falling back to batched similarity computation...")
            torch.cuda.empty_cache()
            return batched_cosine_similarity(X_train, X_test)
        else:
            raise

def cosine_similarity_matrix(X_train, X_test):
    """
    Compute cosine similarity between X_train and X_test using GPU.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32, device='cuda')
    X_test = torch.tensor(X_test, dtype=torch.float32, device='cuda')

    X_train = torch.nn.functional.normalize(X_train, p=2, dim=1)
    X_test = torch.nn.functional.normalize(X_test, p=2, dim=1)

    sim = torch.matmul(X_train, X_test.T)
    return sim.cpu().numpy()

def cosine_distance_matrix(X_train):
    """
    Compute cosine distance matrix on GPU between X_train and itself.
    """
    sim = try_cosine_similarity_matrix(X_train, X_train)
    return 1.0 - sim

def batched_cosine_similarity(X_train, X_test, batch_size=1024):
    """
    Block-wise cosine similarity computation for large matrices.

    Args:
        X_train (np.ndarray)
        X_test (np.ndarray)
        batch_size (int)

    Returns:
        np.ndarray: (n_train, n_test) cosine similarity matrix
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    X_train = torch.nn.functional.normalize(X_train, p=2, dim=1)
    X_test = torch.nn.functional.normalize(X_test, p=2, dim=1)

    sim_matrix = np.empty((X_train.shape[0], X_test.shape[0]), dtype=np.float32)

    for i in range(0, X_train.shape[0], batch_size):
        end_i = min(i + batch_size, X_train.shape[0])
        batch_i = X_train[i:end_i].cuda()
        for j in range(0, X_test.shape[0], batch_size):
            end_j = min(j + batch_size, X_test.shape[0])
            batch_j = X_test[j:end_j].cuda()
            with torch.no_grad():
                block = torch.matmul(batch_i, batch_j.T).cpu().numpy()
            sim_matrix[i:end_i, j:end_j] = block
            del batch_j
            torch.cuda.empty_cache()
        del batch_i
        torch.cuda.empty_cache()

    return sim_matrix

def unitwise_avg_cosine(X_train, unit_train, X_test, unit_test):
    """
    Computes average cosine similarity between units in a fast, vectorized way.

    Args:
        X_train: (n_train, d) array
        unit_train: (n_train,) array of unit labels
        X_test: (n_test, d) array
        unit_test: (n_test,) array of unit labels

    Returns:
        (n_units_train, n_units_test) unit-unit similarity matrix
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_train = torch.nn.functional.normalize(X_train, dim=1)
    X_test = torch.nn.functional.normalize(X_test, dim=1)

    unit_train = np.asarray(unit_train)
    unit_test = np.asarray(unit_test)

    unique_train, train_inv = np.unique(unit_train, return_inverse=True)
    unique_test, test_inv = np.unique(unit_test, return_inverse=True)

    n_train_units = len(unique_train)
    n_test_units = len(unique_test)
    d = X_train.shape[1]

    #per unit sum of normalized embeddings
    sum_train = torch.zeros((n_train_units, d), dtype=torch.float32)
    sum_test = torch.zeros((n_test_units, d), dtype=torch.float32)
    count_train = torch.zeros(n_train_units, dtype=torch.float32)
    count_test = torch.zeros(n_test_units, dtype=torch.float32)

    for i in range(len(X_train)):
        sum_train[train_inv[i]] += X_train[i]
        count_train[train_inv[i]] += 1

    for i in range(len(X_test)):
        sum_test[test_inv[i]] += X_test[i]
        count_test[test_inv[i]] += 1

    #each row in sum_{train,test} is summed vecor for unit
    #compute dot product between each row combo
    sim_matrix = sum_train @ sum_test.T  # (n_train_units, n_test_units)

    #normalize by outer product of sample counts
    norm_matrix = count_train[:, None] * count_test[None, :]
    sim_matrix = sim_matrix / norm_matrix

    return sim_matrix.numpy(), unique_train, unique_test


def save_matrix(X, matrix_type='similarity', out_path='matrix.npy'):
    """
    Save a matrix (cosine similarity or distance) to disk.
    """
    if matrix_type not in ['similarity', 'distance']:
        raise ValueError("matrix_type must be 'similarity' or 'distance'")
    np.save(out_path, X)
    print(f"{matrix_type.capitalize()} matrix saved to {out_path}")

if __name__ == "__main__":
    import pickle

    def load_data(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['ids_train']
    
    data_path = f"/home/libe2152/optimizedsampling/0_data/features/togo/togo_fertility_data_all_2021_Jan_Jun_P20.pkl"
    unit_path = f"/home/libe2152/optimizedsampling/0_data/groups/togo/canton_assignments_dict.pkl"
    out_path_unit = f"/home/libe2152/optimizedsampling/0_data/cosine_similarity/togo/canton_cosine_similarity_train_train.npy"
    out_path_all = f"/home/libe2152/optimizedsampling/0_data/cosine_similarity/togo/cosine_similarity_train_train.npy"

    print(f"Loading data from {data_path}...")
    X_train, ids_train = load_data(data_path)

    with open(unit_path, "rb") as f:
        idx_to_assignment = pickle.load(f)

    assignments_ordered = [idx_to_assignment[idx] for idx in ids_train]

    sim_matrix, _, _ = unitwise_avg_cosine(X_train, assignments_ordered, X_train, assignments_ordered)
    save_matrix(sim_matrix, out_path=out_path_unit)

    sim_matrix_all = batched_cosine_similarity(X_train, X_train)
    save_matrix(sim_matrix_all, out_path = out_path_all)