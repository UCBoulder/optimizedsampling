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
        return data['X_train'], data['X_test']

    data_path = f"/home/libe2152/optimizedsampling/0_data/features/togo/togo_fertility_data_all_2021_Jan_Jun_P20.pkl"
    out_path = f"/home/libe2152/optimizedsampling/0_data/cosine_similarity/togo/cosine_similarity_train_test.npy"

    print(f"Loading data from {data_path}...")
    X_train, X_test = load_data(data_path)

    print("Computing cosine similarity matrix...")
    sim_matrix = batched_cosine_similarity(X_train, X_test)

    save_matrix(sim_matrix, matrix_type='similarity', out_path=out_path)