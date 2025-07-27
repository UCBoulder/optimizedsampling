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
    print("Computing cosine distance matrix...")
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
    print("Using batched method...")
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
    print("done")

    return sim_matrix

def cosine_similarity_multi_gpu(X_train, X_test, gpu_ids=[0,1,2,3]):
    """
    Compute cosine similarity using multiple GPUs by splitting X_train rows.

    Args:
        X_train (np.ndarray): shape (N_train, D)
        X_test (np.ndarray): shape (N_test, D)
        gpu_ids (list of int): GPUs to use, e.g. [0,1]

    Returns:
        np.ndarray: similarity matrix shape (N_train, N_test)
    """
    print("Using multigpu method...")
    X_test_t = torch.tensor(X_test, dtype=torch.float32).cuda(gpu_ids[0])
    X_test_t = torch.nn.functional.normalize(X_test_t, p=2, dim=1)

    n = X_train.shape[0]
    splits = np.array_split(np.arange(n), len(gpu_ids))

    results = []
    for gpu, idxs in zip(gpu_ids, splits):
        X_chunk = torch.tensor(X_train[idxs], dtype=torch.float32).cuda(gpu)
        X_chunk = torch.nn.functional.normalize(X_chunk, p=2, dim=1)
        sim_chunk = torch.matmul(X_chunk, X_test_t.T).cpu().numpy()
        results.append((idxs, sim_chunk))

    #combine results in original order
    sim_matrix = np.empty((n, X_test.shape[0]), dtype=np.float32)
    for idxs, chunk in results:
        sim_matrix[idxs, :] = chunk
    print("done")

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

    for label in ['population', 'treecover']:
        data_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
        out_path = f"/home/libe2152/optimizedsampling/0_data/cosine_distance/usavars/{label}/cosine_distance.npy"

        print(f"Loading data from {data_path}...")
        X_train, X_test = load_data(data_path)

        print("Computing cosine distance matrix...")
        dist_mat = cosine_distance_matrix(X_train)

        save_matrix(dist_mat, matrix_type='distance', out_path=out_path)