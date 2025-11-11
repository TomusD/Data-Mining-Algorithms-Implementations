# Data Mining & Similarity Algorithms

This repository contains Python implementations of various data mining and similarity search algorithms.

## Notebooks

### 1. `big_data_mining_techniques.ipynb`

This notebook implements text classification and an efficient similarity search using Locality Sensitive Hashing (LSH) for large text datasets.

**Technical Overview:**

* **Libraries:** `pandas`, `spacy`, `sklearn`, `datasketch`
* **Workflow:**
    1.  **Data Loading:** Reads `train.csv` and `test_without_labels.csv`.
    2.  **Text Preprocessing:** Cleans and prepares text data for vectorization.
    3.  **Text Classification:**
        * Trains and evaluates `LinearSVC` and `RandomForestClassifier` models using a `TfidfVectorizer`.
    4.  **Similarity Search (LSH):**
        * Implements **MinHash** (`get_minhash`) to create signatures for text documents.
        * Builds a **MinHashLSH** index (`datasketch.MinHashLSH`) for efficient querying.
    5.  **Baseline Comparison:**
        * Implements a brute-force similarity search using `sklearn.neighbors.NearestNeighbors` (with TF-IDF vectors) to serve as a ground truth.
    6.  **Evaluation:**
        * Queries the LSH index and the brute-force model.
        * Compares the query times and precision of the LSH approach against the brute-force method.

---

### 2. `dynamic_time_warping_implementation.ipynb`

This notebook provides a heavily optimized implementation of the Dynamic Time Warping (DTW) algorithm for measuring similarity between two time series.

**Technical Overview:**

* **Libraries:** `numpy`, `pandas`, `numba`
* **Workflow:**
    1.  **Data Parser:** Includes a custom `parser` function to convert string representations of lists (e.g., `"[1, 2.5, 3]"`) into clean `numpy` arrays of floats.
    2.  **DTW Implementation:**
        * The core logic is in the `dtw_distance` function.
    3.  **Optimizations:**
        * **Numba:** The `@jit(nopython=True)` decorator is used to Just-in-Time (JIT) compile the DTW function, dramatically speeding up the `numpy` operations.
        * **Sakoe-Chiba Band:** Implements a window constraint (`r`) to reduce the computational complexity, making it feasible for long series.
    4.  **Execution:**
        * The main script reads a CSV file (`dtw_test.csv`) containing pairs of time series.
        * It iterates through each pair, applying the optimized `dtw_distance` function to calculate their similarity.

## How to Use

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/Data-Mining-Algorithms-Implementations.git](https://github.com/your-username/Data-Mining-Algorithms-Implementations.git)
    cd Data-Mining-Algorithms-Implementations
    ```

2.  Install the required dependencies (preferably in a virtual environment):
    ```bash
    pip install pandas spacy sklearn datasketch numba jupyter
    ```

3.  Launch Jupyter and run the notebooks:
    ```bash
    jupyter notebook
    ```