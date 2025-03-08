# GNN Music Recommendation System    

This project uses a Graph Neural Network (GNN) to build a recommendation system for music playlists and tracks based on audio features and playlist–track interactions. A bipartite graph is constructed where one set of nodes represents playlists and the other represents tracks. Track audio features serve as node attributes, while playlists are initialized with random embeddings. A simple two-layer GCN (using `GCNConv` from PyTorch Geometric) is trained to learn meaningful node embeddings that are then used to recommend similar songs.

## Table of Contents

- [Project Structure](#project-structure)
- [Detailed Code Explanation](#detailed-code-explanation)
  - [1. Data Loading and Graph Construction](#1-data-loading-and-graph-construction)
  - [2. Defining the GNN Model](#2-defining-the-gnn-model)
  - [3. Training the GNN](#3-training-the-gnn)
  - [4. Interactive Recommendation](#4-interactive-recommendation)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Additional Information](#additional-information)
- [License](#license)
- [Contributing](#contributing)

---

## Project Structure

```
GNN-Music-Recommendation-System/
├── app.py                   # Main script: loads data, builds the graph, trains the GNN, and runs recommendations.
├── notebook.ipynb           # Jupyter Notebook with an interactive walk-through of the data preparation, model training, and inference.
├── other/
│   ├── h1.ipynb             # Additional interactive examples and demos.
│   └── h1.md                # Detailed documentation including negative sampling and evaluation.
├── spotify_songs.csv        # CSV file with song and playlist metadata.
├── .gitignore               # Git ignore rules.
└── README.md                # This file.
```

---

## Detailed Code Explanation

### 1. Data Loading and Graph Construction

**CSV Loading and Data Preparation:**  
- The code uses **Pandas** to read `spotify_songs.csv`, which contains essential columns such as `playlist_id`, `track_id`, `track_name`, and audio features like `danceability`, `energy`, `loudness`, etc.
- **Unique Node Sets:**  
  - **Playlists:** Each unique `playlist_id` is assigned an index (0 to `num_playlists - 1`).
  - **Tracks:** Each unique `track_id` is assigned an index starting from `num_playlists` so that playlist and track nodes do not overlap.
- **Mapping IDs to Indices:**  
  Two dictionaries, `playlist_id_to_index` and `track_id_to_index`, are created to map original IDs to their respective node indices.
- **Edge Construction:**  
  - For every occurrence of a track in a playlist, an edge is added from the playlist node to the track node.
  - To represent an undirected graph, edges are added in both directions (playlist → track and track → playlist).
- **Node Features:**  
  - **Track Nodes:** Audio features (e.g., `danceability`, `energy`, `tempo`, etc.) are extracted for each track. The code also handles cases where feature conversion might fail by using zeros.
  - **Playlist Nodes:** Since there are no inherent features, these nodes are initialized with random embeddings.
  - **Normalization:**  
    Track features are optionally normalized using min–max scaling to bring all feature values into a similar range.
- **Data Object:**  
  The features of both playlists and tracks are concatenated to form the overall feature matrix `x`. A PyTorch Geometric `Data` object is then created using this matrix and the computed edge indices.

### 2. Defining the GNN Model

**Model Architecture (`GNNRec` Class):**  
- **Layer 1:**  
  A `GCNConv` layer that transforms the input node features to a hidden representation.
- **Activation and Dropout:**  
  A ReLU activation function introduces non-linearity followed by a dropout layer to prevent overfitting during training.
- **Layer 2:**  
  A second `GCNConv` layer that outputs the final embeddings for each node.
- **Hyperparameters:**  
  - `in_channels` corresponds to the number of features (audio features for tracks).
  - `hidden_channels` (e.g., 128) and `out_channels` (e.g., 64) can be adjusted as needed.

### 3. Training the GNN

**Training Process Overview:**

- **Positive Sampling:**  
  - Uses actual playlist–track pairs from the CSV.
  - The dot product between corresponding playlist and track embeddings is computed. This score is passed through a sigmoid function and then used in a log-sigmoid loss, encouraging the model to produce higher scores for real interactions.
- **Negative Sampling:**  
  - Random pairs of playlist and track nodes that are not connected are sampled.
  - The same dot product operation is applied and the loss is designed (using `-log(1 - sigmoid(score))`) to push these negative pairs' scores lower.
- **Loss and Optimization:**  
  The overall loss is the sum of positive and negative losses. The Adam optimizer is then used to update the model weights over multiple epochs (e.g., 100 epochs), with periodic logging to monitor the training progress.

### 4. Interactive Recommendation

**Post-training Embedding Extraction:**

- **Final Embeddings:**  
  Once training is complete, final node embeddings are computed. The embeddings are then separated into:
  - **Playlist Embeddings:** First part of the overall embeddings.
  - **Track Embeddings:** The remaining embeddings.
  
**Interactive Song Recommendation:**  
- **Query Processing:**  
  A helper function (`get_track_node_index_by_name`) looks up the song by its name (using a case-insensitive exact match) and retrieves its corresponding node index.
- **Similarity Calculation:**  
  - The embedding of the query song is compared against all track embeddings using cosine similarity.
  - Both query and track embeddings are normalized (unit vectors) to compute cosine similarity reliably.
- **Excluding the Query:**  
  The system ensures that the queried song is not included in the recommendations.
- **Output:**  
  The top 10 songs with the highest similarity scores are then returned as recommendations, displaying their names based on metadata.

---

## Dependencies

The project uses the following Python libraries:
- [Pandas](https://pandas.pydata.org/) for CSV file handling.
- [NumPy](https://numpy.org/) for numerical operations.
- [PyTorch](https://pytorch.org/) for tensor computation and model training.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network layers and data management.

To install the required packages (preferably in a virtual environment):

```bash
pip install pandas numpy torch torch-geometric
```

> **Note:** Additional dependencies (e.g., `networkx` or specific versions of torch-related packages) might be required. Check the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for further details.

---

## Getting Started

1. **Prepare Your CSV:**  
   Ensure that your CSV file (e.g., `spotify_songs.csv`) is in the root directory (or update the file path in `app.py`). The CSV should include columns like `playlist_id`, `track_id`, `track_name`, and audio feature columns (`danceability`, `energy`, `loudness`, etc.).

2. **Graph Construction:**  
   The code builds a bipartite graph:
   - **Playlist Nodes:** Indices `0` to `num_playlists - 1`
   - **Track Nodes:** Indices `num_playlists` to `total_nodes - 1`
   - Undirected edges are created by adding both directions between playlists and tracks.

3. **Node Features:**  
   - Track nodes use audio features.
   - Playlist nodes are initialized with random values.

4. **Model Training:**  
   The two-layer GCN is trained using a combination of positive and negative sampling loss. Training progress is logged (e.g., every 10 epochs).

5. **Interactive Recommendation:**  
   After training, the system enters an interactive loop where you can input a song name to receive 10 recommended similar songs.

---

## Running the Project

### Using the Command Line (Script)
Run the main application with:
```bash
python app.py
```
This command will:
- Load the CSV data.
- Build the graph.
- Train the GNN model.
- Launch an interactive recommendation prompt.

### Using Jupyter Notebook
Alternatively, open `notebook.ipynb` (or `other/h1.ipynb`) in Jupyter Notebook or VS Code’s interactive window to walk through the steps of data preparation, model training, and inference interactively.

---

## Additional Information

- **Data Preparation & Graph Construction:**  
  Refer to the detailed explanation in the notebooks (`notebook.ipynb`, `other/h1.ipynb`) and the accompanying documentation (`other/h1.md`).
- **Model Architecture:**  
  The GNN model (`GNNRec` class) is defined in `app.py` and also explained in the notebooks. Adjust hyperparameters (e.g., learning rate, epochs) as necessary.
- **Interactive Recommendation:**  
  The interactive loop in `app.py` demonstrates how to query by song name and obtain recommendations based on cosine similarity between learned embeddings.

