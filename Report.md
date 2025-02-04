# Graph Neural Networks for Music Recommendation: A Detailed Report

## Abstract

In the era of digital music streaming, delivering personalized recommendations is paramount to engaging users and enhancing their listening experience. This report outlines a cutting-edge approach to music recommendation using Graph Neural Networks (GNNs). By modeling the relationships between playlists and tracks as a bipartite graph and leveraging audio features as node attributes, our system learns robust embeddings through a two-layer Graph Convolutional Network (GCN). These embeddings are then used to recommend songs similar in style and mood. The report covers the theoretical foundations of GNNs, details on data preparation and graph construction, the model architecture, training methodologies including positive and negative sampling, and an interactive recommendation process.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background and Motivation](#background-and-motivation)
3. [Project Overview and Architecture](#project-overview-and-architecture)
   - 3.1 [Project Structure](#project-structure)
   - 3.2 [Graph Construction and Data Preparation](#graph-construction-and-data-preparation)
4. [GNN Fundamentals for Music Recommendation](#gnn-fundamentals-for-music-recommendation)
   - 4.1 [Graph Representation](#graph-representation)
   - 4.2 [Message Passing and Graph Convolution](#message-passing-and-graph-convolution)
5. [Detailed Model Architecture](#detailed-model-architecture)
   - 5.1 [The GNN Model: `GNNRec` Class](#the-gnn-model-gnnrec-class)
   - 5.2 [Node Feature Engineering](#node-feature-engineering)
6. [Training the GNN](#training-the-gnn)
   - 6.1 [Positive and Negative Sampling](#positive-and-negative-sampling)
   - 6.2 [Loss Function and Optimization](#loss-function-and-optimization)
7. [Interactive Recommendation System](#interactive-recommendation-system)
8. [Dependencies and Getting Started](#dependencies-and-getting-started)
9. [Conclusion and Future Directions](#conclusion-and-future-directions)
10. [References](#references)

---

## 1. Introduction

Personalized music recommendation is a cornerstone of modern streaming platforms, where users expect seamless discovery of new tracks and curated playlists. Traditional recommendation methods—such as collaborative filtering and content-based filtering—have achieved significant success but are often limited by challenges like data sparsity and the inability to capture nuanced relationships between songs and user preferences.

Graph Neural Networks (GNNs) address these challenges by modeling interactions as graphs, where nodes represent entities (playlists and tracks) and edges represent their interactions. This report explains our implementation of a GNN-based music recommendation system, detailing every step from data preparation to interactive recommendation.

---

## 2. Background and Motivation

### Traditional Methods and Their Limitations

- **Collaborative Filtering:** Leverages historical user–item interactions but suffers from cold-start and sparsity issues.
- **Content-Based Filtering:** Uses item metadata (e.g., audio features, genre) but may miss collaborative insights.
- **Hybrid Approaches:** Combine both methods but often struggle to effectively balance multiple sources of information.

### Why Graph Neural Networks?

GNNs provide a natural framework for representing complex interactions. In our approach, a bipartite graph is constructed with two distinct node sets:
- **Playlists:** Each playlist is represented as a node, initially assigned a random embedding.
- **Tracks:** Nodes are enriched with audio features such as danceability, energy, loudness, tempo, etc.

Edges are created based on playlist–track interactions, and by training a simple two-layer Graph Convolutional Network (GCN), the model learns meaningful representations for both playlists and tracks. These representations are then used to identify and recommend similar songs.

---

## 3. Project Overview and Architecture

### 3.1 Project Structure

The project is organized as follows:

```
GNN-Music-Recommendation-System/
├── app.py                   # Main script: data loading, graph building, GNN training, and recommendation.
├── notebook.ipynb           # Jupyter Notebook for an interactive walk-through.
├── other/
│   ├── h1.ipynb             # Additional interactive examples and demos.
│   └── h1.md                # Detailed documentation on negative sampling and evaluation.
├── spotify_songs.csv        # CSV file containing song and playlist metadata.
├── .gitignore               # Git ignore rules.
└── README.md                # Project documentation.
```

### 3.2 Graph Construction and Data Preparation

1. **Data Loading:**
   - The CSV file (`spotify_songs.csv`) is read using Pandas. Key columns include `playlist_id`, `track_id`, `track_name`, and various audio features.
   - Unique playlists and tracks are identified. Playlists receive indices from 0 to `num_playlists - 1`, and track nodes receive indices starting from `num_playlists` to ensure no overlap.

2. **Mapping and Edge Construction:**
   - Two dictionaries, `playlist_id_to_index` and `track_id_to_index`, are created to map original IDs to graph node indices.
   - Edges are added for every occurrence of a track in a playlist. To simulate an undirected graph, each edge is added twice (playlist → track and track → playlist).

3. **Node Feature Engineering:**
   - **Track Nodes:** Audio features such as danceability, energy, and tempo are extracted and, if needed, normalized using min–max scaling.
   - **Playlist Nodes:** Initialized with random embeddings as they lack inherent features.
   - The final feature matrix is assembled and encapsulated in a PyTorch Geometric `Data` object alongside the edge indices.

---

## 4. GNN Fundamentals for Music Recommendation

### 4.1 Graph Representation

- **Nodes:** Represent entities (playlists and tracks) with associated feature vectors.
- **Edges:** Capture interactions such as a track being part of a playlist.
- **Bipartite Graph:** Naturally models the two distinct sets of nodes and their interrelations.

### 4.2 Message Passing and Graph Convolution

- **Message Passing:** Nodes update their representations by aggregating information from their neighbors.
- **Aggregation:** Functions (mean, sum, or attention-based) are used to combine messages.
- **Graph Convolution:** Similar to convolutions in images, graph convolutions enable the extraction of local structural features, which is critical for understanding both the collaborative and content-based aspects of music recommendation.

---

## 5. Detailed Model Architecture

### 5.1 The GNN Model: `GNNRec` Class

The model is implemented using PyTorch Geometric and consists of a two-layer GCN:

```python
import torch
from torch_geometric.nn import GCNConv

class GNNRec(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GNNRec, self).__init__()
        # Embedding layer for node features (playlists have random initial embeddings; tracks use audio features)
        self.embedding = torch.nn.Embedding(num_nodes, in_channels)
        # First GCN layer: transforms input features into a hidden representation
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Activation and dropout for non-linearity and regularization
        self.dropout = torch.nn.Dropout(p=0.5)
        # Second GCN layer: outputs the final embedding
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: initial node features; edge_index: graph connectivity
        x = self.embedding(x)  # Apply initial embedding (for playlists, these are random; for tracks, pre-computed audio features)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
```

### 5.2 Node Feature Engineering

- **Track Nodes:** Use robust audio features such as danceability, energy, and loudness. Normalization ensures all features contribute uniformly.
- **Playlist Nodes:** Although lacking intrinsic features, these nodes start with random embeddings that evolve through training to capture the latent characteristics of playlists.

---

## 6. Training the GNN

### 6.1 Positive and Negative Sampling

- **Positive Sampling:**  
  Real playlist–track pairs from the CSV form the positive samples. The dot product between corresponding node embeddings (after passing through the GCN) is computed. A sigmoid function then squashes these scores into the range [0, 1], and a log-sigmoid loss is applied to encourage high scores for true interactions.

- **Negative Sampling:**  
  To ensure the model learns to differentiate true interactions from random noise, negative pairs are generated by sampling random playlist–track pairs that do not have an observed interaction. The loss function penalizes high similarity scores for these pairs using a complementary log-sigmoid loss formulation.

### 6.2 Loss Function and Optimization

- **Loss Calculation:**  
  The overall loss is computed as the sum of the positive and negative sampling losses. This loss function directly encourages the model to assign higher similarity scores to actual playlist–track pairs while suppressing scores for unrelated pairs.
  
- **Optimization:**  
  The Adam optimizer is used to update model weights over multiple epochs (e.g., 100 epochs), with periodic logging to monitor convergence and performance.

---

## 7. Interactive Recommendation System

After training, the system extracts final node embeddings which are then split into:
- **Playlist Embeddings:** Represent the latent characteristics of playlists.
- **Track Embeddings:** Represent the nuanced audio and contextual features of songs.

### Interactive Process

1. **Querying by Song Name:**  
   A helper function (`get_track_node_index_by_name`) performs a case-insensitive lookup of a track by name to retrieve its node index.
   
2. **Similarity Calculation:**  
   The system computes cosine similarity between the queried song’s embedding and all other track embeddings. Both sets of embeddings are normalized to unit vectors, ensuring a reliable cosine similarity calculation.

3. **Recommendation Output:**  
   The queried song is excluded from the list, and the top 10 most similar songs (based on similarity scores) are presented as recommendations. This interactive module enables real-time feedback and fine-tuning of the recommendation process.

---

## 8. Dependencies and Getting Started

### Dependencies

The project is built using Python and relies on the following libraries:
- **Pandas:** For CSV handling and data manipulation.
- **NumPy:** For numerical operations.
- **PyTorch:** For tensor computation and deep learning model training.
- **PyTorch Geometric:** For graph-based neural network layers and data management.

Install the dependencies using:

```bash
pip install pandas numpy torch torch-geometric
```

*Additional dependencies (e.g., networkx) may be required depending on your environment. Refer to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for more details.*

### Getting Started

1. **CSV Preparation:**  
   Ensure that your `spotify_songs.csv` is correctly formatted and placed in the root directory. Required columns include `playlist_id`, `track_id`, `track_name`, and audio features (e.g., `danceability`, `energy`, `loudness`).

2. **Graph Construction:**  
   The graph is built as a bipartite graph with playlists and tracks assigned distinct node indices. Both node types receive appropriate features: tracks from audio metadata and playlists from randomly initialized embeddings.

3. **Model Training and Inference:**  
   Run the main application via the command line:

   ```bash
   python app.py
   ```

   This script will load the CSV data, construct the graph, train the GNN, and launch an interactive session for song recommendations. Alternatively, you can explore the process interactively using `notebook.ipynb` or `other/h1.ipynb`.

---

## 9. Conclusion and Future Directions

This report has presented an end-to-end description of a GNN-based music recommendation system. By constructing a bipartite graph to model playlists and tracks and leveraging audio features and collaborative interactions, the system effectively learns embeddings that capture the underlying structure of music data. 

### Future Research Directions

- **Deeper Architectures:**  
  Investigate multi-layer GNNs that capture higher-order interactions.

- **Dynamic Models:**  
  Develop models that adapt to temporal changes in user behavior and music trends.

- **Enhanced Metadata Integration:**  
  Incorporate additional modalities (lyrics, images, user reviews) to enrich the recommendation process.

- **Scalability:**  
  Improve graph sampling and optimization strategies to efficiently handle large-scale music catalogs.

- **Explainability:**  
  Enhance model transparency to better understand the factors influencing recommendations.

---

## 10. References

1. **Graph Neural Networks for Recommender Systems (WSDM '22 Tutorial):** A comprehensive tutorial covering the design and challenges of GNN-based recommendation systems.
2. **A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions:** An extensive review of various GNN models applied to recommendation tasks.
3. **Graph Neural Network and Context-Aware Based User Behavior Prediction and Recommendation System Research:** A study that integrates contextual information with GNNs to enhance recommendation accuracy.
4. **Graph-Based Attentive Sequential Model With Metadata for Music Recommendation:** Research that fuses sequential modeling and metadata using GNNs for improved personalization.
5. **Hybrid Music Recommendation with Graph Neural Networks:** A hybrid approach combining collaborative filtering and content-based filtering using GNNs.
