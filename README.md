# GNN Music Recommendation System

This project uses a Graph Neural Network (GNN) to build a recommendation system for music playlists and tracks based on audio features and playlist-track interactions. It uses a bipartite graph where one set of nodes represents playlists and the other represents tracks. Track audio features are used as node attributes and a simple two-layer GNN (with GCNConv layers from PyTorch Geometric) learns meaningful embeddings for recommendation.

## Project Structure

- **`app.py`**  
  Contains the main implementation for:
  - Loading the CSV data ([`spotify_songs.csv`](spotify_songs.csv))
  - Building unique node sets and the bipartite graph
  - Creating node features (tracks use audio features while playlists are initialized with zeros or random embeddings)
  - Defining and training the GNN model
  - Performing interactive recommendation

- **`notebook.ipynb`**  
  Provides an interactive walk-through of:
  - Data preparation
  - Model definition and training
  - Running inference and interactive song recommendations

- **`other/h1.ipynb` & `other/h1.md`**  
  These files offer additional examples and documentation, including:
  - Detailed steps on negative sampling loss and evaluation
  - Additional demos for constructing the graph, training the model, and performing inference

- **`spotify_songs.csv`**  
  A CSV file with song and playlist metadata used to build the graph. Each row corresponds to a track occurrence in a playlist.

- **`.gitignore`**  
  Specifies files and directories to ignore in version control.

## Dependencies

The project uses the following Python libraries:
- [Pandas](https://pandas.pydata.org/) for CSV file manipulation.
- [NumPy](https://numpy.org/) for numerical operations.
- [PyTorch](https://pytorch.org/) for tensor computation and model training.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network layers and data handling.

Install the required packages (preferably in a virtual environment):

```sh
pip install pandas numpy torch torch-geometric
```

_Note: Additional dependencies (e.g., networkx, or specific versions of torch packages) may be required as indicated during installation._

## Getting Started

1. **Prepare your CSV:**  
   Update the CSV path in `app.py` if necessary. The CSV file should contain columns such as `playlist_id`, `track_id`, `danceability`, `energy`, `loudness`, etc.

2. **Graph Construction:**  
   The project constructs a bipartite graph where playlists and tracks form the nodes (see the graph building section in `app.py`).  
   - Playlist nodes are assigned indices `0` to `num_playlists - 1`.
   - Track nodes are assigned indices `num_playlists` to `total_nodes - 1`.
   - Undirected edges are built by adding both directions (playlist → track and track → playlist).

3. **Node Features:**  
   Track nodes are assigned audio features (e.g., `danceability`, `energy`, `loudness`, etc.) while playlists are initialized with zeros or random values.

4. **Model Training:**  
   A simple two-layer GCN (based on `GCNConv`) is defined and trained using positive and negative sampling loss.
   - See `notebook.ipynb` for an interactive approach and training output logs.
   - Training logs indicate epoch progress and loss values.

5. **Interactive Recommendation:**  
   After training, the model computes final embeddings, and you can perform recommendations by:
   - Mapping the global track node indices back to track IDs.
   - Retrieving track names using the metadata (e.g., via `track_df` in `notebook.ipynb` or `other/h1.ipynb`).
   - Running an interactive loop to enter a song name and receive top-10 recommendation results.

## Running the Project

### Using the Command Line (Script)
Run the main application with:
```sh
python app.py
```
Training progress and interactive prompts will appear on the console.

### Using Jupyter Notebook
Open `notebook.ipynb` in Jupyter Notebook or VS Code’s interactive window to step through the code cells for data preparation, model training, and inference.

## Additional Information

- **Data Preparation & Graph Construction:**  
  See `h1.md` for a detailed explanation on building the graph and creating node features.

- **Model Architecture:**  
  The model is defined in `app.py` and `h1.ipynb` under the `GNNRec` class. Adjust hyperparameters (e.g., learning rate, number of epochs) as needed.

- **Interactive Recommendation:**  
  Both the scripts and notebooks include sections for interactive song recommendations. Modify the mapping and similarity components based on your evaluation needs.

Feel free to explore and adjust the code to fit your dataset and requirements. If you have any questions or need further adjustments, check the inline documentation in the corresponding files.
