import os
import json
import sys
import shutil
import pickle
from train_extract_GFLIMBoFP import *
from PIL import Image
import numpy as np
import heapq

sys.path.insert(1, '/home/isabela/Codes/flim-python')
from pyflim import metrics

np.random.seed(42)

"""
1) read the original and labeled image (1 image)
2) read the graph and superpixel image (1 image)
3) Select a random object and background nodes
4) Let iteration = 1
4) Save the selected nodes as markers
5) Rename the saliency folders to match the iteration
5) Train and evaluate the model with the selected markers
6) Save the evaluation results
7) If iteration > 20, stop the process. Otherwise, proceed
8) Set iteration = iteration + 1
9) Compute the node features with neighborhood aggregation
10) Find the most distant node from the markers and insert it as a new marker
11) Repeat from step 4
"""

class GFLIMData:
    def __init__(self, image_path : str, label_path : str, graph_path : str, superpixel_path : str, num_neighbors : int):
        """Initializes the GFLIMData class.
        arguments:
            image_path: path to the original image
            label_path: path to the ground truth image
            graph_path: path to the graph data
            superpixel_path: path to the superpixel image
            num_neighbors: number of neighbors to consider for feature aggregation
        """
        print("Loading images...")
        self.image = self.readImage(image_path)
        self.label = self.readImage(label_path)
        if self.label.ndim == 3:
            self.label = self.label[:,:,0]  # Assuming the label image is RGB, we take the first channel
        print('Loading graph and superpixel data...')
        self.graph = self.readGraph(graph_path)
        self.superpixel = self.readImage(superpixel_path)
        print('Computing edges ..')
        self.edges = self.readEdges()
        print('Getting node labels ..')
        self.getNodeLabels()
        print('Computing neighborhood features matrix ..')
        self.getNeighbFeaturesMatrix(num_neighbors)  

    def readImage(self, path : str):
        """
        Reads an image from the given path and returns it as a numpy array.
        """
        img = Image.open(path)
        return np.array(img)

    def readGraph(self, path : str):
        """
        Reads the graph data from the graph path.
        """
        with open(path, 'r') as f:
            graph_data = json.load(f)
        return graph_data

    def readEdges(self):
        """
        Reads the edges from the graph data.
        """
        if self.graph is None or 'num_nodes' not in self.graph:
            return {}
        
        edges = {}
        for node_id in range(int(self.graph['num_nodes'])):
            edges[node_id] = set()
        
        for node_id in range(int(self.graph['num_nodes'])):
            node_data = self.graph[f"node{node_id}"]
            if 'neighbors' in node_data:
                for neighbor in node_data['neighbors']:
                    edges[node_id].add(neighbor)
                    edges[neighbor].add(node_id)

        return edges

    def construct_feat_vector(self, Ni : list):
        """
        Construct a feature vector from the k nearest neighbors.
        arguments:
            Ni: list of tuples (distance, neighbor_id)
            graph: the graph containing node features
        returns:
            feats: a list of features from the k nearest neighbors
        """
        num_feats = self.graph["num_feats"]
        k = len(Ni)

        # Pre-allocate the array
        feats = np.empty(k * num_feats, dtype=np.float32)

        # We need to process Ni in sorted order
        for i, (_, vj) in enumerate(heapq.nsmallest(k, Ni)):
            feats[i*num_feats:(i+1)*num_feats] = self.graph[f"node{vj}"]["feats"]
        return feats

    def get_k_neighbors(self, vi : int, k : int):
        """ 
        Get the k closest neighbors according to GFLIM neighbors selection strategy.
        arguments:
            vi: the id of the superpixel to find neighbors for
            k: number of neighbors to select
        returns:
            Ni: a min-heap of the k closest neighbors, where each element is a tuple
            (distance, neighbor_id)
        """
        if len(self.graph) == 0:
            return []

        # Initialize with target node
        Ni = [(0.0, vi)]
        heapq.heapify(Ni)
        visited = {vi}

        # Cache target features once
        target_feats = np.asarray(self.graph[f"node{vi}"]["feats"])

        # select k - 1 nearest neighbors of superpixel vi
        while(len(Ni) < k):

            # Get all unique neighbors of current set
            neighbors = set()
            for _, s in Ni:
                neighbors.update(self.edges.get(s, set()))

            # Get new candidates
            candidates = list(neighbors - visited)
            if not candidates:
                break
            
            # Prepare features for batch distance calculation
            candidate_feats = np.array([self.graph[f"node{vj}"]["feats"] for vj in candidates])

            # Calculate distances for all candidates
            dists = np.sum((candidate_feats - target_feats) ** 2, axis=1)

            # Add candidates to buffer
            candidate_buffer = list(zip(dists, candidates))
            visited.update(candidates)

            # If we have enough candidates, we can sort and keep only the k smallest
            if len(Ni) + len(candidate_buffer) >= k:
                candidate_buffer.sort()

            for dist, vj in candidate_buffer[:k - len(Ni)]:
                heapq.heappush(Ni, (dist, vj))

        return Ni

    def getNeighbFeaturesMatrix(self, num_neighbors : int):
        if len(self.graph) == 0 or len(self.edges) == 0:
            return

        num_nodes = self.graph["num_nodes"]
        feat_matrix = np.zeros((num_nodes, num_neighbors * self.graph["num_feats"]), dtype=np.float32)
        for vi in range(num_nodes):
            Ni = self.get_k_neighbors(vi, num_neighbors)
            feat_matrix[vi] = self.construct_feat_vector(Ni)

        self.feat_matrix = feat_matrix

    def getNodeLabels(self):
        """
        Efficiently assigns labels to nodes by vectorized mapping from superpixel image.
        """
        num_nodes = self.graph["num_nodes"]
        self.node_labels = np.zeros(num_nodes, dtype=np.int64)

        # Flatten superpixel and label images for fast lookup
        spx_flat = self.superpixel.flatten()
        label_flat = self.label.flatten()

        # For each node, find the first occurrence of its superpixel index and get the label
        for node_id in range(num_nodes):
            idx = np.flatnonzero(spx_flat == node_id + 1)
            if idx.size > 0:
                # get the label with the maximum occurrence in the superpixel
                labels_in_spx = label_flat[idx]
                # if labels 0 and 255 are present have the maximum occurence, we consider 255 as the label
                unique_labels, counts = np.unique(labels_in_spx, return_counts=True)
                obj_spxs = counts[unique_labels == 255]
                if len(obj_spxs) > 0 and counts.argmax() == obj_spxs[0]:
                    self.node_labels[node_id] = 255
                else:
                    self.node_labels[node_id] = unique_labels[counts.argmax()]
            else:
                raise ValueError(f"Superpixel {node_id} not found in the superpixel image.")

    def compute_TSNE(self, perplexity : int = 30, n_components : int = 2, random_state : int = 42):
        """
        Computes the t-SNE embedding of the features matrix.
        arguments:
            perplexity: t-SNE perplexity parameter
            n_components: number of dimensions for the embedding
            random_state: random seed for reproducibility
        """
        from sklearn.manifold import TSNE

        if not hasattr(self, 'feat_matrix'):
            raise ValueError("Features matrix not computed. Please run getNeighbFeaturesMatrix() first.")

        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        self.feat_matrix_tsne = tsne.fit_transform(self.feat_matrix)


class MarkersSelection:
    def __init__(self, gflim_data : GFLIMData, model : GFLIMModel, 
                 markers_path : str, img_name : str, original_image_path : str, labels_path : str,
                 save_path : str, max_markers : int=20,
                 initial_iteration : int = -1, tsne : bool = False):
        """
        Initializes the MarkersSelection class.
        arguments:
            gflim_data: GFLIMData object containing the graph and superpixel data, and the neighbor features matrix
            model: GFLIMModel object containing the model to be trained
            markers_path: path to save the image markers
            img_name: name of the image from which markers are selected
            original_image_path: path to the original images for evaluation
            labels_path: path to the image labels for evaluation
            save_path: path to save the results
            max_markers: maximum number of markers to select (default is 20)
            initial_iteration: initial iteration for the selection process. 
                If -1, it will read the last iteration saved. 
                Otherwise, it will start from the given iteration (default is -1)
        """
        self.gflim_data = gflim_data
        self.model = model
        self.max_markers = max_markers
        self.markers = []
        self.markers_path = markers_path
        self.img_name = img_name
        self.save_path = save_path
        self.original_image_path = original_image_path # for evaluation purposes
        self.labels_path = labels_path # for evaluation purposes
        self.initial_iteration = initial_iteration
        self.tsne = tsne
        if tsne:
            self.gflim_data.compute_TSNE()

    def evaluate_node_border(self, node_id : int):
        """
        Evaluates if a node is on the border of the object considering the neighbors from the features matrix.
        arguments:
            node_id: the id of the node to evaluate
        returns:
            True if the node is on the border, False otherwise
        """
        num_neighbors = int(len(self.gflim_data.feat_matrix[0]) / self.gflim_data.graph["num_feats"])
        Ni = self.gflim_data.get_k_neighbors(node_id, num_neighbors)

        # get the labels of the neighbors and verify if there is a mix of labels
        labels = [self.gflim_data.node_labels[vj] for _, vj in Ni]
        return len(np.unique(labels)) > 1

    def select_random_marker(self, markers, label : int, avoid_neighbors : bool = False):
        """
        Selects a random node from the graph.
        """

        if label is not None and label not in np.unique(self.gflim_data.node_labels):
            raise ValueError(f"Label {label} is not present in the node labels.")

        # select nodes with the given label
        nodes_with_label = np.where(self.gflim_data.node_labels == label)[0]
        
        # Avoid selecting neighbors of existing markers
        if avoid_neighbors and len(markers) > 0:
            markers = set(markers)
            for m in markers:
                markers.update(self.gflim_data.edges.get(m, set()))
            markers = np.array(list(markers))

        # remove nodes that are already selected as markers
        nodes_with_label = np.setdiff1d(nodes_with_label, markers)
        if len(nodes_with_label) == 0:
            raise ValueError(f"No nodes with label {label} available for selection.")
        if len(nodes_with_label) < 1:
            raise ValueError(f"Not enough nodes with label {label} to select a marker.")
        
        # select a random marker from the remaining nodes
        if len(nodes_with_label) == 1:
            marker = nodes_with_label[0]
        else:
            marker = np.random.choice(nodes_with_label, replace=False)

        return marker

    def node_homogeneity(self, node_id : int, num_feats : int, num_neighbors : int):
        """
        Calculates the homogeneity of a node based on its features.
        arguments:
            node_id: the id of the node to evaluate
        returns:
            homogeneity score as the mean of the standard deviations of the features
        """

        # Get the features of the node and reshape them into k nearest neighbors
        features = self.gflim_data.feat_matrix[node_id].reshape((num_neighbors, num_feats))
        
        # Calculate the homogeneity score as the mean of the standard deviations of the features
        return np.mean(np.std(features, axis=0))

    def select_node(self, markers, node_scores, select_function, avoid_neighbors):
        """
        Checks if the node_id is a neighbor of any existing markers.
        arguments:
            markers: list of existing markers
            node_id: the id of the node to check
        returns:
            True if the node_id is a neighbor of any marker, False otherwise
        """
        dist, node = select_function(node_scores, key=lambda x: x[0])

        # Avoid selecting neighbors of existing markers
        if avoid_neighbors and len(markers) > 0:
            neighbors = set()
            if node in neighbors:
                for m in markers:
                    neighbors.update(self.gflim_data.edges.get(m, set()))
            while node in neighbors:
                node_scores.remove((dist, node))
                dist, node = select_function(node_scores, key=lambda x: x[0])

        return node

    def select_most_homogeneous_marker(self, markers, label=None, avoid_neighbors : bool = False):
        """
        Selects the most homogeneous marker for the given label.
        """

        num_feats = self.gflim_data.graph["num_feats"]
        num_neighbors = int(len(self.gflim_data.feat_matrix[0]) / num_feats)

        if label is not None and label not in np.unique(self.gflim_data.node_labels):
            raise ValueError(f"Label {label} is not present in the node labels.")

        # If we already have markers, select the most homogeneous node from the existing markers
        variance_scores = []
        for node_id in range(self.gflim_data.graph["num_nodes"]):

            # Skip nodes that are already selected as markers
            if node_id in markers:
                continue

            # If a label is specified, skip nodes that do not match the label
            if label is not None and self.gflim_data.node_labels[node_id] != label:
                continue

            # Calculate the homogeneity score as the mean of the standard deviations of the features
            variance_scores.append((self.node_homogeneity(node_id, num_feats, num_neighbors), node_id))

        if not variance_scores:
            raise ValueError("No valid nodes left to select.")

        # Select the node with the lowest variance
        return self.select_node(markers, variance_scores, min, avoid_neighbors)

    def select_marker_euclDist(self, markers, label=None, avoid_neighbors : bool = False, avoid_borders : bool = False):
        """
        Selects a random marker for the given label.
        """

        if len(markers) == 0:
            raise ValueError("No markers selected yet. Please select a marker first.")
        
        if label is not None and label not in np.unique(self.gflim_data.node_labels):
            raise ValueError(f"Label {label} is not present in the node labels.")

        # If we already have markers, select the most distant node from the existing markers
        distances = []
        for node_id in range(self.gflim_data.graph["num_nodes"]):

            # Skip nodes that are already selected as markers
            if node_id in markers:
                continue

            # If a label is specified, skip nodes that do not match the label
            if label is not None and self.gflim_data.node_labels[node_id] != label:
                continue

            # Calculate the distance to all existing markers
            dist = np.mean([np.linalg.norm(self.gflim_data.feat_matrix[node_id] - self.gflim_data.feat_matrix[m]) for m in markers])
            distances.append((dist, node_id))

        if not distances:
            raise ValueError("No valid nodes left to select.")
        
        # Select the most distant node
        return self.select_node(markers, distances, max, avoid_neighbors)

    def select_marker_euclDist_std(self, markers, label=None, avoid_neighbors : bool = False):
        """
        Selects a random marker for the given label.
        """

        num_feats = self.gflim_data.graph["num_feats"]
        num_neighbors = int(len(self.gflim_data.feat_matrix[0]) / num_feats)

        if len(markers) == 0:
            raise ValueError("No markers selected yet. Please select a marker first.")
        
        if label is not None and label not in np.unique(self.gflim_data.node_labels):
            raise ValueError(f"Label {label} is not present in the node labels.")

        # If we already have markers, select the most distant node from the existing markers
        distances, inhomogeneity_scores, nodes = [], [], []
        for node_id in range(self.gflim_data.graph["num_nodes"]):

            # Skip nodes that are already selected as markers
            if node_id in markers or node_id in [19410, 19970, 19980, 18950, 19679, 19993, 19075, 19242, 19800, 19722, 18754, 4298, 16775, 18722, 19288, 18552, 18550, 13485, 19317, 15297, 6504, 19371, 13126, 17007]:
                continue

            # If a label is specified, skip nodes that do not match the label
            if label is not None and self.gflim_data.node_labels[node_id] != label:
                continue

            # Calculate the distance to all existing markers
            dist = np.mean([np.linalg.norm(self.gflim_data.feat_matrix[node_id] - self.gflim_data.feat_matrix[m]) for m in markers])
            distances.append(dist)
            inhomogeneity_scores.append(self.node_homogeneity(node_id, num_feats, num_neighbors))
            nodes.append(node_id)

        # normalize the inhomogeneity scores with min-max scaling
        if len(inhomogeneity_scores) > 0:
            min_val = min(inhomogeneity_scores)
            max_val = max(inhomogeneity_scores)
            if max_val - min_val > 0:
                inhomogeneity_scores = [(inhomog - min_val) / (max_val - min_val) for inhomog in inhomogeneity_scores]
            else:
                inhomogeneity_scores = [0.0 for _ in inhomogeneity_scores]
        
        # normalize the distances with min-max scaling
        if len(distances) > 0:
            min_val = min(distances)
            max_val = max(distances)
            if max_val - min_val > 0:
                distances = [(dist - min_val) / (max_val - min_val) for dist in distances]
            else:
                distances = [0.0 for _ in distances]

        # Combine distances and inhomogeneity scores
        combined_scores = []
        for dist, node, inhomog in zip(distances, nodes, inhomogeneity_scores):
            combined_scores.append((dist - inhomog, node))

        if not combined_scores:
            raise ValueError("No valid nodes left to select.")
        
        # Select the most distant node
        return self.select_node(markers, combined_scores, max, avoid_neighbors)

    def select_marker_euclDistTSNE_std(self, markers, label=None, avoid_neighbors : bool = False):
        """
        Selects a random marker for the given label.
        """

        feat_matrix = self.gflim_data.feat_matrix_tsne if self.tsne else self.gflim_data.feat_matrix

        num_feats = self.gflim_data.graph["num_feats"]
        num_neighbors = int(len(self.gflim_data.feat_matrix[0]) / num_feats)

        if len(markers) == 0:
            raise ValueError("No markers selected yet. Please select a marker first.")
        
        if label is not None and label not in np.unique(self.gflim_data.node_labels):
            raise ValueError(f"Label {label} is not present in the node labels.")

        # If we already have markers, select the most distant node from the existing markers
        distances, inhomogeneity_scores, nodes = [], [], []
        for node_id in range(self.gflim_data.graph["num_nodes"]):

            # Skip nodes that are already selected as markers
            if node_id in markers or node_id in [17007, 18950]:
                continue

            # If a label is specified, skip nodes that do not match the label
            if label is not None and self.gflim_data.node_labels[node_id] != label:
                continue

            # Calculate the distance to all existing markers
            dist = np.mean([np.linalg.norm(feat_matrix[node_id] - feat_matrix[m]) for m in markers])
            distances.append(dist)
            inhomogeneity_scores.append(self.node_homogeneity(node_id, num_feats, num_neighbors))
            nodes.append(node_id)

        # normalize the inhomogeneity scores with min-max scaling
        if len(inhomogeneity_scores) > 0:
            min_val = min(inhomogeneity_scores)
            max_val = max(inhomogeneity_scores)
            if max_val - min_val > 0:
                inhomogeneity_scores = [(inhomog - min_val) / (max_val - min_val) for inhomog in inhomogeneity_scores]
            else:
                inhomogeneity_scores = [0.0 for _ in inhomogeneity_scores]
        
        # normalize the distances with min-max scaling
        if len(distances) > 0:
            min_val = min(distances)
            max_val = max(distances)
            if max_val - min_val > 0:
                distances = [(dist - min_val) / (max_val - min_val) for dist in distances]
            else:
                distances = [0.0 for _ in distances]

        # Combine distances and inhomogeneity scores
        combined_scores = []
        for dist, node, inhomog in zip(distances, nodes, inhomogeneity_scores):
            combined_scores.append((dist - inhomog, node))

        if not combined_scores:
            raise ValueError("No valid nodes left to select.")
        
        # Select the most distant node
        return self.select_node(markers, combined_scores, max, avoid_neighbors)

    def write_img_markers(self):
        """
        Writes the selected markers to a file.
        """
        if not os.path.exists(self.markers_path):
            os.makedirs(self.markers_path)

        # extract pixels from the markers
        markers_pixels = []
        for (marker, label) in self.markers:
            pixels = np.where(self.gflim_data.superpixel == marker + 1)
            if len(pixels[0]) > 0:
                # append all the pixels of the marker
                for x, y in zip(pixels[1], pixels[0]):
                    markers_pixels.append((label, x, y))
            else:
                raise ValueError(f"Marker {marker} not found in the superpixel image.")

        width, height = self.gflim_data.superpixel.shape[1], self.gflim_data.superpixel.shape[0]
        markers_file = os.path.join(self.markers_path, f'{self.img_name}-seeds.txt')
        with open(markers_file, 'w') as f:

            # write nsuperpixels width and height
            f.write(f"{len(markers_pixels)} {width} {height}\n")

            # write x, y coordinates of the markers
            for label, x, y in markers_pixels:
                f.write(f"{x} {y} -1 {1 if label == 255 else 0} 0\n")

        print(f"Markers saved to {markers_file}")

    def write_graph_markers(self):
        """
        Writes the selected markers to a file.
        """
        path = os.path.join(self.model.model_dir, 'markers')
        if not os.path.exists(path):
            os.makedirs(path)

        # extract pixels from the markers
        markers_pixels = []
        i = 0
        for (marker, label) in self.markers:
            pixels = np.where(self.gflim_data.superpixel == marker + 1)
            if len(pixels[0]) > 0:
                # append all the pixels of the marker
                markers_pixels.append((marker, i, 1 if label == 255 else 0))
                i += 1
            else:
                raise ValueError(f"Marker {marker} not found in the superpixel image.")

        markers_file = os.path.join(path, f'{self.img_name}-seeds_graph.txt')
        with open(markers_file, 'w') as f:
            f.write(f"{len(markers_pixels)}\n")
            for marker, i, label in markers_pixels:
                f.write(f"{marker} {i} {label}\n")

        print(f"Markers saved to {markers_file}")
    
    def read_graph_markers(self, markers_path):
        """
        Reads the markers from the file.
        """
        markers_file = os.path.join(markers_path, f'{self.img_name}-seeds_graph.txt')
        if not os.path.exists(markers_file):
            raise FileNotFoundError(f"Markers file {markers_file} does not exist.")

        with open(markers_file, 'r') as f:
            lines = f.readlines()
        
        # first line contains the number of markers, width and height
        nmarkers = int(lines[0].strip().split()[0])
        self.markers = []
        
        # subsequent lines contain the x, y coordinates of the markers
        for line in lines[1:nmarkers + 1]:
            spx, _, label = map(int, line.strip().split())
            if (spx, label) not in self.markers:
                self.markers.append((spx, label*255))
            else:
                print(f"Marker {spx} already exists, skipping.")

    def move_results(self, dst_path : str):
        """
        Moves the saliency maps and results folders to match the current iteration.
        """

        # Copy the model folder
        src_folder = os.path.join(self.model.model_dir, 'model')
        if os.path.exists(src_folder):
            dst_folder = os.path.join(dst_path, 'model')
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
            shutil.rmtree(src_folder, ignore_errors=True)

        # copy graph markers
        src_folder = os.path.join(self.model.model_dir, 'markers')
        if os.path.exists(src_folder):
            dst_folder = os.path.join(dst_path, 'markers')
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
            shutil.rmtree(src_folder, ignore_errors=True)

        # copy image markers
        src_folder = os.path.join(self.model.markers_dir)
        if os.path.exists(src_folder):
            dst_folder = os.path.join(dst_path, 'markers_img')
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

        # copy saliency maps and activations
        nlayers = self.model.arch['nlayers']
        for layer in range(1, nlayers + 1):
            src_folder = os.path.join(self.model.model_dir, f'activationNoMerge{layer}')
            if os.path.exists(src_folder):
                dst_folder = os.path.join(dst_path, f'activationNoMerge{layer}')
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                shutil.rmtree(src_folder, ignore_errors=True)
            
            src_folder = os.path.join(self.model.model_dir, f'activation{layer}')
            if os.path.exists(src_folder):
                dst_folder = os.path.join(dst_path, f'activation{layer}')
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                shutil.rmtree(src_folder, ignore_errors=True)
            
            for d in ["graph-adaptive", "hybrid"]:
                src_folder = os.path.join(self.model.model_dir, f'{d}_{layer}')
                if os.path.exists(src_folder):
                    dst_folder = os.path.join(dst_path, f'{d}_{layer}')
                    shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
                    shutil.rmtree(src_folder, ignore_errors=True)

    def evaluate(self, layer : int, save_path : str, img_folder : str, labels_folder : str):
        for d in ["graph-adaptive", "hybrid"]:
            save_file = f'{save_path}_{d}_{layer}.pkl'
            saliency_dir = f"{self.model.model_dir}/{d}_{layer}/label2"

            metricas = metrics.FLIMMetrics(use_smeasure=True)
            metricas.evaluate_saliency_results(saliency_dir, labels_folder, file_list=[file.split('.')[0] for file in os.listdir(img_folder)])
            metricas.evaluate_detection_results(saliency_dir, labels_folder, file_list=[file.split('.')[0] for file in os.listdir(img_folder)])

            print('Saving metrics to file:', save_file)
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, 'wb') as file:
                pickle.dump(metricas, file)

    def iteration(self, iteration : int, train : bool):
        print(f"Selected markers: {self.markers}")
        save_path = os.path.join(self.save_path, f'iteration_{iteration}')
        self.write_img_markers()
        self.write_graph_markers()
        
        if train:
            self.model.run_model(self.train_img_folder, 'train', save_intermediate=False, layers_activations=[1,2,3], 
                            layers_decode=[1,2,3], decoders=['hybrid', 'graph-adaptive'], batchsize=64,
                            graphs_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/graphs',
                            spx_label_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/labels')
            self.model.run_model(self.val_img_folder, 'extract', save_intermediate=False, layers_activations=[1,2,3], 
                            layers_decode=[1,2,3], decoders=['hybrid', 'graph-adaptive'], batchsize=64,
                            graphs_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/graphs',
                            spx_label_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/labels')
            print(f"Moving results for iteration {iteration}...")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for layer in range(1, self.model.arch['nlayers'] + 1):
                self.evaluate(layer, os.path.join(save_path,'train'), os.path.join(self.original_image_path, 'train'), os.path.join(self.labels_path, 'train'))
                self.evaluate(layer, os.path.join(save_path,'val'), os.path.join(self.original_image_path, 'val'), os.path.join(self.labels_path, 'val'))
        
        self.move_results(save_path)
        

    def iterative_marker_selection_byDistance(self):
        """
        Iteratively selects one marker per iteration until the maximum number of markers is reached.
        """

        if self.initial_iteration > 0:
            print(f"Loading markers from iteration {self.initial_iteration}...")
            path = os.path.join(self.save_path, f'iteration_{self.initial_iteration}', 'markers')
            self.read_graph_markers(path)
            print(f"Markers loaded: {self.markers}")
            iteration = self.initial_iteration
        else:
            iteration = 1

        print(f"Iteration {iteration}: Selecting markers...")
        for label in np.unique(self.gflim_data.node_labels):
            self.markers.append(self.select_random_marker(markers=[m for (m,l) in self.markers], label=label, avoid_neighbors=False))
        self.iteration(iteration)

        while len(self.markers) < self.max_markers:
            iteration += 1
            print(f"Iteration {iteration}: Selecting markers...")
            self.markers.append(self.select_marker_euclDist(markers=[m for (m,l) in self.markers], avoid_neighbors=False))
            self.iteration(iteration)

        print("Marker selection completed.")

    def iterative_marker_selection_pairs(self, train_img_folder, val_img_folder, train=True):
        """
        Iteratively selects pairs of markers until the maximum number of markers is reached.
        Also avoids selecting neighbors of existing markers.
        """
        self.train_img_folder = train_img_folder
        self.val_img_folder = val_img_folder

        if self.initial_iteration > 0:
            print(f"Loading markers from iteration {self.initial_iteration}...")
            path = os.path.join(self.save_path, f'iteration_{self.initial_iteration}', 'markers')
            self.read_graph_markers(path)
            print(f"Markers loaded: {self.markers}")
            iteration = self.initial_iteration
        else:
            iteration = 1

        print(f"Iteration {iteration}: Selecting markers...")
        for label in np.unique(self.gflim_data.node_labels):
            self.markers.append((self.select_random_marker(markers=[m for (m,l) in self.markers], label=label, avoid_neighbors=True), label))
        self.iteration(iteration, train)

        while len(self.markers) < self.max_markers:
            iteration += 1
            print(f"Iteration {iteration}: Selecting markers...")
            for label in np.unique(self.gflim_data.node_labels):
                self.markers.append((self.select_marker_euclDist(markers=[m for (m,l) in self.markers], label=label, avoid_neighbors=True), label))
            self.iteration(iteration, train)

        print("Marker selection completed.")

    def iterative_marker_selection(self, train_img_folder, val_img_folder, train=True):
        """
        Iteratively selects pairs of markers until the maximum number of markers is reached.
        Also avoids selecting neighbors of existing markers.
        """
        self.train_img_folder = train_img_folder
        self.val_img_folder = val_img_folder

        # If initial_iteration is -1, get the last iteration
        if self.initial_iteration == -1:
            it = 0
            path = os.path.join(self.save_path, f'iteration_{it+1}')
            while( os.path.exists(path)):
                it += 1
                path = os.path.join(self.save_path, f'iteration_{it+1}')
            print(f"Last iteration found: {it}.")
            self.initial_iteration = it if it > 0 else 0

        if self.initial_iteration > 0:
            print(f"Loading markers from iteration {self.initial_iteration}...")
            path = os.path.join(self.save_path, f'iteration_{self.initial_iteration}', 'markers')
            self.read_graph_markers(path)
            if len(self.markers) == 0:
                raise ValueError("No markers found.")
            else:
                print(f"Markers loaded: {self.markers}")
                iteration = self.initial_iteration
        else:
            iteration = 1
            print(f"Iteration {iteration}: Selecting markers...")
            for label in np.unique(self.gflim_data.node_labels):
                self.markers.append((self.select_most_homogeneous_marker([m for (m,l) in self.markers if l == label], label=label, avoid_neighbors=True), label))
            
            save_path = os.path.join(self.save_path, f'iteration_{iteration}', 'markers')
            #self.read_graph_markers(save_path)
            self.iteration(iteration, train)

        while len(self.markers) < self.max_markers:
            iteration += 1
            print(f"Iteration {iteration}: Selecting markers...")
            for label in np.unique(self.gflim_data.node_labels):
                self.markers.append((self.select_marker_euclDistTSNE_std([m for (m,l) in self.markers if l == label], label=label, avoid_neighbors=True), label))
            
            save_path = os.path.join(self.save_path, f'iteration_{iteration}', 'markers')
            #self.read_graph_markers(save_path)
            self.iteration(iteration, train)

        print("Marker selection completed.")


if __name__ == "__main__":

    DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/cells_dataset'
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = 'DISF'
    INIT_SEEDS = 50000
    FINAL_SPX = 20000
    GRAPH_FEATS = 'SUM'
    ARCH_NAME = 'arch_cells4'
    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'automatic_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False
    SPLIT_ID = 5

    model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                            STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                            ATTENTION, AGGREGATION, DROPOUT, INIT_SEEDS)
    
    print('Creating model ..')
    model = GFLIMModel(model_dir, f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                    f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                    SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                    STRIDE, ATTENTION, AGGREGATION, DROPOUT)
    
    print('Getting gflim data ..')
    img_name = '6'
    gflim_data = GFLIMData(
        image_path=os.path.join(DATASET_PATH, 'imgs', 'orig', f'{img_name}.png'),
        label_path=os.path.join(DATASET_PATH, 'imgs', 'label', f'{img_name}.png'),
        graph_path=os.path.join(DATASET_PATH, 'superpixels', SPX_ALG, f'{INIT_SEEDS}-{FINAL_SPX}', GRAPH_FEATS, 'graphs', f'{img_name}.json'),
        superpixel_path=os.path.join(DATASET_PATH, 'superpixels', SPX_ALG, f'{INIT_SEEDS}-{FINAL_SPX}', GRAPH_FEATS, 'labels', f'{img_name}.pgm'),
        num_neighbors=model.arch['layer1']['conv']['kernel_size']
    )
    
    print('GFLIM data loaded.')
    markers_selection = MarkersSelection(
        gflim_data=gflim_data,
        model=model,
        markers_path=f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}',
        img_name=img_name,
        original_image_path=os.path.join(DATASET_PATH, 'automatic_splits', f'split{SPLIT_ID}', 'orig'),
        labels_path=os.path.join(DATASET_PATH, 'automatic_splits', f'split{SPLIT_ID}', 'label'),
        save_path=os.path.join(model.model_dir, 'results_semi_automatic'),
        max_markers=60,
        initial_iteration=0, tsne=True
    )

    train_img_folder = os.path.join(DATASET_PATH, 'automatic_splits', f'split{SPLIT_ID}', 'orig', 'train')
    val_img_folder = os.path.join(DATASET_PATH, 'automatic_splits', f'split{SPLIT_ID}', 'orig', 'val')

    markers_selection.iterative_marker_selection(train_img_folder, val_img_folder, train=True)

    """
    STRIDE = 'none'
    ATTENTION = 'CLASS_DOMINANCE'
    DROPOUT = True
    inputs += [(arch_id,split_id, STRIDE, ATTENTION, DROPOUT) for arch_id in [1] for split_id in range(1,6)]
    """

    #with multiprocessing.Pool() as pool:
    #    pool.starmap(run_model_parallel, ((arch_id, split_id, graphs_path_src, STRIDE, ATTENTION, DROPOUT) for (arch_id,split_id, STRIDE, ATTENTION, DROPOUT) in tqdm(inputs)))
