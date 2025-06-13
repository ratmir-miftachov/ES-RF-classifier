from typing import Tuple, Optional, List
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class Node:
    """
    A class representing a node in a decision tree.

    Attributes:
        split_threshold (float): Threshold for splitting on the feature.
        feature (int): Index of the feature used for splitting.
        left_child (Optional[Node]): Left child node after split.
        right_child (Optional[Node]): Right child node after split.
        is_terminal (bool): Indicator if the node is a leaf.
        n_node_samples (int): Number of samples in the node.
        node_prediction (float): Prediction value for the node.
        data_indices (List[int]): Indices of data points in this node.
    """

    def __init__(self) -> None:
        self.split_threshold: Optional[float] = None
        self.feature: Optional[int] = None
        self.left_child: Optional[Node] = None
        self.right_child: Optional[Node] = None
        self.is_terminal: bool = False
        self.n_node_samples: int = 0
        self.node_prediction: Optional[float] = (
            None  # rn, only leaf nodes have a prediction
        )
        self.data_indices: List[int] = (
            []
        )  # Store indices rather than data to avoid copies

    def set_params(self, split_threshold: float, feature: int) -> None:
        """
        Set the split threshold and feature for this node.
        """
        self.split_threshold = split_threshold
        self.feature = feature


class DecisionTreeLevelWise:
    """
    A decision tree class that grows in a level-wise (breadth-first search) manner.
    Includes early stopping based on mean squared error (MSE) at each level.
    """

    def __init__(
        self,
        max_depth: int = np.inf,
        min_samples_split: int = 2,
        kappa: float = np.nan,
        max_features: str = "all",
        random_state: Optional[int] = None,
        es_offset: int = 0,
        rf_train_mse: bool = False,
    ) -> None:

        if max_depth is None:
            self.max_depth = np.inf
        elif max_depth < 1:
            raise ValueError("max_depth must be an integer >= 1")
        else:
            self.max_depth = max_depth
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        else:
            self.min_samples_split = min_samples_split
        if kappa is None:
            self.kappa = (
                np.nan
            )  # comparison to np.nan is always False, so ES is never applied
        else:
            self.kappa = kappa
        if max_features is None:
            self.max_features = "all"
        elif isinstance(max_features, int):
        
            self.max_features = max_features
        elif max_features not in ["all", "sqrt"]:
            raise ValueError("max_features must be 'all', 'sqrt', or a positive integer")
        else:
            self.max_features = max_features
        # Initialize the root node
        self.root: Node = Node()
        self.leaf_count = (
            0  # Initialize leaf count; if fit starts --> sets directly to 1 (root)
        )
        self.max_depth_reached = -1  # Initialize max depth reached to 0 (root)

        self.random_state = random_state
        # At the beginning of fit or any method using randomness:
        self._random = np.random.RandomState(self.random_state)

        self.es_offset = es_offset
        self.rf_train_mse = rf_train_mse

    def _calculate_mse(self, y: np.ndarray) -> float:
        """
        Calculate mean squared error (MSE) for the given response values.
        """
        mean_value = np.mean(y)
        return np.mean((y - mean_value) ** 2)

    def _gini(self, y: np.array) -> float:
        """
        Private function to define the Gini Coefficient

        Input:
            D -> data to compute the Gini Coefficient over
        Output:
            Gini Coefficient over D (Data from node)
        """
        p_hat = np.mean(y)
        # gini = 2 * p_hat * (1 - p_hat)
        gini = p_hat * (1 - p_hat)
        return gini

    def _node_prediction_value(self, y: np.ndarray) -> float:
        """
        Calculate the prediction value for a node based on the mean of y.
        """
        return np.mean(y)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, indices: List[int]
    ) -> Tuple[Optional[int], Optional[float], List[int], List[int]]:
        """
        Find the best split based on impurity reduction for the given indices.
        """
        best_impurity = float("inf")
        best_split = (None, None, [], [])

        n_features = X.shape[1]

        # This is how scikit-learn selects features when max_features < n_features
        if self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(n_features)))

            # scikit-learn's feature selection logic
            features = np.arange(n_features, dtype=np.intp)
            self._random.shuffle(features)
            features = features[:max_features]

            # Sort to match scikit-learn's feature processing order
            features.sort()
            
        elif isinstance(self.max_features, int):
            max_features = self.max_features

            # scikit-learn's feature selection logic
            features = np.arange(n_features, dtype=np.intp)
            self._random.shuffle(features)
            features = features[:max_features]

            # Sort to match scikit-learn's feature processing order
            features.sort()
            
        else:
            # max_features == "all"
            features = np.arange(n_features, dtype=np.intp)
            np.random.shuffle(features)

        for feature in features:
            # Extract just the rows relevant to 'indices'
            values = X[indices, feature]

            # Sort 'values' (and 'labels') once per feature
            sorted_idx_local = np.argsort(
                values
            )  # e.g. 122, 89, 0, 1, 13, 7, ... in range(len(values))
            sorted_values = values[
                sorted_idx_local
            ]  # e.g. 700, 600, 600, 400, 300, 200, ...

            # Get unique values and midpoints (candidate thresholds)
            unique_vals = np.unique(sorted_values)  # e.g. 700, 600, 400, 300, 200, ...
            if len(unique_vals) <= 1:
                # No valid split if all values are identical
                continue

            # Midpoints between consecutive unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            # Evaluate each threshold
            for threshold in thresholds:
                # 'pos' is the number of elements <= threshold
                pos = np.searchsorted(
                    sorted_values, threshold, side="right"
                )  # insert position of threshold in sorted_values

                # Map sorted local indices back to the original 'indices'
                left_indices = np.array(indices)[sorted_idx_local[:pos]]
                right_indices = np.array(indices)[sorted_idx_local[pos:]]

                # Skip if one side is empty
                if pos == 0 or pos == len(sorted_values):
                    continue

                # Compute impurity: weighted sum of left/right Gini
                left_impurity = self._gini(y[left_indices])
                right_impurity = self._gini(y[right_indices])
                impurity = (
                    left_impurity * len(left_indices)
                    + right_impurity * len(right_indices)
                ) / len(indices)

                # Update best split if this is better
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split = (feature, threshold, left_indices, right_indices)

        return best_split

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_whole_rf: np.ndarray = None,
        y_whole_rf: np.ndarray = None,
    ) -> None:
        """
        Fit the decision tree on the given data using a level-wise growth strategy with early stopping.
        Updates the leaf count and max depth reached as nodes are marked terminal.

        Parameters:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target values.
        """
        # Initialize Root
        self.root = Node()
        self.root.data_indices = list(range(len(y)))
        self.root.n_node_samples = len(y)
        self.root.node_prediction = self._node_prediction_value(y)
        self.leaf_count = 1  # Start with root as the first leaf
        self.max_depth_reached = -1  # Initial root depth is 0

        # Initialize the queue for breadth-first growth
        level_queue = deque([(self.root, 0)])  # Start with depth 0
        # Set initial level MSE

        if self.rf_train_mse:
            y_pred_whole_rf = self.predict_proba(X_whole_rf)[:, 1]
            level_es_score = np.mean((y_whole_rf - y_pred_whole_rf) ** 2)
        else:
            level_es_score = self._gini(y[self.root.data_indices])

        # Initialize a counter for consecutive levels meeting the stopping criterion
        consecutive_stop_levels = 0

        # Initialize the level-wise growth/BFS
        while level_queue:
            self.max_depth_reached += 1
            if level_es_score < self.kappa:  # compared to np.nan this is always False
                # Record the first level where stopping criterion is met if not already set
                consecutive_stop_levels += 1

                # Only break if we've seen enough consecutive stopping levels
                if consecutive_stop_levels > self.es_offset:
                    for node, _ in level_queue:
                        if not node.is_terminal:
                            node.is_terminal = True
                    break
            # Initialize new Round/Level
            current_level_size = len(level_queue)
            # Initialize new Round/Level
            level_es_score_sum = 0  # Only for first level, this is redundant
            # level_data_count = 0

            for _ in range(
                current_level_size
            ):  # Process all nodes at the current level
                node, depth = level_queue.popleft()
                indices = node.data_indices
                node.n_node_samples = len(indices)  # n_node_samples here

                # Check stopping conditions
                if (
                    depth >= self.max_depth
                    or len(indices) < self.min_samples_split
                    or self._gini(y[indices]) == 0
                ):
                    if not node.is_terminal:
                        node.is_terminal = True
                    continue

                # Find the best split for the current node (taking max_n_features into account)
                feature, threshold, left_indices, right_indices = self._best_split(
                    X=X, y=y, indices=indices
                )

                # If no valid split is found, mark as terminal
                if feature is None:
                    node.is_terminal = True
                    continue

                # Apply split and create children nodes
                # Apply split in current Node
                node.set_params(threshold, feature)  # split_threshold, feature
                # left_child, right_child, data_indices here
                node.left_child = Node()
                node.left_child.data_indices = left_indices
                node.left_child.node_prediction = self._node_prediction_value(
                    y[left_indices]
                )
                node.right_child = Node()
                node.right_child.data_indices = right_indices
                node.right_child.node_prediction = self._node_prediction_value(
                    y[right_indices]
                )
                # is_terminal, node_prediction during set_terminal

                # Update the leaf count every time a split is made
                self.leaf_count += 1

                # Add the new nodes for the next level
                level_queue.append((node.left_child, depth + 1))
                level_queue.append((node.right_child, depth + 1))
                # Calculate level MSE contributions
                level_es_score_sum += (
                    self._gini(y[left_indices]) * len(left_indices)
                ) + (self._gini(y[right_indices]) * len(right_indices))
                # level_data_count += len(indices)

            # Calculate mean squared error (MSE) for the entire level at the end
            if self.rf_train_mse:
                y_pred_whole_rf = self.predict_proba(X_whole_rf)[:, 1]
                level_es_score = np.mean((y_whole_rf - y_pred_whole_rf) ** 2)
            else:
                level_es_score = (
                    level_es_score_sum
                    / self.root.n_node_samples  # if level_data_count > 0 else 0
                )

    def _predict_single_proba(
        self,
        x: np.ndarray,
        node: Node,
        max_depth: Optional[int],
        current_depth: int = 1,
    ) -> float:
        """
        Predict the value for a single sample by traversing the tree with a depth limit.

        Parameters:
            x (np.ndarray): A single sample's feature values.
            node (Node): The current node in the decision tree.
            max_depth (Optional[int]): Maximum depth to traverse during prediction.
            current_depth (int): Current depth in the tree traversal.

        Returns:
            float: The predicted value from the appropriate node.
        """
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be an integer >= 1")
        # Check if we've reached a terminal node or the maximum allowed depth
        # Also check if node has no children (important during tree construction)
        if (
            node.is_terminal
            or node.left_child is None
            or node.right_child is None
            or (max_depth is not None and current_depth >= max_depth)
        ):
            return [1 - node.node_prediction, node.node_prediction]

        # Traverse to the appropriate child node
        if x[node.feature] <= node.split_threshold:
            return self._predict_single_proba(
                x, node.left_child, max_depth, current_depth + 1
            )
        else:
            return self._predict_single_proba(
                x, node.right_child, max_depth, current_depth + 1
            )

    def predict_proba(self, X: np.ndarray, depth: Optional[int] = None) -> np.ndarray:
        """
        Predict values for the given dataset, with an optional depth limit.

        Parameters:
            X (np.ndarray): Dataset to predict.
            depth (Optional[int]): Maximum depth to traverse during prediction.

        Returns:
            np.ndarray: Predicted values for each sample in X.
        """
        if len(X.shape) == 1:  # if only one sample
            X = X.reshape(1, -1)
        y_pred_prob = np.array(
            [self._predict_single_proba(x, self.root, max_depth=depth) for x in X]
        )
        # y_pred_labels = (y_pred_prob >= 0.5).astype(int)
        return y_pred_prob

    def predict(self, X: np.ndarray, depth: Optional[int] = None) -> np.ndarray:
        """
        Predict values for the given dataset, with an optional depth limit.

        Parameters:
            X (np.ndarray): Dataset to predict.
            depth (Optional[int]): Maximum depth to traverse during prediction.

        Returns:
            np.ndarray: Predicted values for each sample in X.
        """
        # if depth is not None and depth > self.max_depth_reached:
        #     depth = self.max_depth_reached
        y_pred_prob = self.predict_proba(X=X, depth=depth)
        y_pred_labels = (y_pred_prob[:, 1] >= 0.5).astype(int)
        return y_pred_labels

    def get_n_leaves(self) -> int:
        """
        Retrieve the current number of leaf nodes in the tree.

        Returns:
            int: Number of leaf nodes.
        """
        return self.leaf_count

    def get_depth(self) -> int:
        """
        Retrieve the maximum depth reached in the tree.

        Returns:
            int: Maximum depth reached during tree construction.
        """
        return self.max_depth_reached

    def get_bfs_attributes(self):
        queue = deque(
            [(self.root, 1)]
        )  # Initialize the queue with the root node and its depth (0)
        splits = {}
        while queue:
            current_node, depth = queue.popleft()  # Get the next node and its depth
            if depth not in splits:
                splits[depth] = []
            splits[depth].append((current_node.feature, current_node.split_threshold))
            # Add children to the queue with the next depth level if they exist
            if current_node.left_child:
                queue.append((current_node.left_child, depth + 1))
            if current_node.right_child:
                queue.append((current_node.right_child, depth + 1))
        return splits

    def plot_splits(self, X, y):
        # Check if X is 2D
        if X.shape[1] != 2:
            raise ValueError("X must have 2 features for plotting")

        splits = self.get_bfs_attributes()
        cumulative_splits = {}
        for depth in range(2, max(splits.keys()) + 2):
            cumulative_splits[depth] = []
            for d in range(1, depth):
                cumulative_splits[depth].extend(splits.get(d, []))

        fig, axs = plt.subplots(1, len(cumulative_splits), figsize=(20, 5))

        for depth, ax in enumerate(axs, start=0):
            ax.scatter(
                X[y == 0, 0],
                X[y == 0, 1],
                label="Class 0",
                marker="o",
                s=50,
                edgecolor="k",
            )
            ax.scatter(
                X[y == 1, 0],
                X[y == 1, 1],
                label="Class 1",
                marker="^",
                s=50,
                edgecolor="k",
            )
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.set_title(f"Depth {depth}")
            ax.set_aspect("equal")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

            # Initialize the bounds for each split
            x_bounds = [0, 5]
            y_bounds = [0, 5]

            for feature, threshold in cumulative_splits.get(depth, []):
                if feature == 0:  # Vertical split on x-axis
                    ax.vlines(
                        threshold + 0.5,
                        ymin=y_bounds[0],
                        ymax=y_bounds[1],
                        color="red",
                        linewidth=5,
                        linestyle="--",
                    )
                    # Update the x bounds for future splits within this region
                    if threshold + 0.5 > x_bounds[0] and threshold + 0.5 < x_bounds[1]:
                        x_bounds[1] = threshold + 0.5
                elif feature == 1:  # Horizontal split on y-axis
                    ax.hlines(
                        threshold + 0.5,
                        xmin=x_bounds[0],
                        xmax=x_bounds[1],
                        color="red",
                        linewidth=5,
                        linestyle="--",
                    )
                    # Update the y bounds for future splits within this region
                    if threshold + 0.5 > y_bounds[0] and threshold + 0.5 < y_bounds[1]:
                        y_bounds[1] = threshold + 0.5

        plt.tight_layout(w_pad=2, h_pad=2)
        plt.show()