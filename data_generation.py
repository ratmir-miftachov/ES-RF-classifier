import numpy as np
import matplotlib.colors as mcolors
from sklearn.datasets import make_blobs


####################################################################################################
###################################### 2 Dimensional Case ##########################################
####################################################################################################


# Function to generate meshgrid for 2D X
# Functions to generate respective y to X
# Function that aggregates all the above functions to generate data for classification
def generate_two_dim_X_meshgrid(n_ticks_per_ax_meshgrid=128):
    """Generate X for 2D classification,
    n_train_samples = (n_ticks/2)^2,
    n_test_samples = (n_ticks/2)^2,
    n_samples = n_ticks^2"""
    # n_ticks_per_ax_meshgrid Number of points in each dimension for meshgrid
    X1, X2 = np.linspace(0, 1, n_ticks_per_ax_meshgrid), np.linspace(
        0, 1, n_ticks_per_ax_meshgrid
    )

    X1_train, X2_train = X1[::2], X2[::2]  # 64 for train and test each
    X1_train, X2_train = np.meshgrid(X1_train, X2_train)  # 64 x 64
    X_train = np.c_[
        X1_train.ravel(), X2_train.ravel()
    ]  # 64 ** 2 = 4096, 2 ist der shape

    X1_test, X2_test = X1[1::2], X2[1::2]  # same shape as above
    X1_test, X2_test = np.meshgrid(X1_test, X2_test)
    X_test = np.c_[X1_test.ravel(), X2_test.ravel()]

    return X_train, X_test


def generate_rectangular_classification(X, p):
    """Draw X.shape[0] Y from a Bernoulli distribution with probability p if X is inside the rectangle and probability 0.2 else.

    Return: y = Bernoullie draws,
            f = p = f(X) = true probabilities of resp. DGP/Ber"""
    # X equidistant, rectangular in middle:
    X_in_rectangular = (
        (1 / 3 <= X[:, 0])
        * (X[:, 0] <= 2 / 3)
        * (1 / 3 <= X[:, 1])
        * (X[:, 1] <= 2 / 3)
    )
    f = 0.2 + X_in_rectangular.astype(int) * (p - 0.2)
    y = np.random.binomial(1, f, X.shape[0])
    return y, f


def generate_circular_classification(X, p):
    """Draw X.shape[0] Y from a Bernoulli distribution with probability p if X is inside the circle and probability 0.2 else.

    Return: y = Bernoullie draws,
            f = p = f(X) = true probabilities of resp. DGP/Ber"""
    X_in_circular = np.sqrt((X[:, 0] - 1 / 2) ** 2 + (X[:, 1] - 1 / 2) ** 2) <= 1 / 4
    f = 0.2 + X_in_circular.astype(int) * (p - 0.2)
    y = np.random.binomial(1, f, X.shape[0])
    return y, f


def generate_smooth_signal_classification(X):
    """Draw X.shape[0] Y from a Bernoulli distribution with changing probabilities depending on X in the circle or not with smooth transitions.

    Return: y = Bernoullie draws,
            f = p = f(X) = true probabilities of resp. DGP/Ber"""
    Z = (
        np.exp(-((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2) / 0.1) * 20
    )  # Original Z values
    # Normalize Z values between 0 and 1
    norm = mcolors.Normalize(vmin=0, vmax=np.max(Z))
    f = norm(Z)
    y = np.random.binomial(1, f, X.shape[0])
    return y, f


def generate_sine_cosine_classification(X):
    """Draw X.shape[0] Y from a Bernoulli distribution where the probability parameter p is determined by the square in which X is. The transition between probabilities is smooth.

    Return: y = Bernoullie draws,
            f = p = f(X) = true probabilities of resp. DGP/Ber"""
    gamma = 1.5  # Controls the sharpness of the transitions
    f = 1 / (
        1 + np.exp(-gamma * (np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])))
    )
    y = np.random.binomial(1, f, X.shape[0])
    return y, f


def generate_data_for_classification_from_two_dim_X(X, bernoulli_p, dgp_name):
    """Generate data for classification from 2-dim X"""
    if dgp_name == "rectangular":
        return generate_rectangular_classification(X=X, p=bernoulli_p)
    elif dgp_name == "circular":
        return generate_circular_classification(X=X, p=bernoulli_p)
    elif dgp_name == "smooth_signal":
        return generate_smooth_signal_classification(X=X)
    elif dgp_name == "sine_cosine":
        return generate_sine_cosine_classification(X=X)


####################################################################################################
###################################### Additive Models #############################################
####################################################################################################


def generate_X_hiabu_et_al(n_samples, feature_dim):
    # Step 1: Generate from a multivariate normal with mean 0 and covariance 0.3 for off-diagonals
    mean = np.zeros(feature_dim)
    cov = np.full(
        (feature_dim, feature_dim), 0.3
    )  # Create a d x d matrix with 0.3 everywhere
    np.fill_diagonal(cov, 1)  # Set diagonal to 1 for variance

    # Generate n_samples x d matrix of predictors ~ N(0, cov)
    X_tilde = np.random.multivariate_normal(mean, cov, n_samples)

    # Step 2: Apply transformation
    X = (2.5 / np.pi) * np.arctan(X_tilde)

    return X


# Additive Model I generates X and y given n_samples & feature_dim
# Other Additive Models generate y to X, and share same X Process
# Function that aggregates all the above functions to generate data for classification
def additive_model_I(X):
    # Define component functions for centering (without a1, a2, a3, a4 for now)
    def g1(x):
        return x

    def g2(x):
        return (2 * x - 1) ** 2

    def g3(x):
        return (np.sin(2 * np.pi * x)) / (2 - np.sin(2 * np.pi * x))

    def g4(x):
        return (
            1 / 10 * np.sin(2 * np.pi * x)
            + 2 / 10 * np.cos(2 * np.pi * x)
            + 3 / 10 * np.sin(2 * np.pi * x) ** 2
            + 4 / 10 * np.cos(2 * np.pi * x) ** 3
            + 5 / 10 * np.sin(2 * np.pi * x) ** 3
        )

    # Compute constants for centering (mean values of g1, g2, g3, g4 over uniform samples)
    a1 = np.mean([g1(x) for x in np.random.uniform(0, 1, 1000)])
    a2 = np.mean([g2(x) for x in np.random.uniform(0, 1, 1000)])
    a3 = np.mean([g3(x) for x in np.random.uniform(0, 1, 1000)])
    a4 = np.mean([g4(x) for x in np.random.uniform(0, 1, 1000)])

    # Define component functions again with centering using a1, a2, a3, a4
    def g1(x):
        return x - a1

    def g2(x):
        return (2 * x - 1) ** 2 - a2

    def g3(x):
        return (np.sin(2 * np.pi * x)) / (2 - np.sin(2 * np.pi * x)) - a3

    def g4(x):
        return (
            1 / 10 * np.sin(2 * np.pi * x)
            + 2 / 10 * np.cos(2 * np.pi * x)
            + 3 / 10 * np.sin(2 * np.pi * x) ** 2
            + 4 / 10 * np.cos(2 * np.pi * x) ** 3
            + 5 / 10 * np.sin(2 * np.pi * x) ** 3
            - a4
        )

    # Vectorized versions of f_j for the first 4 predictors
    f1 = lambda X: 5 * g1(X[:, 0])  # Apply to the first column (X1)
    f2 = lambda X: 3 * g2(X[:, 1])  # Apply to the second column (X2)
    f3 = lambda X: 4 * g3(X[:, 2])  # Apply to the third column (X3)
    f4 = lambda X: 6 * g4(X[:, 3])  # Apply to the fourth column (X4)

    # Function to generate p(x) for the whole dataset (vectorized)
    def p_x(X):
        return 1 / (1 + np.exp(-(f1(X) + f2(X) + f3(X) + f4(X))))

    # Calculate p(x) for all samples (vectorized)
    f = p_x(X)
    Y = np.random.binomial(1, f, X.shape[0])
    return Y, f


def additive_model_sparse_smooth(X):
    # Define m1(x1) and m2(x2)
    m1_x1 = -2 * np.sin(np.pi * X[:, 0])
    m2_x2 = 2 * np.sin(np.pi * X[:, 1])

    f = 1 / (1 + np.exp(-(m1_x1 + m2_x2)))
    Y = np.random.binomial(1, f, X.shape[0])
    return Y, f


def additive_model_sparse_jump(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    # Define m1(x1) and m2(x2) using piecewise conditions
    m1_x1 = np.where(
        x1 >= 0, 2 * (-1) ** 1 * np.sin(np.pi * x1) - 2, -2 * np.sin(np.pi * x1) + 2
    )
    m2_x2 = np.where(
        x2 >= 0, 2 * (-1) ** 2 * np.sin(np.pi * x2) - 2, -2 * np.sin(np.pi * x2) + 2
    )

    f = 1 / (1 + np.exp(-(m1_x1 + m2_x2)))
    Y = np.random.binomial(1, f, X.shape[0])
    return Y, f


def hierarchical_interaction_sparse_jump(X):
    def m_k(x, k):
        """
        Compute m_k(x_k) based on the formula:
        m_k(x) = (-1)**k * 2 * np.sin(np.pi * x)
        """
        return (-1) ** k * 2 * np.sin(np.pi * x)

    def m_kj(x_k, x_j, k):
        """
        Compute m_kj(x_k, x_j) based on the formula:
        m_kj(x_k, x_j) = 2 * (-1)**k * np.sin(np.pi * x_k * x_j)
        """
        return 2 * (-1) ** k * np.sin(np.pi * x_k * x_j)

    # Sum of m_k(x_k) for k=1, 2, 3
    m_individual = sum(m_k(X[:, k - 1], k) for k in range(1, 4))
    # Sum of m_kj(x_k, x_j) for 1 <= k < j <= 3
    m_interactions = (
        m_kj(X[:, 0], X[:, 1], k=1)
        + m_kj(X[:, 0], X[:, 2], k=1)
        + m_kj(X[:, 1], X[:, 2], k=2)
    )

    f = 1 / (1 + np.exp(-(m_individual + m_interactions)))
    Y = np.random.binomial(1, f, X.shape[0])
    return Y, f


def generate_y_f_classification_additive_models(X, dgp_name):
    if dgp_name == "additive_model_I":
        Y, f = additive_model_I(X=X)
    elif dgp_name == "additive_sparse_smooth":
        Y, f = additive_model_sparse_smooth(X=X)
    elif dgp_name == "additive_sparse_jump":
        Y, f = additive_model_sparse_jump(X=X)
    elif dgp_name == "hierarchical-interaction_sparse_jump":
        Y, f = hierarchical_interaction_sparse_jump(X=X)
    else:
        raise ValueError(f"Unknown model type: {dgp_name}")

    return Y, f


def generate_X_y_f_classification(
    dgp_name,  
    bernoulli_p,  
    n_samples,  # size of dateset (train + test) 
    feature_dim,  # for additive models 
    random_state,  # random seed 
    n_ticks_per_ax_meshgrid=None,  # for 2 dim case 
):
    np.random.seed(random_state)
    if dgp_name in ["circular", "smooth_signal", "rectangular", "sine_cosine"]:
        # Get X:
        if n_ticks_per_ax_meshgrid is not None:
            X_train, X_test = generate_two_dim_X_meshgrid(
                n_ticks_per_ax_meshgrid=n_ticks_per_ax_meshgrid
            )
        else:
            X = np.random.uniform(0, 1, (n_samples, 2))
            X_train, X_test = X[: int(0.5 * n_samples)], X[int(0.5 * n_samples) :]
        # Get Y & true signal for X
        y_train, f_train = generate_data_for_classification_from_two_dim_X(
            X=X_train, bernoulli_p=bernoulli_p, dgp_name=dgp_name
        )
        y_test, f_test = generate_data_for_classification_from_two_dim_X(
            X=X_test, bernoulli_p=bernoulli_p, dgp_name=dgp_name
        )
        # Add additional random uniform features for 2D cases
        if feature_dim > 2:
            X_train_additional = np.random.uniform(
                0, 1, (X_train.shape[0], feature_dim - 2)
            )  # 2 is number of signal features
            X_test_additional = np.random.uniform(
                0, 1, (X_test.shape[0], feature_dim - 2)
            )
            X_train = np.hstack((X_train, X_train_additional))
            X_test = np.hstack((X_test, X_test_additional))

    elif dgp_name in [
        "additive_model_I",
        "additive_sparse_smooth",
        "additive_sparse_jump",
        "hierarchical-interaction_sparse_jump",
    ]:
        if dgp_name == "additive_model_I":
            X = np.random.uniform(0, 1, (n_samples, feature_dim))
        else:
            X = generate_X_hiabu_et_al(n_samples=n_samples, feature_dim=feature_dim)
        y, f = generate_y_f_classification_additive_models(X, dgp_name=dgp_name)
        X_train, X_test = X[: int(0.5 * n_samples)], X[int(0.5 * n_samples) :]
        y_train, y_test = y[: int(0.5 * n_samples)], y[int(0.5 * n_samples) :]
        f_train, f_test = f[: int(0.5 * n_samples)], f[int(0.5 * n_samples) :]
    else:
        raise ValueError(f"Unknown model type: {dgp_name}")
    return X_train, X_test, y_train, y_test, f_train, f_test


# Helper function to create a simple dataset
def create_test_sample_data(case: str = "one_X_cut"):
    if case == "one_X_cut":
        X = np.array([[2, 3], [3, 5], [6, 6], [8, 9], [9, 12], [10, 15], [11, 14]])
        y = np.array([0, 0, 0, 1, 1, 1, 1])
    elif case == "two_X_cuts":
        X = np.array([[2, 3], [3, 5], [6, 6], [8, 9], [9, 12], [10, 15], [11, 14]])
        y = np.array([0, 0, 0, 1, 1, 0, 0])
    elif case == "chess_simple":
        X = np.array(
            [
                [1, 1],
                [1, 2],
                [2, 1],
                [2, 2],
            ]
        )
        y = np.array([0, 1, 1, 0])
    elif case == "chess":
        X = np.array(
            [
                [1, 1],
                [1, 2],
                [2, 1],
                [2, 2],
                [3, 1],
                [3, 2],
                [4, 1],
                [4, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 3],
                [3, 4],
                [4, 3],
                [4, 4],
            ]
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    elif case == "rectangular_top_right":
        X = np.array(
            [
                [1, 1],
                [1, 2],
                [2, 1],
                [2, 2],
                [3, 1],
                [3, 2],
                [4, 1],
                [4, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 3],
                [3, 4],
                [4, 3],
                [4, 4],
            ]
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    elif case == "mixed_blobs":
        X, y = make_blobs(n_samples=2000, centers=2, cluster_std=4.5, random_state=7)
    return X, y