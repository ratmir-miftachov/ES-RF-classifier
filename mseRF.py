from sklearn.ensemble import RandomForestClassifier
import noise_level_estimator as noise_est
import numpy as np


class mseRF(object):
    def __init__(
        self, n_estimators=100, random_state=None, vote_probability=True, es_offset=0
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.vote_probability = vote_probability
        if es_offset is None:
            self.es_offset = 0
        else:
            self.es_offset = es_offset
        self.estimators_ = []

    def fit(self, X_train, y_train, f_train=None):
        # Get critical value
        if f_train is not None:
            mean_estimated_train_noise = np.mean(f_train * (1 - f_train))
        else:
            noise_estimator = noise_est.Estimator(X_train, y_train)
            mean_estimated_train_noise = noise_estimator.estimate_1NN()

        # Add small tolerance for zero noise case
        tolerance = 9e-3  # You can adjust this value
        noise_threshold = max(mean_estimated_train_noise, tolerance)

        # Add maximum iteration limit
        max_iterations = 50  # You can adjust this value

        # Initialize the MSE
        mse = np.inf
        offset_counter = (
            -1
        )  # Counter for additional iterations after stopping criterion
        iteration = 1

        # Keep training until MSE criterion is met AND offset iterations are completed
        while (
            mse > noise_threshold or offset_counter < self.es_offset
        ) and iteration <= max_iterations:
            # Fit the RF
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=iteration,
                random_state=self.random_state,
            )
            rf.fit(X_train, y_train)
            if self.vote_probability:
                y_pred_proba_train = np.mean(
                    [tree.predict(X_train) for tree in rf.estimators_],
                    axis=0,
                )

                # Calculate the MSE
            else:
                y_pred_proba_train = rf.predict_proba(X_train)[
                    :, 1
                ]  # these two are the same!
                # y_pred_proba_train = np.mean(
                #     [tree.predict_proba(X_train)[:, 1] for tree in rf.estimators_],
                #     axis=0,
                # )
            mse = np.mean((y_pred_proba_train - y_train) ** 2)

            # Track offset iterations after stopping criterion is met
            if mse <= noise_threshold:
                offset_counter += 1
            iteration += 1

        return rf
