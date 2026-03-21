import numpy as np

"""Add sinusoid features to the features set"""


def generate_sinusoids(dataset, sinusoid_degree):
    """Extends data set with sinusoid features.

    Returns a new feature array with more features, comprising of
    sin(x).

    :param dataset: data set.
    :param sinusoid_degree: multiplier for sinusoid parameter multiplications
    """

    # Create sinusoids matrix.
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    # Generate sinusoid features of specified degree.
    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    # Return generated sinusoidal features.
    return sinusoids

"""Normalize features"""


def normalize(features):
    """Normalize features.

    Normalizes input features X. Returns a normalized version of X where the mean value of
    each feature is 0 and deviation is close to 1.

    :param features: set of features.
    :return: normalized set of features.
    """

    # Copy original array to prevent it from changes.
    features_normalized = np.copy(features).astype(float)

    # Get average values for each feature (column) in X.
    features_mean = np.mean(features, 0)

    # Calculate the standard deviation for each feature.
    features_deviation = np.std(features, 0)

    # Subtract mean values from each feature (column) of every example (row)
    # to make all features be spread around zero.
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # Normalize each feature values so that all features are close to [-1:1] boundaries.
    # Also prevent division by zero error.
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation


"""Add polynomial features to the features set"""


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """Extends data set with polynomial features of certain degree.

    Returns a new feature array with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.

    :param dataset: dataset that we want to generate polynomials for.
    :param polynomial_degree: the max power of new features.
    :param normalize_data: flag that indicates whether polynomials need to normalized or not.
    """

    # Split features on two halves.
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    # Extract sets parameters.
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # Check if two sets have equal amount of rows.
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    # Check if at list one set has features.
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    # Replace empty set with non-empty one.
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # Make sure that sets have the same number of features in order to be able to multiply them.
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # Create polynomials matrix.
    polynomials = np.empty((num_examples_1, 0))

    # Generate polynomial features of specified degree.
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # Normalize polynomials if needed.
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    # Return generated polynomial features.
    return polynomials

"""Prepares the dataset for training"""


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """Prepares data set for training on prediction"""

    # Calculate the number of examples.
    num_examples = data.shape[0]

    # Prevent original data from being modified.
    data_processed = np.copy(data)

    # Normalize data set.
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        # Replace processed data with normalized processed data.
        # We need to have normalized data below while we will adding polynomials and sinusoids.
        data_processed = data_normalized

    # Add sinusoidal features to the dataset.
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # Add polynomial features to data set.
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # Add a column of ones to X.
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation

"""Linear Regression Module"""


class LinearRegression:
    # pylint: disable=too-many-instance-attributes
    """Linear Regression Class"""

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # pylint: disable=too-many-arguments
        """Linear regression constructor.

        :param data: training set.
        :param labels: training set outputs (correct values).
        :param polynomial_degree: degree of additional polynomial features.
        :param sinusoid_degree: multipliers for sinusoidal features.
        :param normalize_data: flag that indicates that features should be normalized.
        """

        # Normalize features and add ones column.
        (
            data_processed,
            features_mean,
            features_deviation
        ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # Initialize model parameters.
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, lambda_param=0, num_iterations=500):
        """Trains linear regression.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        :param num_iterations: number of gradient descent iterations.
        """

        # Run gradient descent.
        cost_history = self.gradient_descent(alpha, lambda_param, num_iterations)

        return self.theta, cost_history

    def gradient_descent(self, alpha, lambda_param, num_iterations):
        """Gradient descent.

        It calculates what steps (deltas) should be taken for each theta parameter in
        order to minimize the cost function.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        :param num_iterations: number of gradient descent iterations.
        """

        # Initialize J_history with zeros.
        cost_history = []

        for _ in range(num_iterations):
            # Perform a single gradient step on the parameter vector theta.
            self.gradient_step(alpha, lambda_param)

            # Save the cost J in every iteration.
            cost_history.append(self.cost_function(self.data, self.labels, lambda_param))

        return cost_history

    def gradient_step(self, alpha, lambda_param):
        """Gradient step.

        Function performs one step of gradient descent for theta parameters.

        :param alpha: learning rate (the size of the step for gradient descent)
        :param lambda_param: regularization parameter
        """

        # Calculate the number of training examples.
        num_examples = self.data.shape[0]

        # Predictions of hypothesis on all m examples.
        predictions = LinearRegression.hypothesis(self.data, self.theta)

        # The difference between predictions and actual values for all m examples.
        delta = predictions - self.labels

        # Calculate regularization parameter.
        reg_param = 1 - alpha * lambda_param / num_examples

        # Create theta shortcut.
        theta = self.theta

        # Vectorized version of gradient descent.
        theta = theta * reg_param - alpha * (1 / num_examples) * (delta.T @ self.data).T
        # We should NOT regularize the parameter theta_zero.
        theta[0] = theta[0] - alpha * (1 / num_examples) * (self.data[:, 0].T @ delta).T

        self.theta = theta

    def get_cost(self, data, labels, lambda_param):
        """Get the cost value for specific data set.

        :param data: the set of training or test data.
        :param labels: training set outputs (correct values).
        :param lambda_param: regularization parameter
        """

        data_processed = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

        return self.cost_function(data_processed, labels, lambda_param)

    def cost_function(self, data, labels, lambda_param):
        """Cost function.

        It shows how accurate our model is based on current model parameters.

        :param data: the set of training or test data.
        :param labels: training set outputs (correct values).
        :param lambda_param: regularization parameter
        """

        # Calculate the number of training examples and features.
        num_examples = data.shape[0]

        # Get the difference between predictions and correct output values.
        delta = LinearRegression.hypothesis(data, self.theta) - labels

        # Calculate regularization parameter.
        # Remember that we should not regularize the parameter theta_zero.
        theta_cut = self.theta[1:, 0]
        reg_param = lambda_param * (theta_cut.T @ theta_cut)

        # Calculate current predictions cost.
        cost = (1 / 2 * num_examples) * (delta.T @ delta + reg_param)

        # Let's extract cost value from the one and only cost numpy matrix cell.
        return cost[0][0]

    def predict(self, data):
        """Predict the output for data_set input based on trained theta values

        :param data: training set of features.
        """

        # Normalize features and add ones column.
        data_processed = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

        # Do predictions using model hypothesis.
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions

    @staticmethod
    def hypothesis(data, theta):
        """Hypothesis function.

        It predicts the output values y based on the input values X and model parameters.

        :param data: data set for what the predictions will be calculated.
        :param theta: model params.
        :return: predictions made by model based on provided theta.
        """

        predictions = data @ theta

        return predictions
