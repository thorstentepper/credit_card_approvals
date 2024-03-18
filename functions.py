# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def load_dataset(file_path):
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the dataset.

    Returns
    -------
    pandas.DataFrame
        The dataset loaded from the CSV file.
    """

    dataset = pd.read_csv(file_path, header=None)
    return dataset


def drop_features(dataset, columns_to_drop):
    """
    Drop specified columns from the dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset from which columns will be dropped.
    columns_to_drop : list
        The list of columns to be dropped from the dataset.

    Returns
    -------
    pandas.DataFrame
        The dataset after dropping the specified columns.
    """

    # Drop specified columns
    dataset = dataset.drop(columns_to_drop, axis=1)
    return dataset


def split_train_test(dataset, test_size=0.33, random_state=42):
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be split into train and test sets.
    test_size : float, optional
        The proportion of the dataset to include in the test split
        (default is 0.33).
    random_state : int, optional
        Controls the randomness of the dataset splitting (default is 42).

    Returns
    -------
    tuple of pandas.DataFrame
        The train and test sets.
    """

    # Split into train and test sets
    train_set, test_set = train_test_split(dataset, test_size=test_size,
                                           random_state=random_state)
    return train_set, test_set


def replace_question_marks_with_nan(dataset):
    """
    Replace '?' with NaN in the dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset in which '?' will be replaced with NaN.

    Returns
    -------
    pandas.DataFrame
        The dataset with '?' replaced with NaN.
    """

    # Replace '?' with NaN
    dataset = dataset.replace("?", np.NaN)
    return dataset


def mean_imputation(train_set, test_set):
    """
    Perform mean imputation to fill missing values in both train and test sets.

    Parameters
    ----------
    train_set : pandas.DataFrame
        The training dataset.
    test_set : pandas.DataFrame
        The test dataset.

    Returns
    -------
    tuple of pandas.DataFrame
        The train and test sets after mean imputation.
    """

    # Get only numeric columns for mean calculation
    numeric_cols = train_set.select_dtypes(include=np.number).columns

    # Fill missing values with the mean of numeric columns for the train set
    train_set = train_set.fillna(train_set[numeric_cols].mean())

    # Fill missing values with the mean of numeric columns
    # from the train set for the test set
    test_set = test_set.fillna(train_set[numeric_cols].mean())

    return train_set, test_set


def mode_imputation(train_set, test_set):
    """
    Perform mode imputation to fill missing values in both train and test sets.

    Parameters
    ----------
    train_set : pandas.DataFrame
        The training dataset.
    test_set : pandas.DataFrame
        The test dataset.

    Returns
    -------
    tuple of pandas.DataFrame
        The train and test sets after mode imputation.
    """

    # Iterate over each column of the train set
    for col in train_set.columns:
        # Check if the column is of type "object"
        if train_set[col].dtypes == "object":
            # Impute with the most frequent value
            most_frequent_value = train_set[col].value_counts().index[0]
            train_set = train_set.fillna({col: most_frequent_value})
            test_set = test_set.fillna({col: most_frequent_value})

    return train_set, test_set


def encode_categorical_features(dataset):
    """
    Encode categorical features in the dataset using one-hot encoding.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset containing categorical features to be encoded.

    Returns
    -------
    pandas.DataFrame
        The dataset after encoding categorical features using one-hot encoding.
    """

    # Convert categorical features in the dataset
    dataset = pd.get_dummies(dataset)

    return dataset


def align_test_set_with_train_set(train_set, test_set):
    """
    Align the columns of the test set with the train set.

    Parameters
    ----------
    train_set_encoded : pandas.DataFrame
        The encoded training dataset.
    test_set_encoded : pandas.DataFrame
        The encoded test dataset.

    Returns
    -------
    pandas.DataFrame
        The test set with columns aligned with the train set.
    """

    # Reindex the columns of the test set to align with the train set
    test_set = test_set.reindex(columns=train_set.columns, fill_value=0)
    return test_set


def segregate_features_and_labels(dataset):
    """
    Segregate features and labels from the dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset containing both features and labels.

    Returns
    -------
    tuple of array-like
        X : The feature matrix.
        y : The target labels.
    """

    # Extract features and labels
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, [-1]].values.ravel()
    return X, y


def rescale_features(X_train, X_test, feature_range=(0, 1)):
    """
    Rescale features to a specified range using Min-Max scaling.

    Parameters
    ----------
    X_train : array-like
        The feature matrix of the training dataset.
    X_test : array-like
        The feature matrix of the test dataset.
    feature_range : tuple, optional
        Desired range of transformed data (default is (0, 1)).

    Returns
    -------
    tuple of array-like
        X_train : The rescaled feature matrix of the training dataset.
        X_test : The rescaled feature matrix of the test dataset.
    """

    # Instantiate MinMaxScaler
    scaler = MinMaxScaler(feature_range=feature_range)

    # Rescale X_train
    X_train = scaler.fit_transform(X_train)

    # Rescale X_test
    X_test = scaler.transform(X_test)

    return X_train, X_test


def train_logreg_model(X_train, y_train):
    """
    Train a logistic regression model using the training data.

    Parameters
    ----------
    X_train : array-like
        The feature matrix of the training dataset.
    y_train : array-like
        The target labels of the training dataset.

    Returns
    -------
    LogisticRegression
        The trained logistic regression model.
    """

    # Instantiate a LogisticRegression classifier
    logreg = LogisticRegression()

    # Fit logreg to the train set
    logreg.fit(X_train, y_train)

    return logreg


def evaluate_logreg_model(model, X_test, y_test):
    """
    Evaluate the logistic regression model using the test data.

    Parameters
    ----------
    model : LogisticRegression
        The trained logistic regression model.
    X_test : array-like
        The feature matrix of the test dataset.
    y_test : array-like
        The true target labels of the test dataset.

    Returns
    -------
    None
        This function prints the confusion matrix of the
        logistic regression model.
    """

    # Use logreg to predict instances from the test set
    y_pred = model.predict(X_test)

    # Print the confusion matrix of the logreg model
    print("Confusion matrix: \n",
          confusion_matrix(y_test, y_pred))


def perform_grid_search(X_train, y_train, estimator, param_grid, cv=5):
    """
    Perform grid search to find the best hyperparameters for the
    given estimator.

    Parameters
    ----------
    X_train : array-like
        The feature matrix of the training dataset.
    y_train : array-like
        The target labels of the training dataset.
    estimator : estimator object
        The estimator (classifier or regressor) for which grid search
        is performed.
    param_grid : dict
        Dictionary with parameter names (string) as keys and lists of
        parameter settings to try as values.
    cv : int, optional
        Number of cross-validation folds (default is 5).

    Returns
    -------
    GridSearchCV
        The fitted GridSearchCV object.
    """

    # Instantiate GridSearchCV with the required parameters
    grid_model = GridSearchCV(estimator=estimator,
                              param_grid=param_grid, cv=cv)

    # Fit grid_model to the data
    grid_model_result = grid_model.fit(X_train, y_train.ravel())

    return grid_model_result


def summarise_grid_search_results(grid_model_result, X_test, y_test):
    """
    Summarize the results of grid search and evaluate the best model
    on the test set.

    Parameters
    ----------
    grid_model_result : GridSearchCV
        The result of grid search.
    X_test : array-like
        The feature matrix of the test dataset.
    y_test : array-like
        The true target labels of the test dataset.

    Returns
    -------
    None
        This function prints the best score achieved and the accuracy
        of the best model on the test set.
    """

    # Summarise the results
    best_score, best_params = (grid_model_result.best_score_,
                               grid_model_result.best_params_)
    print("The best score of {} is achieved with these parameters: {}".format(
        best_score, best_params))

    # Extract the best model and evaluate it on the test set
    best_model = grid_model_result.best_estimator_
    print("Accuracy of logistic regression classifier: ",
          best_model.score(X_test, y_test))
