from functions import load_dataset, drop_features, split_train_test, \
    replace_question_marks_with_nan, mean_imputation, mode_imputation, \
    encode_categorical_features, align_test_set_with_train_set, \
    segregate_features_and_labels, rescale_features, train_logreg_model, \
    evaluate_logreg_model, perform_grid_search, summarise_grid_search_results


# Specify file path
file_path = "data/cc_approvals.data"

# Determine columns to drop
columns_to_drop = [11, 13]

# Create hyperparameter grid
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)


def main():
    # Import data
    cc_apps = load_dataset(file_path)

    # Drop features
    cc_apps = drop_features(cc_apps, columns_to_drop)

    # Split into training and test set
    train_set, test_set = split_train_test(cc_apps)

    # Replace question marks with NaN
    train_set = replace_question_marks_with_nan(train_set)
    test_set = replace_question_marks_with_nan(test_set)

    # Impute mean for numeric columns
    train_set, test_set = mean_imputation(train_set, test_set)

    # Impute mode for columns with data type "object"
    train_set, test_set = mode_imputation(train_set, test_set)

    # Encode categorical features
    train_set = encode_categorical_features(train_set)
    test_set = encode_categorical_features(test_set)

    # Align test set with training set
    test_set_aligned = align_test_set_with_train_set(train_set, test_set)

    # Segregate features and labels
    X_train, y_train = segregate_features_and_labels(train_set)
    X_test, y_test = segregate_features_and_labels(test_set_aligned)

    # Rescale features with MinMaxScaler
    rescaled_X_train, rescaled_X_test = rescale_features(X_train, X_test)

    # Train LogReg model
    logreg = train_logreg_model(rescaled_X_train, y_train)

    # Evaluate the model
    evaluate_logreg_model(logreg, rescaled_X_test, y_test)

    # Perform GridSearch CV
    grid_model_result = perform_grid_search(rescaled_X_train, y_train,
                                            logreg, param_grid)

    # Summarise GridSearch result
    summarise_grid_search_results(grid_model_result, rescaled_X_test, y_test)


if __name__ == "__main__":
    main()
