import argparse
import datetime
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm  # For progress bar
import data_preprocessing  # Assuming this module handles data loading and splitting

def prepare_data(log_with_prefixes, subset='training', max_length=None):
    """
    Convert sequential data into tabular format for Random Forest or Gradient Boosting.
    Args:
        log_with_prefixes: Dictionary containing prefixes and suffixes.
        subset: 'training' or 'validation'.
        max_length: Maximum sequence length (calculated globally).
    Returns:
        X: Feature matrix (n_samples, n_features).
        y_activity: Activity labels (n_samples,).
        y_time: Time labels (n_samples,).
        max_length: Maximum sequence length (used for consistency).
    """
    activities_prefixes = []
    times_prefixes = []
    activities_suffixes_target = []
    times_suffixes_target = []

    # If max_length is not provided, calculate it globally
    if max_length is None:
        max_length = max(
            max(len(seq) for seq in log_with_prefixes['training_prefixes_and_suffixes']['activities']['prefixes'].values()),
            max(len(seq) for seq in log_with_prefixes['training_prefixes_and_suffixes']['times']['prefixes'].values()),
            max(len(seq) for seq in log_with_prefixes['validation_prefixes_and_suffixes']['activities']['prefixes'].values()),
            max(len(seq) for seq in log_with_prefixes['validation_prefixes_and_suffixes']['times']['prefixes'].values())
        )

    print(f"Max length: {max_length}")  # Debugging

    # Use tqdm for progress bar
    for prefix in tqdm(log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['prefixes'].keys(), desc=f"Preparing {subset} data"):
        # Extract prefixes
        activities_seq = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['prefixes'][prefix]
        times_seq = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['prefixes'][prefix]

        # Flatten sequences (if they are nested)
        activities_seq = np.array(activities_seq).flatten()
        times_seq = np.array(times_seq).flatten()

        # Truncate sequences longer than max_length
        if len(activities_seq) > max_length:
            activities_seq = activities_seq[:max_length]
        if len(times_seq) > max_length:
            times_seq = times_seq[:max_length]

        # Pad sequences shorter than max_length
        if len(activities_seq) < max_length:
            activities_padded = np.pad(activities_seq, (0, max_length - len(activities_seq)), mode='constant', constant_values=0)
        else:
            activities_padded = activities_seq  # No padding needed

        if len(times_seq) < max_length:
            times_padded = np.pad(times_seq, (0, max_length - len(times_seq)), mode='constant', constant_values=0.0)
        else:
            times_padded = times_seq  # No padding needed

        activities_prefixes.append(activities_padded)
        times_prefixes.append(times_padded)

        # Extract suffixes (single position target)
        activities_suffixes_target.append(log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target'][prefix][0])
        times_suffixes_target.append(log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['suffixes']['target'][prefix][0])

    # Convert lists to numpy arrays
    activities_prefixes = np.array(activities_prefixes)
    times_prefixes = np.array(times_prefixes)
    activities_suffixes_target = np.array(activities_suffixes_target)
    times_suffixes_target = np.array(times_suffixes_target)

    # Combine prefixes into feature matrix
    X = np.column_stack([activities_prefixes, times_prefixes])
    y_activity = activities_suffixes_target.flatten()
    y_time = times_suffixes_target.flatten()

    return X, y_activity, y_time, max_length

def train_model(log_with_prefixes, args, output_path, max_length=None):
    """
    Train and evaluate a model (Random Forest or Gradient Boosting).
    Args:
        log_with_prefixes: Dictionary containing training and validation data.
        args: Command-line arguments.
        output_path: Directory to save results.
    """
    # Prepare training data and calculate max_length
    X_train, y_train_activity, y_train_time, max_length = prepare_data(log_with_prefixes, subset='training', max_length=max_length)

    # Prepare validation data using the same max_length
    X_val, y_val_activity, y_val_time, _ = prepare_data(log_with_prefixes, subset='validation', max_length=max_length)

    # Choose model based on boosting flag
    if args.boosting:
        print("Using Gradient Boosting for activity prediction...")
        activity_model = GradientBoostingClassifier(n_estimators=args.n_estimators, random_state=args.random_seed)
        print("Using Gradient Boosting for time prediction...")
        time_model = GradientBoostingRegressor(n_estimators=args.n_estimators, random_state=args.random_seed)
    else:
        print("Using Random Forest for activity prediction...")
        activity_model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_seed)
        print("Using Random Forest for time prediction...")
        time_model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_seed)

    # Train models with progress bar
    if args.boosting:
        # Gradient Boosting supports warm_start for incremental training
        print("Training activity model...")
        for i in tqdm(range(args.n_estimators), desc="Training activity model"):
            activity_model.n_estimators = i + 1
            activity_model.fit(X_train, y_train_activity)

        print("Training time model...")
        for i in tqdm(range(args.n_estimators), desc="Training time model"):
            time_model.n_estimators = i + 1
            time_model.fit(X_train, y_train_time)
    else:
        # Random Forest does not support warm_start, so train in one go
        print("Training activity model...")
        activity_model.fit(X_train, y_train_activity)

        print("Training time model...")
        time_model.fit(X_train, y_train_time)

    # Evaluate models
    y_pred_activity = activity_model.predict(X_val)
    y_pred_time = time_model.predict(X_val)

    activity_accuracy = accuracy_score(y_val_activity, y_pred_activity)
    time_mse = mean_squared_error(y_val_time, y_pred_time)

    print(f"Validation Activity Accuracy: {activity_accuracy:.4f}")
    print(f"Validation Time MSE: {time_mse:.4f}")

    # Save results
    with open(os.path.join(output_path, 'results.txt'), 'w') as f:
        f.write(f"Validation Activity Accuracy: {activity_accuracy:.4f}\n")
        f.write(f"Validation Time MSE: {time_mse:.4f}\n")

def main(args):
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Load and preprocess data
    logs_dir = './logs/'
    with open(os.path.join('config', 'logs_meta.json')) as f:
        logs_meta = json.load(f)

    distributions, logs = data_preprocessing.create_distributions(logs_dir)

    for log_name in logs:
        processed_log = data_preprocessing.create_structured_log(logs[log_name], log_name=log_name)
        split_log = data_preprocessing.create_split_log(processed_log, validation_ratio=args.validation_split)

        # Create prefixes and suffixes
        log_with_prefixes = data_preprocessing.create_prefixes(
            split_log,
            min_prefix=1,
            create_tensors=False,  # Random Forest/Gradient Boosting doesn't need tensors
            add_special_tokens=True,
            pad_sequences=True,
            pad_token=args.pad_token,
            to_wrap_into_torch_dataset=False,  # No need for PyTorch DataLoader
            training_batch_size=args.training_batch_size,
            validation_batch_size=args.validation_batch_size,
            single_position_target=args.single_position_target
        )

        # Create output directory
        output_path = os.path.join('results', 'random_forest' if not args.boosting else 'gradient_boosting', str(processed_log['id']))
        os.makedirs(output_path, exist_ok=True)

        # Train and evaluate model
        train_model(log_with_prefixes, args, output_path, max_length=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Random Forest or Gradient Boosting model for process suffix prediction.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest or boosting stages.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data to use for validation.')
    parser.add_argument('--training_batch_size', type=int, default=2560, help='Batch size for training data.')
    parser.add_argument('--validation_batch_size', type=int, default=2560, help='Batch size for validation data.')
    parser.add_argument('--pad_token', type=int, default=0, help='Token used for padding sequences.')
    parser.add_argument('--single_position_target', type=bool, default=True, help='Whether to predict a single position or the entire suffix.')
    parser.add_argument('--boosting', action='store_true', help='Use Gradient Boosting instead of Random Forest.')

    args = parser.parse_args()
    main(args)