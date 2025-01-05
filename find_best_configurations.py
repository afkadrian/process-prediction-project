import os
import sys
import json
from collections import defaultdict
from decimal import Decimal
import argparse


def main(args):

    n_seeds = args.n_seeds

    # Directory containing results
    results_dir = "results"  # Change this to your results directory if different

    # Initialize dictionaries to store sums for each configuration
    rf_sum_activity = defaultdict(Decimal)
    gb_sum_activity = defaultdict(Decimal)
    rf_sum_time = defaultdict(Decimal)
    gb_sum_time = defaultdict(Decimal)

    # Loop through all results directories
    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            results_path = os.path.join(root, "results.json")
            try:
                with open(results_path, "r") as file:
                    data = json.load(file)

                # Extract configuration details
                n_estimators = data.get("n_estimators")
                window_size = data.get("max_length")
                random_seed = data.get("random_seed")
                boosting = data.get("boosting", False)
                accuracy = data.get("activity_accuracy")
                time_mse = data.get("time_mse")

                # Validate accuracy value
                try:
                    accuracy = Decimal(accuracy)
                except (ValueError, TypeError):
                    print(
                        f"Warning: Invalid accuracy value '{accuracy}' in {results_path}. Skipping."
                    )
                    continue

                try:
                    time_mse = Decimal(time_mse)
                except (ValueError, TypeError):
                    print(
                        f"Warning: Invalid time_mse value '{time_mse}' in {results_path}. Skipping."
                    )
                    continue

                # Create a unique key for the configuration (excluding random_seed)
                config_key = f"{n_estimators}_{window_size}"

                # Update sums based on boosting type
                if boosting == True:
                    gb_sum_activity[config_key] += accuracy
                    gb_sum_time[config_key] += time_mse
                elif boosting == False:
                    rf_sum_activity[config_key] += accuracy
                    rf_sum_time[config_key] += time_mse
                else:
                    raise ValueError(
                        f"Invalid boosting value '{boosting}' in {results_path}"
                    )

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {results_path}: {e}")

    # Function to find the configuration with the highest sum and calculate the average
    def find_best_config_activity(sum_dict, model_name, n_seeds):
        if not sum_dict:
            print(f"No configurations found for {model_name}.")
            return

        best_config = max(sum_dict, key=sum_dict.get)
        best_sum = sum_dict[best_config]
        avg_sum = best_sum / n_seeds

        print(
            f"Best activity {model_name} configuration for activity: {best_config} with average sum per seed {avg_sum:.4f}"
        )

    def find_best_config_time(sum_dict, model_name, n_seeds):
        if not sum_dict:
            print(f"No configurations found for {model_name}.")
            return

        best_config = min(sum_dict, key=sum_dict.get)
        best_sum = sum_dict[best_config]
        avg_sum = best_sum / n_seeds

        print(
            f"Best {model_name} configuration for time: {best_config} with average sum per seed {avg_sum:.4f}"
        )

    # Find and print the best configurations
    find_best_config_activity(rf_sum_activity, "Random Forest", n_seeds)
    find_best_config_activity(gb_sum_activity, "Gradient Boosting", n_seeds)
    find_best_config_time(rf_sum_time, "Random Forest", n_seeds)
    find_best_config_time(gb_sum_time, "Gradient Boosting", n_seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the best configurations for Random Forest and Gradient Boosting models."
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of seeds used for each configuration (default: 10)",
    )

    args = parser.parse_args()
    main(args)
