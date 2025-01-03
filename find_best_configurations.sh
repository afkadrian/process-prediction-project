#!/bin/bash

# Check if n_seeds is provided as an argument
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <n_seeds>"
  exit 1
fi

# Number of seeds (passed as an argument)
n_seeds=$1

# Directories containing results
results_dir="results"  # Change this to your results directory if different

# Initialize arrays to store sums for each configuration
rf_sum=()
gb_sum=()

# Loop through all results directories
for dir in "$results_dir"/*; do
  if [[ -f "$dir/BPI_Challenge_2013_open_problems.xes/results.json" ]]; then
    # Extract configuration details from the results.json file
    n_estimators=$(jq -r '.n_estimators' "$dir/BPI_Challenge_2013_open_problems.xes/results.json")
    window_size=$(jq -r '.max_length' "$dir/BPI_Challenge_2013_open_problems.xes/results.json")
    random_seed=$(jq -r '.random_seed' "$dir/BPI_Challenge_2013_open_problems.xes/results.json")
    boosting=$(jq -r '.boosting' "$dir/BPI_Challenge_2013_open_problems.xes/results.json")
    accuracy=$(jq -r '.activity_accuracy' "$dir/BPI_Challenge_2013_open_problems.xes/results.json")

    # Validate accuracy value
    if [[ -z "$accuracy" || ! "$accuracy" =~ ^[0-9]*\.?[0-9]+$ ]]; then
      echo "Warning: Invalid accuracy value '$accuracy' in $dir/BPI_Challenge_2013_open_problems.xes/results.json. Skipping."
      continue
    fi

    # Create a unique key for the configuration (excluding random_seed)
    config_key="${n_estimators}_${window_size}"

    # Update sums based on boosting type
    if [[ "$boosting" == "true" ]]; then
      # Gradient Boosting
      index=-1
      for i in "${!gb_sum[@]}"; do
        if [[ "${gb_sum[$i]%% *}" == "$config_key" ]]; then
          index=$i
          break
        fi
      done
      if [[ "$index" -eq -1 ]]; then
        gb_sum+=("$config_key 0")
        index=$((${#gb_sum[@]} - 1))
      fi
      sum=$(echo "${gb_sum[$index]#* } + $accuracy" | bc)
      gb_sum[$index]="$config_key $sum"
    else
      # Random Forest
      index=-1
      for i in "${!rf_sum[@]}"; do
        if [[ "${rf_sum[$i]%% *}" == "$config_key" ]]; then
          index=$i
          break
        fi
      done
      if [[ "$index" -eq -1 ]]; then
        rf_sum+=("$config_key 0")
        index=$((${#rf_sum[@]} - 1))
      fi
      sum=$(echo "${rf_sum[$index]#* } + $accuracy" | bc)
      rf_sum[$index]="$config_key $sum"
    fi
  fi
done

# Function to find the configuration with the highest sum and return the average per seed
find_best_config() {
  local sum_array=("${!1}")
  local model_name=$2
  local n_seeds=$3

  best_sum=0
  best_config=""

  for entry in "${sum_array[@]}"; do
    config_key="${entry%% *}"
    sum="${entry#* }"
    if (( $(echo "$sum > $best_sum" | bc -l) )); then
      best_sum=$sum
      best_config="$config_key"
    fi
  done

  # Calculate the average sum per seed
  avg_sum=$(echo "scale=4; $best_sum / $n_seeds" | bc)

  echo "Best $model_name configuration: $best_config with average sum per seed $avg_sum"
}

# Find and print the best configurations
find_best_config rf_sum[@] "Random Forest" "$n_seeds"
find_best_config gb_sum[@] "Gradient Boosting" "$n_seeds"