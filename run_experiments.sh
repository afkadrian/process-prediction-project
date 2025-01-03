#!/bin/bash

# Arrays of parameters
n_estimators=(50 100 500 1000 5000)
boosting=("" "--boosting")
window_size=(2 3 5)
random_seed=(1 2 3 4 5 6 7 8 9 10)

# Counter for trial_id
trial_id=1

# Loop through all combinations of parameters
for estimators in "${n_estimators[@]}"; do
  for window in "${window_size[@]}"; do
    for seed in "${random_seed[@]}"; do
      for boost in "${boosting[@]}"; do
        # Create a unique tmux session name
        session_name="training_${boost}_${trial_id}"
        
        # Start a new tmux session and run the training script
        tmux new-session -d -s "$session_name" \
          "python3 training_rfc.py --n_estimators $estimators $boost --window_size $window --random_seed $seed --trial_id $trial_id"
    
      done
      # Increment trial_id for the next run
      trial_id=$((trial_id + 1))  
    done
  done
done

echo "All training sessions started in tmux."