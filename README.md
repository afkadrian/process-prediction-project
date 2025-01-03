# Project Process Prediction: Unified framework for sequential DL model comparisons

- Original Paper: https://arxiv.org/abs/2110.10225v1
- Task: https://moodle.hu-berlin.de/pluginfile.php/6441682/mod_resource/content/1/ProjectInstructions.pdf

## Team Members

- Adrian Schiller
- Denis A. Kasakow
- Ehiwen Obaseki aka. Aari

## Abgaben

- Presentation: https://docs.google.com/presentation/d/1zwoWt3s0DLO7dLLgZ32y4eB7C6agWLoB2lFTGqw8NBQ/edit#slide=id.p
- Report: https://latex.hu-berlin.de/project/67616832e2dbb4e0fe375cc0

## Code Structure

Scripts with main functions:

- `training_<xxx>.py`: Training of the models
  - results/:
    - Pro Modell, Pro Event Log:

      - checkpoints: **for evaluation**
        - model

      - split_log_timestamp.json: **for evaluation**
        - basically eventlog in json format
        - mit training + test split

      - experiment_parameters.json
        - arguments aus training_rnn.py

      - training_figures_timestamp.json
        - pro epoche
          - loss, accuracy, f1, precision, recall

- `evaluation.py`: Inference and evaluation of the models. Calculates DLS and MAE results.
  - results/:
    - Pro Modell, Pro Event Log:

      - suffix_evaluation_result_dls_mae_timestamp.json: **for visualization**
        - DLS and MAE results

      - suffix_generation_result_timestamp.json
        - generated event log

      - evaluation_parameters.json
        - arguments aus evaluation.py

- `all_models_dls.py` / `all_models_mae_denormalised.py` / `case_length_statistics.py`: Visualize the results of the models.
  - figures/:
    - all_models_dls.png: **for figure 4**
    - all_models_mae_denormalised.png: **for figure 5**
    - case_length_statistics.png: **for figure 3**
    - nb_traces_longer_than_prefix.json: utility file
    - table_dls.tex
    - table_mae_normalized.tex
    - table_transpose_dls.tex: **for table 1**
    - table_transpose_mae_normalized.tex: **for table 2**

### Logs

https://drive.google.com/file/d/1IGOI7YYL6njWb9CTtuXwqBeScJslh3T0/view?usp=sharing