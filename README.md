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

1. `training_<xxx>.py`: Training of the models
2. `evaluation.py`: Inference and evaluation of the models. Calculates DLS and MAE results.
3. `all_models_dls.py` / `all_models_mae_denormalised.py` / `case_length_statistics.py`: Visualize the results of the models.

### Logs

https://drive.google.com/file/d/1IGOI7YYL6njWb9CTtuXwqBeScJslh3T0/view?usp=sharing