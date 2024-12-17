import models
import torch.nn as nn
import torch
import numpy as np
import datetime
import socket
import json
import argparse
import data_preprocessing
import random
import os
import math
import copy
import utils
import pandas as pd
import pm4py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


# Funktion zum Laden und Vorverarbeiten der Daten
def load_data(log_with_prefixes, subset, batch_size):
    # Lade Präfixe und Zielwerte für Aktivitäten und Zeiten aus den Logs
    activities_prefixes = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['prefixes']
    activities_suffixes_target = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target']
    times_prefixes = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['prefixes']
    times_suffixes_target = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['suffixes']['target']

    X_activities = []
    y_activities = []
    X_times = []
    y_times =[]
    for prefix_length in activities_prefixes.keys():
        X_activities += activities_prefixes[prefix_length]
        y_activities += activities_suffixes_target[prefix_length]
        X_times += times_prefixes[prefix_length]
        y_times += times_suffixes_target[prefix_length]
    
    #activities_prefixes_array = np.array(list(activities_prefixes.values()))
    #activities_suffixes_target_array = np.array(list(activities_suffixes_target.values()))
    #times_prefixes_array = np.array(list(times_prefixes.values()))
    #times_suffixes_target_array = np.array(list(times_suffixes_target.values()))

    # Flach die Präfixe und Zielwerte auf, damit sie für tabellarische Modelle wie Random Forest geeignet sind
    #X_activities = pd.DataFrame.from_dict(data=activities_prefixes)#.reshape(-1, activities_prefixes_array.shape[-1])
    #y_activities = pd.DataFrame.from_dict(data=activities_suffixes_target)#.reshape(-1)
    #X_times = pd.DataFrame.from_dict(data=times_prefixes)#.reshape(-1, times_prefixes_array.shape[-1])
    #y_times = pd.DataFrame.from_dict(data=times_suffixes_target)#.reshape(-1)

    return X_activities, y_activities, X_times, y_times


# Hauptfunktion zur Ausführung des Trainings- und Evaluierungsprozesses
def main(args, dt_object):

    # Data prep
    logs_dir = './logs/'

    with open(os.path.join('config', 'logs_meta.json')) as f:
        logs_meta = json.load(f)

    # data_preprocessing.download_logs(logs_meta, logs_dir)
    distributions, logs = data_preprocessing.create_distributions(logs_dir)

    for log_name in logs:
        # Verarbeite ein einzelnes Log für das Experiment
        processed_log = data_preprocessing.create_structured_log(logs[log_name], log_name=log_name)

        if os.path.isdir(os.path.join('split_logs', log_name)):
            for file_name in sorted(os.listdir(os.path.join('split_logs', log_name))):
                if file_name.startswith('split_log_'):
                    split_log_file_name = os.path.join('split_logs', log_name, file_name)
                    with open(split_log_file_name) as f_in:
                        split_log = json.load(f_in)
                    print(split_log_file_name + ' is used as common data')
            del processed_log
        else:
            split_log = data_preprocessing.create_split_log(processed_log, validation_ratio=args.validation_split)


        # Erstelle Präfixe aus dem Log für das Training und die Validierung
        log_with_prefixes = data_preprocessing.create_prefixes(split_log,
                                                               min_prefix=2,
                                                               create_tensors=False,  # Für Random Forest nicht notwendig
                                                               add_special_tokens=False,
                                                               pad_sequences=False,
                                                               to_wrap_into_torch_dataset=False)

        # Lade Trainings- und Validierungsdaten für Aktivitäten und Zeiten
        X_train_activities, y_train_activities, X_train_times, y_train_times = load_data(log_with_prefixes, 'training', args.training_batch_size)
        X_val_activities, y_val_activities, X_val_times, y_val_times = load_data(log_with_prefixes, 'validation', args.validation_batch_size)

        # Trainiere den Random Forest Classifier für die Vorhersage von Aktivitäten
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=args.random_seed)
        rf_classifier.fit(X_train_activities, y_train_activities)  # Passe das Modell an die Trainingsdaten an

        # Evaluieren der Vorhersagegenauigkeit für Aktivitäten
        y_pred_activities = rf_classifier.predict(X_val_activities)  # Vorhersagen auf Validierungsdaten
        activity_accuracy = accuracy_score(y_val_activities, y_pred_activities)  # Berechne die Genauigkeit
        print(f'Activity Prediction Accuracy: {activity_accuracy:.4f}')  # Ausgabe der Genauigkeit

        # Trainiere den Random Forest Regressor für die Vorhersage von Zeiten
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=args.random_seed)
        rf_regressor.fit(X_train_times, y_train_times)  # Passe das Modell an die Trainingsdaten an

        # Evaluieren des mittleren quadratischen Fehlers (MSE) für Zeiten
        y_pred_times = rf_regressor.predict(X_val_times)  # Vorhersagen auf Validierungsdaten
        time_mse = mean_squared_error(y_val_times, y_pred_times)  # Berechne den mittleren quadratischen Fehler
        print(f'Time Prediction MSE: {time_mse:.4f}')  # Ausgabe des Fehlers

# Ausführung der Hauptfunktion mit Befehlszeilenargumenten
if __name__ == '__main__':
    dt_object = datetime.datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
    parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
    parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
    parser.add_argument('--nb_epoch', help='training iterations', default=400, type=int)
    parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=2560, type=int)
    parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=2560, type=int)
    parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
    parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks', type=str) # we would like to end up with some non-stochastic & at least pseudo likelihood metric
    parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
    parser.add_argument('--dropout_prob', help='dropout_prob', default=0.3, type=float)
    parser.add_argument('--training_learning_rate', help='GD learning rate', default=1e-4, type=float)
    parser.add_argument('--training_gaussian_process', help='GP', default=1e-5, type=float)
    parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
    parser.add_argument('--dataset', help='dataset', default='', type=str)
    parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
    parser.add_argument('--random', help='if random', default=True, type=bool)
    parser.add_argument('--gpu', help='gpu', default=6, type=int)
    parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!', default='[0,1,4,10,15]', type=str)
    parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
    parser.add_argument('--architecture', help='BERT or GPT', default='BERT', type=str)
    parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
    parser.add_argument('--device', help='GPU or CPU', default='CPU', type=str)
    parser.add_argument('--lagrange_a', help='Langrange multiplier', default=1.0, type=float)
    parser.add_argument('--save_criterion_threshold', help='save_criterion_threshold', default=4.0, type=float)
    parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
    parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
    parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=1.0, type=float)
    parser.add_argument('--early_stopping', help='early_stopping', default=True, type=bool)
    parser.add_argument('--single_position_target', help='single_position_target', default=True, type=bool)

    args = parser.parse_args()
    
    vars(args)['hostname'] = str(socket.gethostname())
    
    print('This is training of: ' + dt_object.strftime("%Y%m%d%H%M"))

    main(args, dt_object)