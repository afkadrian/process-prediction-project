from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import json
import os
import argparse

# Funktion zum Laden und Vorverarbeiten der Daten
def load_data(log_with_prefixes, subset, batch_size):
    # Lade Präfixe und Zielwerte für Aktivitäten und Zeiten aus den Logs
    activities_prefixes = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['prefixes']
    activities_suffixes_target = log_with_prefixes[subset + '_prefixes_and_suffixes']['activities']['suffixes']['target']
    times_prefixes = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['prefixes']
    times_suffixes_target = log_with_prefixes[subset + '_prefixes_and_suffixes']['times']['suffixes']['target']

    # Flach die Präfixe und Zielwerte auf, damit sie für tabellarische Modelle wie Random Forest geeignet sind
    X_activities = activities_prefixes.reshape(-1, activities_prefixes.shape[1])
    y_activities = activities_suffixes_target.reshape(-1)
    X_times = times_prefixes.reshape(-1, times_prefixes.shape[1])
    y_times = times_suffixes_target.reshape(-1)

    return X_activities, y_activities, X_times, y_times

# Hauptfunktion zur Ausführung des Trainings- und Evaluierungsprozesses
def main(args):
    # Definiere das Verzeichnis, in dem die Logs gespeichert sind
    logs_dir = './logs/'

    # Lade Metadaten über die Logs aus einer JSON-Konfigurationsdatei
    with open(os.path.join('config', 'logs_meta.json')) as f:
        logs_meta = json.load(f)

    # Erstelle Verteilungen und lade Logs
    distributions, logs = data_preprocessing.create_distributions(logs_dir)

    # Verarbeite ein einzelnes Log für das Experiment
    processed_log = data_preprocessing.create_structured_log(logs[args.dataset], log_name=args.dataset)

    # Erstelle Präfixe aus dem Log für das Training und die Validierung
    log_with_prefixes = data_preprocessing.create_prefixes(processed_log,
                                                           min_prefix=2,
                                                           create_tensors=False,  # Für Random Forest nicht notwendig
                                                           add_special_tokens=False,
                                                           pad_sequences=False)

    # Lade Trainings- und Validierungsdaten für Aktivitäten und Zeiten
    X_train_activities, y_train_activities, X_train_times, y_train_times = load_data(log_with_prefixes, 'training', args.batch_size)
    X_val_activities, y_val_activities, X_val_times, y_val_times = load_data(log_with_prefixes, 'validation', args.batch_size)

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
    parser = argparse.ArgumentParser()

    # Hinzufügen von Argumenten zur Steuerung der Skriptausführung
    parser.add_argument('--dataset', help='Dataset name', default='sample_log', type=str)  # Name des Datensatzes
    parser.add_argument('--random_seed', help='Random seed for reproducibility', default=42, type=int)  # Zufallssamen
    parser.add_argument('--batch_size', help='Batch size for data loading', default=128, type=int)  # Batch-Größe

    args = parser.parse_args()  # Analysiere die Argumente
    main(args)  # Starte die Hauptfunktion
