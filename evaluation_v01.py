from data import * 
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


#%%
# Lista nazw zbiorów danych
#datasets =  ['ISA_09_10', 'EPL_09_10', 'FL1_09_10', 'GB1_09_10', 'SLL_09_10']
datasets = ['EPL_09_10', 'EPL_10_11', 'EPL_11_12','EPL_12_13', 'EPL_13_14', 'EPL_14_15', 'EPL_15_16', 'EPL_16_17', 'EPL_17_18', 'EPL_18_19', 'EPL_19_20', 'EPL_20_21', 'EPL_21_22',
             'FL1_09_10', 'FL1_10_11', 'FL1_11_12','FL1_12_13', 'FL1_13_14', 'FL1_14_15', 'FL1_15_16', 'FL1_16_17', 'FL1_17_18', 'FL1_18_19', 'FL1_20_21', 'FL1_21_22',
             'GB1_09_10', 'GB1_10_11', 'GB1_11_12','GB1_12_13', 'GB1_13_14', 'GB1_14_15', 'GB1_15_16', 'GB1_16_17', 'GB1_17_18', 'GB1_18_19', 'GB1_19_20', 'GB1_20_21', 'GB1_21_22',
             'ISA_09_10', 'ISA_10_11', 'ISA_11_12','ISA_12_13', 'ISA_13_14', 'ISA_14_15', 'ISA_15_16', 'ISA_16_17', 'ISA_17_18', 'ISA_18_19', 'ISA_19_20', 'ISA_20_21', 'ISA_21_22',
             'SLL_09_10', 'SLL_10_11', 'SLL_11_12','SLL_12_13', 'SLL_13_14', 'SLL_14_15', 'SLL_15_16', 'SLL_16_17', 'SLL_17_18', 'SLL_18_19', 'SLL_19_20', 'SLL_20_21', 'SLL_21_22']

# Słownik do przechowywania wyników dla każdego zbioru danych
results = {}

# Pętla przez każdy zbiór danych
for dataset_name in datasets:
    # Pobierz dane
    data = globals()[dataset_name]
    
    # Obliczenia średnich
    if dataset_name.startswith('GB1'):  # Bundesliga - 18 drużyn
        n_matches = 306
    else:  # EPL, FL1, ISA, SLL - 20 drużyn
        n_matches = 380
    
    # Wielkość zbioru testowego = dwie ostatnie kolejki
    # Bundesliga: 2 × 9 meczów = 18
    # Pozostałe ligi: 2 × 10 meczów = 20
    test_size = 18 if dataset_name.startswith('GB1') else 20
    
    # Tworzenie tabeli
    table = pd.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS'))
     
    train_data = data.iloc[:-test_size].copy()

    avg_home_scored = train_data.FTHG.sum() / len(train_data)
    avg_away_scored = train_data.FTAG.sum() / len(train_data)
    avg_home_conceded = avg_away_scored
    avg_away_conceded = avg_home_scored
    
    # Grupowanie danych
    res_home = train_data.groupby('HomeTeam')
    res_away = train_data.groupby('AwayTeam')
    
    # Wypełnianie tabeli
    table.Team = list(res_home.groups.keys())
    table.HGS = res_home.FTHG.sum().values
    table.HGC = res_home.FTAG.sum().values
    table.AGS = res_away.FTAG.sum().values
    table.AGC = res_away.FTHG.sum().values
    
    # Liczba meczów u siebie/na wyjeździe per drużyna (dynamicznie!)
    n_home_games = res_home.size().values  # ile meczów każda drużyna grała u siebie w treningu
    n_away_games = res_away.size().values  # ile meczów każda drużyna grała na wyjeździe w treningu
    
    table.HAS = (table.HGS / n_home_games) / avg_home_scored  # HAS home attack strength
    table.AAS = (table.AGS / n_away_games) / avg_away_scored  # AAS away attack strength
    table.HDS = (table.HGC / n_home_games) / avg_home_conceded  # HDS home defence strength
    table.ADS = (table.AGC / n_away_games) / avg_away_conceded  # ADS away defence strength
    
    # Przygotowanie cech
    features = data[['HomeTeam','AwayTeam','FTR']]
    # Łączenie ze statystykami (merge zamiast pętli)
    features = features.merge(table[['Team', 'HAS', 'HDS']], left_on='HomeTeam', right_on='Team', how='left')
    features = features.merge(table[['Team', 'AAS', 'ADS']], left_on='AwayTeam', right_on='Team', how='left', suffixes=('', '_away'))
    features['Result'] = features[['FTR']].replace(['H', 'A', 'D'], [1, 2, 0])
    y = features["Result"]
    
    # Zapisz wyniki do słownika
    results[dataset_name] = {
        'table': table,
        'features': features,
        'y': y
    }

#%%
# Definiowanie modeli i ich parametrów do strojenia
models = {
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': np.arange(3, 21, 2)
    }),
    'SVM': (SVC(), {
        'C': np.logspace(-1, 1, 10),
        'gamma': np.logspace(-4, -2, 5),
        'kernel': ['linear', 'rbf']
    }),
    'RF': (RandomForestClassifier(), {
        'n_estimators': [50, 100, 150, 200, 225, 250],
          'max_depth': [None, 10, 15, 25, 20, 30]
    }),
    'DT': (DecisionTreeClassifier(), {
        'max_depth': [None, 5, 10, 15, 20, 25, 30]
    }),
    'NB': (GaussianNB(), {}),  # Brak hiperparametrów do strojenia
    'NN': (MLPClassifier(max_iter=500), {
        'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (20, 10), (30, 15)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]  #L2 regularization!
        }
        )
}

# Tabela do przechowywania wyników ewaluacji
all_evaluation_results_01 = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy', 'Recall_weighted',
                                                  'Recall_macro', 'Precision_weighted', 'Precision_macro',
                                                  'F1_weighted', 'F1_macro', 'Best Params'])

# Pętla przez każdy zbiór danych
for dataset_name in datasets:
    # Pobierz dane
    data = globals()[dataset_name]
    
    # Przygotowanie cech i etykiet
    features = results[dataset_name]['features']
    y = results[dataset_name]['y']
    X = features.drop(columns=['HomeTeam','AwayTeam','Team','Team_away','FTR','Result'])  # Usuwamy kolumny niepotrzebne do modelowania
     
    # Podział na zbiór treningowy i testowy
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Podział na zbiór treningowy i testowy (dwie ostatnie kolejki)
    # Dwie ostatnie kolejki:
    # Bundesliga: 18 drużyn -> 9 meczów w kolejce -> 18 meczów w dwóch kolejkach
    # Pozostałe ligi: 20 drużyn -> 10 meczów w kolejce -> 20 meczów w dwóch kolejkach
    if dataset_name.startswith('GB1'):
        test_size = 18
    else:
        test_size = 20
    
    X_train = X.iloc[:-test_size].copy()
    X_test = X.iloc[-test_size:].copy()
    
    y_train = y.iloc[:-test_size].copy()
    y_test = y.iloc[-test_size:].copy()


    # Skalowanie dopasowane wyłącznie na zbiorze treningowym
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # # Tabela do przechowywania wyników ewaluacji
    evaluation_results = pd.DataFrame(columns=['Dataset','Model','Accuracy','Recall_weighted',
                                               'Recall_macro','Precision_weighted','Precision_macro',
                                               'F1_weighted','F1_macro'])
    
    # Pętla przez każdy model
    for model_name, (model, params) in models.items():
        print(f"Trenowanie modelu {model_name} dla zbioru {dataset_name}...")

        if model_name == 'RF':  # RandomSearchCV dla RandomForest
            search = RandomizedSearchCV(model, param_distributions=params, n_iter=20, cv=5, scoring='f1_macro', random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            best_params = search.best_params_
        elif not params:  # GaussianNB - brak hiperparametrów
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            best_params = 'N/A'
        else:  # GridSearchCV dla pozostałych modeli
            search = GridSearchCV(model, param_grid=params, cv=5, scoring='f1_macro', n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            best_params = search.best_params_

        # Raport
        print(classification_report(y_test, y_pred)) 

        accuracy = accuracy_score(y_test, y_pred)
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Tworzenie nowego wiersza
        new_row = pd.DataFrame({
            'Dataset': [dataset_name],
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Recall_weighted': [recall_weighted],
            'Recall_macro': [recall_macro],
            'Precision_weighted': [precision_weighted],
            'Precision_macro': [precision_macro],
            'F1_weighted': [f1_weighted],
            'F1_macro': [f1_macro],
            'Best Params': [best_params]})
        
        # Dodanie wyników do tabeli
        evaluation_results = pd.concat([evaluation_results, new_row], ignore_index=True)

    # Dodawanie wyników do globalnej tabeli
    # global all_evaluation_results
    all_evaluation_results_01 = pd.concat([all_evaluation_results_01, evaluation_results], ignore_index=True)

    # Zapis wyników do pliku CSV
    evaluation_results.to_csv(f'evaluation_{dataset_name}.csv', index=False)
    print(f"Wyniki dla zbioru {dataset_name} zapisane w pliku evaluation_{dataset_name}.csv")

#%%Wizualizacja
all_evaluation_results_01.to_csv('all_evaluation_results_01.csv', index=False)
