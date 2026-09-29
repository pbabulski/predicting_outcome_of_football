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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit
)
import matplotlib.pyplot as plt
import seaborn as sns
#%%DEF
def calculate_season_team_stats(season_data):
    """
    Statystyki jednej ligi w jednym sezonie.
    Zwraca jeden wiersz na drużynę.
    """

    home_stats = (
        season_data
        .groupby('HomeTeam')
        .agg(
            HGS=('FTHG', 'mean'),
            HGC=('FTAG', 'mean'),
            HST=('HST', 'mean'),
            HC=('HC', 'mean'),
            HY=('HY', 'mean'),
            HR=('HR', 'mean'),
            Home_matches=('FTHG', 'size')
        )
        .reset_index()
        .rename(columns={'HomeTeam': 'Team'})
    )

    away_stats = (
        season_data
        .groupby('AwayTeam')
        .agg(
            AGS=('FTAG', 'mean'),
            AGC=('FTHG', 'mean'),
            AST=('AST', 'mean'),
            AC=('AC', 'mean'),
            AY=('AY', 'mean'),
            AR=('AR', 'mean'),
            Away_matches=('FTAG', 'size')
        )
        .reset_index()
        .rename(columns={'AwayTeam': 'Team'})
    )

    season_table = home_stats.merge(
        away_stats,
        on='Team',
        how='outer'
    )

    #Średnie ligowe potrzebne do współczynników HAS/HDS/AAS/ADS
    avg_home_scored = season_data['FTHG'].mean()
    avg_away_scored = season_data['FTAG'].mean()

    season_table['HAS'] = season_table['HGS'] / avg_home_scored
    season_table['HDS'] = season_table['HGC'] / avg_away_scored
    season_table['AAS'] = season_table['AGS'] / avg_away_scored
    season_table['ADS'] = season_table['AGC'] / avg_home_scored

    return season_table[
        [
            'Team',
            'HAS', 'HDS', 'AAS', 'ADS',
            'HST', 'AST',
            'HC', 'AC',
            'HY', 'AY',
            'HR', 'AR'
        ]
    ]

def aggregate_history(history_tables, weighted=False):
    """
    Łączy sezonowe tabele statystyk.

    history_tables:
        lista DataFrame'ów ze statystykami kolejnych poprzednich sezonów.

    weighted=False:
        zwykła średnia z sezonów.

    weighted=True:
        średnia ważona; najstarszy sezon ma wagę 1,
        najnowszy sezon historii ma największą wagę.
    """

    history = pd.concat(
        history_tables,
        ignore_index=True
    )

    feature_columns = [
        'HAS', 'HDS', 'AAS', 'ADS',
        'HST', 'AST',
        'HC', 'AC',
        'HY', 'AY',
        'HR', 'AR'
    ]

    if not weighted:
        return (
            history
            .groupby('Team', as_index=False)[feature_columns]
            .mean()
        )

    #Kolejne tabele to kolejne sezony historii:
    #1 = najstarszy, len(history_tables) = najnowszy.
    weighted_tables = []

    for weight, season_table in enumerate(history_tables, start=1):
        temp = season_table.copy()

        for column in feature_columns:
            temp[column] = temp[column] * weight

        temp['Weight'] = weight
        weighted_tables.append(temp)

    weighted_history = pd.concat(
        weighted_tables,
        ignore_index=True
    )

    weighted_sums = (
        weighted_history
        .groupby('Team', as_index=False)[feature_columns]
        .sum()
    )

    weight_sums = (
        weighted_history
        .groupby('Team', as_index=False)['Weight']
        .sum()
    )

    aggregated = weighted_sums.merge(
        weight_sums,
        on='Team',
        how='left'
    )

    for column in feature_columns:
        aggregated[column] = aggregated[column] / aggregated['Weight']

    return aggregated.drop(columns='Weight')


def build_second_approach_league(
    dataset_names,
    persistent_teams,
    weighted=False
):
    """
    Buduje zbiór dla jednej ligi.

    - Pierwszy sezon jest sezonem bazowym.
    - Dla każdego kolejnego sezonu cechy są liczone
      tylko z sezonów wcześniejszych.
    - Sezony poza ostatnim tworzą trening.
    - Ostatni sezon tworzy test.
    """

    #Tabele statystyk sezonowych, tylko dla meczów między zespołami trwałymi
    season_tables = []
    season_matches = []

    for dataset_name in dataset_names:
        season_data = globals()[dataset_name].copy()

        #wyłącznie mecze drużyn obecnych we wszystkich sezonach
        mask = (
            season_data['HomeTeam'].isin(persistent_teams) &
            season_data['AwayTeam'].isin(persistent_teams)
        )

        season_data = season_data.loc[mask].copy()
        season_data['Dataset'] = dataset_name

        season_table = calculate_season_team_stats(season_data)

        season_tables.append(season_table)
        season_matches.append(season_data)

    all_feature_sets = []

    #Zaczynamy od drugiego sezonu, bo pierwszy nie ma historii
    for season_index in range(1, len(dataset_names)):
        dataset_name = dataset_names[season_index]

        #Historia obejmuje wyłącznie wcześniejsze sezony
        history_tables = season_tables[:season_index]

        historical_stats = aggregate_history(
            history_tables,
            weighted=weighted
        )

        current_matches = season_matches[season_index].copy()

        features = current_matches[
            ['HomeTeam', 'AwayTeam', 'FTR', 'Dataset']
        ].copy()

        # Łączenie danych gospodarza
        home_history = historical_stats[
            ['Team', 'HAS', 'HDS', 'HST', 'HC', 'HY', 'HR']
        ].rename(columns={
            'HAS': 'HAS_home',
            'HDS': 'HDS_home',
            'HST': 'HST_home',
            'HC': 'HC_home',
            'HY': 'HY_home',
            'HR': 'HR_home'
        })

        features = features.merge(
            home_history,
            left_on='HomeTeam',
            right_on='Team',
            how='left'
        ).drop(columns='Team')

        # Łączenie danych gościa
        away_history = historical_stats[
            ['Team', 'AAS', 'ADS', 'AST', 'AC', 'AY', 'AR']
        ].rename(columns={
            'AAS': 'AAS_away',
            'ADS': 'ADS_away',
            'AST': 'AST_away',
            'AC': 'AC_away',
            'AY': 'AY_away',
            'AR': 'AR_away'
        })

        features = features.merge(
            away_history,
            left_on='AwayTeam',
            right_on='Team',
            how='left'
        ).drop(columns='Team')

        # Zmienna docelowa
        features['Result'] = features['FTR'].map({
            'H': 1,
            'D': 0,
            'A': 2
        })

        all_feature_sets.append(features)

    full_features = pd.concat(
        all_feature_sets,
        ignore_index=True
    )

    # Wszystko poza ostatnim sezonem to trening
    last_dataset = dataset_names[-1]

    train_features = full_features[
        full_features['Dataset'] != last_dataset
    ].copy()

    test_features = full_features[
        full_features['Dataset'] == last_dataset
    ].copy()

    return {
        'features': full_features,
        'X_train': train_features.drop(
            columns=[
                'HomeTeam', 'AwayTeam',
                'FTR', 'Dataset', 'Result'
            ]
        ),
        'X_test': test_features.drop(
            columns=[
                'HomeTeam', 'AwayTeam',
                'FTR', 'Dataset', 'Result'
            ]
        ),
        'y_train': train_features['Result'],
        'y_test': test_features['Result'],
        'train_features': train_features,
        'test_features': test_features
    }

def add_season_column(df, league, start_season="2009/2010"):
    # Ustalanie liczby meczów na sezon w zależności od ligi
    matches_per_season = 380 if league != "GB1_09_22" else 306

    # Obliczamy liczbę sezonów
    num_seasons = len(df) // matches_per_season

    # Tworzymy listę nazw sezonów
    seasons = []
    current_season = start_season

    for _ in range(num_seasons):
        seasons.extend([current_season] * matches_per_season)
        start_year, end_year = map(int, current_season.split('/'))
        current_season = f"{end_year}/{end_year + 1}"

    # Dodajemy kolumnę 'Season' do oryginalnego DataFrame
    df = df.copy()
    df['Season'] = seasons
    return df


def find_persistent_teams(df):
    # Wyciągamy wszystkie unikalne sezony w danych
    seasons = df['Season'].unique()

    # Zbieramy drużyny grające jako gospodarze i goście
    home_teams = df[['HomeTeam', 'Season']].rename(columns={'HomeTeam': 'Team'})
    away_teams = df[['AwayTeam', 'Season']].rename(columns={'AwayTeam': 'Team'})

    all_teams = pd.concat([home_teams, away_teams])

    # Grupujemy po nazwie drużyny i liczymy, w ilu unikalnych sezonach grała
    team_season_counts = all_teams.groupby('Team')['Season'].nunique().reset_index()
    team_season_counts.columns = ['Teams', 'Seasons played']


    persistent_teams = team_season_counts[team_season_counts['Seasons played'] == len(seasons)]['Teams'].tolist()

    return team_season_counts, persistent_teams


def plot_team_season_counts(team_season_counts, persistent_teams):
    team_season_counts = team_season_counts.sort_values('Seasons played', ascending=False)

    plt.figure(figsize=(8, 6))
    colors = ['green' if team in persistent_teams else 'darkblue' for team in team_season_counts['Teams']]

    plt.title('Number of seasons played by teams between the 2009/2010 and 2021/2022 seasons')
    plt.xlabel('Teams')
    plt.ylabel('Seasons played')
    plt.grid(axis = 'y')
    plt.xticks(rotation=90)
    #Kropki
    plt.scatter(team_season_counts['Teams'], team_season_counts['Seasons played'], color=colors, s=100, zorder=3)
    plt.vlines(x=team_season_counts['Teams'], ymin=0, ymax=team_season_counts['Seasons played'], linewidth=2, color=colors)
    plt.tight_layout()
    plt.show()
    
#%%
# Lista nazw zbiorów danych
#datasets =  ['ISA_09_10','ISA_10_11', 'EPL_09_10', 'EPL_10_11', 'FL1_09_10', 'FL1_10_11', 'GB1_09_10', 'GB1_10_11', 'SLL_09_10', 'SLL_10_11']# 'ISA_10_11', 'ISA_11_12', 'ISA_12_13', 'ISA_13_14', 'ISA_14_15', 'ISA_15_16']
datasets = ['EPL_09_10', 'EPL_10_11', 'EPL_11_12','EPL_12_13', 'EPL_13_14', 'EPL_14_15', 'EPL_15_16', 'EPL_16_17', 'EPL_17_18', 'EPL_18_19', 'EPL_19_20', 'EPL_20_21', 'EPL_21_22',
             'FL1_09_10', 'FL1_10_11', 'FL1_11_12','FL1_12_13', 'FL1_13_14', 'FL1_14_15', 'FL1_15_16', 'FL1_16_17', 'FL1_17_18', 'FL1_18_19', 'FL1_20_21', 'FL1_21_22',
             'GB1_09_10', 'GB1_10_11', 'GB1_11_12','GB1_12_13', 'GB1_13_14', 'GB1_14_15', 'GB1_15_16', 'GB1_16_17', 'GB1_17_18', 'GB1_18_19', 'GB1_19_20', 'GB1_20_21', 'GB1_21_22',
             'ISA_09_10', 'ISA_10_11', 'ISA_11_12','ISA_12_13', 'ISA_13_14', 'ISA_14_15', 'ISA_15_16', 'ISA_16_17', 'ISA_17_18', 'ISA_18_19', 'ISA_19_20', 'ISA_20_21', 'ISA_21_22',
             'SLL_09_10', 'SLL_10_11', 'SLL_11_12','SLL_12_13', 'SLL_13_14', 'SLL_14_15', 'SLL_15_16', 'SLL_16_17', 'SLL_17_18', 'SLL_18_19', 'SLL_19_20', 'SLL_20_21', 'SLL_21_22']


#%%
# Wczytywanie danych ze wszystkich sezonów dla każdej z lig
EPL_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/english-premier-league_zip/data/EPL_09_22.csv")
FL1_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/french-ligue-1_zip/data/FL1_09_22.csv")
GB1_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/german-bundesliga_zip/data/GB1_09_22.csv")
ISA_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/italian-serie-a_zip/data/ISA_09_22.csv")
SLL_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/spanish-la-liga_zip/data/SLL_09_22.csv")

# Dodawanie kolumny 'Season'
EPL_09_22 = add_season_column(EPL_09_22, 'EPL_09_22')
FL1_09_22 = add_season_column(FL1_09_22, 'FL1_09_22')
GB1_09_22 = add_season_column(GB1_09_22, 'GB1_09_22')
ISA_09_22 = add_season_column(ISA_09_22, 'ISA_09_22')
SLL_09_22 = add_season_column(SLL_09_22, 'SLL_09_22')

# Znajdowanie drużyn grających we wszystkich sezonach
#%%
# Generowanie wykresu
EPL_team_season_counts, EPL_persistent_teams = find_persistent_teams(EPL_09_22)
plot_team_season_counts(EPL_team_season_counts, EPL_persistent_teams)
FL1_team_season_counts, FL1_persistent_teams = find_persistent_teams(FL1_09_22)
plot_team_season_counts(FL1_team_season_counts, FL1_persistent_teams)
GB1_team_season_counts, GB1_persistent_teams = find_persistent_teams(GB1_09_22)
plot_team_season_counts(GB1_team_season_counts, GB1_persistent_teams)
ISA_team_season_counts, ISA_persistent_teams = find_persistent_teams(ISA_09_22)
plot_team_season_counts(ISA_team_season_counts, ISA_persistent_teams)
SLL_team_season_counts, SLL_persistent_teams = find_persistent_teams(SLL_09_22)
plot_team_season_counts(SLL_team_season_counts, SLL_persistent_teams)

#%%
# Wyświetlanie wyników
print('Premier League:', EPL_persistent_teams)
print('Ligue 1:', FL1_persistent_teams)
print('Bundesliga 1:', GB1_persistent_teams)
print('Serie A:', ISA_persistent_teams)
print('La Liga:', SLL_persistent_teams)


# Słownik mapujący ligę na listę trwałych drużyn
persistent_teams_by_league = {
    'EPL': EPL_persistent_teams,
    'FL1': FL1_persistent_teams,
    'GB1': GB1_persistent_teams,
    'ISA': ISA_persistent_teams,
    'SLL': SLL_persistent_teams
}
#%%
from collections import defaultdict

# Grupowanie sezonowych zbiorów danych według ligi
groups = defaultdict(list)

for dataset_name in datasets:
    league_prefix = dataset_name.split('_')[0]
    groups[league_prefix].append(dataset_name)

# Sortowanie chronologiczne sezonów w obrębie każdej ligi
for league_prefix in groups:
    groups[league_prefix] = sorted(groups[league_prefix])

# Kontrola: sprawdź, czy grupy są poprawne
for league_prefix, dataset_names in groups.items():
    print(league_prefix, dataset_names)
#%%
second_approach_simple = {}
second_approach_weighted = {}

for league_prefix, dataset_names in groups.items():

    persistent_teams = persistent_teams_by_league[league_prefix]

    second_approach_simple[league_prefix] = (
        build_second_approach_league(
            dataset_names=dataset_names,
            persistent_teams=persistent_teams,
            weighted=False
        )
    )

    second_approach_weighted[league_prefix] = (
        build_second_approach_league(
            dataset_names=dataset_names,
            persistent_teams=persistent_teams,
            weighted=True
        )
    )
    
#%%
for league_prefix in groups:
    simple = second_approach_simple[league_prefix]
    weighted = second_approach_weighted[league_prefix]

    print('\n' + '=' * 60)
    print(f'Liga: {league_prefix}')

    print('Średnie zwykłe:')
    print('X_train:', simple['X_train'].shape)
    print('X_test :', simple['X_test'].shape)
    print('Braki X_train:', simple['X_train'].isna().sum().sum())
    print('Braki X_test :', simple['X_test'].isna().sum().sum())

    print('Średnie ważone:')
    print('X_train:', weighted['X_train'].shape)
    print('X_test :', weighted['X_test'].shape)
    print('Braki X_train:', weighted['X_train'].isna().sum().sum())
    print('Braki X_test :', weighted['X_test'].isna().sum().sum())
#%%

# Kolejność kolumn statystycznych
stat_columns = [
    'HST', 'AST',
    'HC', 'AC',
    'HY', 'AY',
    'HR', 'AR'
]

# Statystyki gospodarzy liczone z wcześniejszych meczów domowych
home_stat_columns = ['HST', 'HC', 'HY', 'HR']

# Statystyki gości liczone z wcześniejszych meczów wyjazdowych
away_stat_columns = ['AST', 'AC', 'AY', 'AR']


groups = defaultdict(list)

for dataset_name in datasets:
    league_prefix = dataset_name.split('_')[0]
    groups[league_prefix].append(dataset_name)

# Chronologiczne sortowanie
for league_prefix in groups:
    groups[league_prefix] = sorted(groups[league_prefix])

#%%
# Definiowanie modeli i ich parametrów do strojenia
best_teams_evaluation_results_01 = pd.DataFrame(
    columns=[
        'Dataset', 'Model',
        'Accuracy',
        'Recall_weighted', 'Recall_macro',
        'Precision_weighted', 'Precision_macro',
        'F1_weighted', 'F1_macro',
        'Best Params'
    ]
)

best_teams_evaluation_results_02 = pd.DataFrame(
    columns=[
        'Dataset', 'Model',
        'Accuracy',
        'Recall_weighted', 'Recall_macro',
        'Precision_weighted', 'Precision_macro',
        'F1_weighted', 'F1_macro',
        'Best Params'
    ]
)

#%%
all_predictions = []
def evaluate_second_approach(
    prepared_data,
    league_prefix,
    result_table,
    aggregation_name
):
    """
    Trenuje modele na sezonach 2010/2011–2020/2021
    i ocenia je na sezonie 2021/2022.
    """

    X_train = prepared_data['X_train'].copy()
    X_test = prepared_data['X_test'].copy()

    y_train = prepared_data['y_train'].copy()
    y_test = prepared_data['y_test'].copy()

    tscv = TimeSeriesSplit(n_splits=5)

    evaluation_rows = []

    for model_name, (model, params) in models.items():

        print(
            f'Trenowanie {model_name} '
            f'dla ligi {league_prefix}...'
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        pipeline_params = {
            f'model__{param_name}': values
            for param_name, values in params.items()
        }

        if model_name == 'RF':
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=pipeline_params,
                n_iter=100,
                cv=tscv,
                scoring='f1_macro',
                random_state=42,
                n_jobs=-1,
                refit=True
                )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params = search.best_params_

        elif not params:
            best_model = pipeline.fit(X_train, y_train)
            best_params = 'N/A'

        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=pipeline_params,
                cv=tscv,
                scoring='f1_macro',
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params = search.best_params_

        y_pred = best_model.predict(X_test)
        predictions_table = pd.DataFrame({
            'League': league_prefix,
            'Aggregation': aggregation_name,
            'Model': model_name,
            'Actual': y_test.to_numpy(),
            'Predicted': y_pred
            })

        all_predictions.append(predictions_table)
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        new_row = {
            'Dataset': league_prefix,
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),

            'Recall_weighted': recall_score(
                y_test, y_pred,
                average='weighted',
                zero_division=0
            ),

            'Recall_macro': recall_score(
                y_test, y_pred,
                average='macro',
                zero_division=0
            ),

            'Precision_weighted': precision_score(
                y_test, y_pred,
                average='weighted',
                zero_division=0
            ),

            'Precision_macro': precision_score(
                y_test, y_pred,
                average='macro',
                zero_division=0
            ),

            'F1_weighted': f1_score(
                y_test, y_pred,
                average='weighted',
                zero_division=0
            ),

            'F1_macro': f1_score(
                y_test, y_pred,
                average='macro',
                zero_division=0
            ),

            'Best Params': best_params
        }

        evaluation_rows.append(new_row)

    return pd.concat(
        [result_table, pd.DataFrame(evaluation_rows)],
        ignore_index=True
    )

models = {
    'KNN': (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    ),
    'SVM': (
        SVC(),
        {
            'C': np.logspace(-2, 2, 9),
            'gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 9)),
            'kernel': ['linear', 'rbf']
        }
    ),
    'RF': (
        RandomForestClassifier(
            random_state=42
        ),
        {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced']
        }
    ),
    'DT': (
        DecisionTreeClassifier(
            random_state=42
        ),
        {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [2, 4, 8, 12],
            'class_weight': [None, 'balanced']
        }
    ),
    'NB': (
        GaussianNB(),
        {
            'var_smoothing': np.logspace(-12, -6, 7)
        }
    ),
    'NN': (
        MLPClassifier(
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20
        ),
        {
            'hidden_layer_sizes': [
                (10,),
                (20,),
                (30,),
                (40,),
                (20, 10),
                (30, 15)
            ],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.0005, 0.001, 0.01]
        }
    )
}
#%%
for league_prefix, prepared_data in second_approach_simple.items():


    best_teams_evaluation_results_01 = evaluate_second_approach(
        prepared_data=prepared_data,
        league_prefix=league_prefix,
        result_table=best_teams_evaluation_results_01,
        aggregation_name='Mean'
    )

best_teams_evaluation_results_01.to_csv(
    'best_teams_evaluation_results_01.csv',
    index=False,
    encoding='utf-8-sig'
)

for league_prefix, prepared_data in second_approach_weighted.items():


    best_teams_evaluation_results_02 = evaluate_second_approach(
        prepared_data=prepared_data,
        league_prefix=league_prefix,
        result_table=best_teams_evaluation_results_02,
        aggregation_name='Weighted mean'
    )

best_teams_evaluation_results_02.to_csv(
    'best_teams_evaluation_results_02.csv',
    index=False,
    encoding='utf-8-sig'
)

predictions_all = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_all.to_csv(
    'all_test_predictions.csv',
    index=False,
    encoding='utf-8-sig'
)
#%%Wizualizacja

#all_evaluation_results_02.to_csv('all_evaluation_results_02.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Wczytanie wyników (jeśli nie masz już w pamięci)
# best_teams_evaluation_results_01 = pd.read_csv('best_teams_evaluation_results_01.csv')
# best_teams_evaluation_results_02 = pd.read_csv('best_teams_evaluation_results_02.csv')

# Dodanie kolumny z typem agregacji
best_teams_evaluation_results_01['Aggregation'] = 'Mean'
best_teams_evaluation_results_02['Aggregation'] = 'Weighted mean'

# Połączenie obu tabel
eval_all = pd.concat(
    [best_teams_evaluation_results_01, best_teams_evaluation_results_02],
    ignore_index=True
)

# Kolejność lig i modeli
league_order = ['EPL', 'FL1', 'GB1', 'ISA', 'SLL']

model_order = ['KNN', 'SVM', 'RF', 'DT', 'NB', 'NN']

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style='whitegrid',
    context='notebook'
)

#Przygotowanie danych z drugiego podejścia
mean_results = best_teams_evaluation_results_01.copy()
weighted_results = best_teams_evaluation_results_02.copy()

mean_results['Wariant'] = 'Średnia arytmetyczna'
weighted_results['Wariant'] = 'Średnia ważona'

all_results_approach_2 = pd.concat(
    [mean_results, weighted_results],
    ignore_index=True
)

#Średni F1-macro w 5 ligach
plot_df = (
    all_results_approach_2
    .groupby(
        ['Wariant', 'Model'],
        as_index=False,
        observed=True
    )['F1_macro']
    .mean()
)

model_order = [
    'SVM',
    'RF',
    'KNN',
    'NB',
    'DT',
    'NN'
]

variant_order = [
    'Średnia arytmetyczna',
    'Średnia ważona'
]

plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=plot_df,
    x='Model',
    y='F1_macro',
    hue='Wariant',
    order=model_order,
    hue_order=variant_order,
    errorbar=None
)

#Dodanie wartości nad słupkami
for container in ax.containers:
    labels = [
        f'{bar.get_height():.4f}'.replace('.', ',')
        for bar in container
    ]

    ax.bar_label(
        container,
        labels=labels,
        padding=3,
        fontsize=9
    )

ax.set_title(
    'Średnia wartość F1-score modeli w dwóch wariantach drugiego podejścia'
)

ax.set_xlabel('Model')
ax.set_ylabel('Średni Macro F1-score')
ax.set_ylim(0, 0.7)

plt.legend(title='Sposób agregacji danych historycznych')
plt.tight_layout()

plt.savefig(
    'sredni_f1_macro_modele_podejscie_2.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
#%%
league_order = ['EPL', 'FL1', 'GB1', 'ISA', 'SLL']
model_order = ['KNN', 'SVM', 'RF', 'DT', 'NB', 'NN']

for league in league_order:

    league_data = eval_all[
        eval_all['Dataset'] == league
    ].copy()

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=league_data,
        x='Model',
        y='F1_macro',
        hue='Aggregation',
        order=model_order,
        hue_order=['Mean', 'Weighted mean'],
        errorbar=None
    )

    plt.title(
        f'F1-macro for models in {league} '
        f'(second approach)'
    )
    plt.xlabel('Model')
    plt.ylabel('F1-macro')
    plt.ylim(0, 0.65)
    plt.legend(title='Aggregation')
    plt.tight_layout()

    # Opcjonalnie: zapis każdego wykresu do osobnego pliku PNG
    plt.savefig(
        f'F1_macro_models_{league}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
#%%
diff = (
    eval_all
    .pivot_table(
        index=['Dataset', 'Model'],
        columns='Aggregation',
        values='F1_macro'
    )
    .reset_index()
)

# Różnica
diff['Różnica F1-score macro'] = (
    diff['Weighted mean']
    - diff['Mean']
)
league_names = {
    'EPL': 'Premier League',
    'FL1': 'Ligue 1',
    'GB1': 'Bundesliga',
    'ISA': 'Serie A',
    'SLL': 'La Liga'
}

diff['Liga'] = diff['Dataset'].map(league_names)

league_order_polish = [
    'Premier League',
    'Ligue 1',
    'Bundesliga',
    'Serie A',
    'La Liga'
]

plt.figure(figsize=(12, 6))

sns.barplot(
    data=diff,
    x='Liga',
    y='Różnica F1-score macro',
    hue='Model',
    order=league_order_polish,
    errorbar=None
)

#Linia 0
plt.axhline(
    0,
    color='black',
    linewidth=1
)

plt.title(
    'Zmiana wartości F1-score macro po zastosowaniu średniej ważonej danych historycznych'
)

plt.xlabel('Liga')
plt.ylabel(
    'Różnica F1-score macro\n'
    '(średnia ważona − średnia arytmetyczna)'
)

ax.legend(
    title='Model',
    loc='upper right',
    frameon=True,
    fontsize=8,
    title_fontsize=9
)

plt.tight_layout()

plt.savefig(
    'roznica_f1_macro_srednia_wazona_minus_arytmetyczna.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()



#%%
# Obliczenie różnicy bezwzględnej
diff['Różnica F1-score macro'] = (
    diff['Weighted mean']
    - diff['Mean']
)

# Obliczenie zmiany procentowej względem średniej arytmetycznej
diff['Zmiana procentowa'] = (
    (
        diff['Weighted mean']
        - diff['Mean']
    )
    / diff['Mean']
    * 100
)

print(diff[
    [
        'Dataset',
        'Model',
        'Mean',
        'Weighted mean',
        'Różnica F1-score macro',
        'Zmiana procentowa'
    ]
].head())

plt.figure(figsize=(12, 6))

ax_percent = sns.barplot(
    data=diff,
    x='Liga',
    y='Zmiana procentowa',
    hue='Model',
    order=league_order_polish,
    errorbar=None
)

plt.axhline(
    0,
    color='black',
    linewidth=1
)

plt.title(
    'Procentowa zmiana F1-score macro po zastosowaniu\n'
    'średniej ważonej danych historycznych'
)

plt.xlabel('Liga')

plt.ylabel(
    'Zmiana procentowa F1-score macro [%]\n'
    '(średnia ważona względem średniej arytmetycznej)'
)

ax_percent.legend(
    title='Model',
    loc='upper right',
    frameon=True,
    fontsize=8,
    title_fontsize=9
)

#Dodanie wartości procentowych
for container in ax_percent.containers:
    labels = [
        f'{bar.get_height():+.1f}%'.replace('.', ',')
        for bar in container
    ]

    ax_percent.bar_label(
        container,
        labels=labels,
        padding=3,
        fontsize=7,
        rotation=0
    )

plt.tight_layout()

plt.savefig(
    'procentowa_zmiana_f1_macro_srednia_wazona.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
#%%
# Tabela porównawcza
difference = (
    eval_all
    .pivot_table(
        index=['Dataset', 'Model'],
        columns='Aggregation',
        values='F1_macro',
        aggfunc='first'
    )
    .reset_index()
)

# Różnica wmean-mean
difference['Difference'] = (
    difference['Weighted mean']
    - difference['Mean']
)

# Macierz do heatmapy
heatmap_data = (
    difference
    .pivot(
        index='Model',
        columns='Dataset',
        values='Difference'
    )
)

heatmap_league_order = ['EPL', 'FL1', 'GB1', 'ISA', 'SLL']

heatmap_model_order = [
    'SVM',
    'RF',
    'KNN',
    'NB',
    'DT',
    'NN'
]

heatmap_data = (
    heatmap_data
    .reindex(
        index=heatmap_model_order,
        columns=heatmap_league_order
    )
)

league_names = {
    'EPL': 'Premier League',
    'FL1': 'Ligue 1',
    'GB1': 'Bundesliga',
    'ISA': 'Serie A',
    'SLL': 'La Liga'
}

heatmap_data = heatmap_data.rename(
    columns=league_names
)


plt.figure(figsize=(12, 6))

ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".4f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    cbar_kws={
        "label": "Zmiana macro F1-score"
    }
)

ax.set_title(
    "Zmiana F1-score macro po zastosowaniu średniej ważonej"
)

ax.set_xlabel("Liga")
ax.set_ylabel("Model")

plt.tight_layout()
plt.show()

#%%
summary_sizes = []

for league in ['EPL', 'FL1', 'GB1', 'ISA', 'SLL']:

    simple = second_approach_simple[league]
    weighted = second_approach_weighted[league]

    summary_sizes.append({
        'Liga': league,

        'Trening — średnia zwykła': simple['X_train'].shape[0],
        'Test — średnia zwykła': simple['X_test'].shape[0],

        'Trening — średnia ważona': weighted['X_train'].shape[0],
        'Test — średnia ważona': weighted['X_test'].shape[0],

        'Liczba cech': simple['X_train'].shape[1],

        'Sezony treningowe': (
            simple['train_features']['Dataset']
            .nunique()
        ),

        'Sezon testowy': (
            simple['test_features']['Dataset']
            .iloc[0]
        )
    })

dataset_sizes = pd.DataFrame(summary_sizes)

print(dataset_sizes.to_string(index=False))


# %%
# Porównanie modeli według lig — podejście 2

# Wyniki wariantu: średnia zwykła
mean_results = best_teams_evaluation_results_01.copy()

mean_results['Variant'] = 'Średnia zwykła'

# Wyniki wariantu: średnia ważona
weighted_results = best_teams_evaluation_results_02.copy()

weighted_results['Variant'] = 'Średnia ważona'

# Łączenie wariantów
plot_df_approach_2 = pd.concat(
    [mean_results, weighted_results],
    ignore_index=True
)

league_name_mapping = {
    'EPL': 'Premier League',
    'FL1': 'Ligue 1',
    'GB1': 'Bundesliga',
    'ISA': 'Serie A',
    'SLL': 'La Liga'
}

plot_df_approach_2['League_name'] = (
    plot_df_approach_2['Dataset']
    .map(league_name_mapping)
)

league_order = [
    'Premier League',
    'Serie A',
    'Bundesliga',
    'La Liga',
    'Ligue 1'
]

variant_order = [
    'Średnia zwykła',
    'Średnia ważona'
]

model_order = [
    'SVM',
    'RF',
    'KNN',
    'NB',
    'DT',
    'NN'
]
g = sns.catplot(
    data=plot_df_approach_2,
    x='League_name',
    y='F1_macro',
    hue='Model',
    col='Variant',
    kind='bar',
    height=6,
    aspect=1.3,
    errorbar=None,
    order=league_order,
    col_order=variant_order,
    hue_order=model_order
)

g.set_axis_labels(
    'Liga',
    'Średni Macro F1-score'
)

g.set_titles('{col_name}')
g.set(ylim=(0, 0.7))

for ax in g.axes.flat:
    ax.tick_params(
        axis='x',
        rotation=25
    )

g.fig.suptitle(
    'Porównanie modeli w poszczególnych ligach — podejście 2',
    y=1.04
)

plt.show()
#%%
#liga + model + F1-macro dla obu wariantów
comparison_f1_table = (
    eval_all
    .pivot_table(
        index=['Dataset', 'Model'],
        columns='Aggregation',
        values='F1_macro',
        aggfunc='first'
    )
    .reset_index()
)

# Różnica F1

comparison_f1_table['Różnica F1-score macro'] = (
    comparison_f1_table['Weighted mean']
    - comparison_f1_table['Mean']
)


league_names = {
    'EPL': 'Premier League',
    'FL1': 'Ligue 1',
    'GB1': 'Bundesliga',
    'ISA': 'Serie A',
    'SLL': 'La Liga'
}

comparison_f1_table['Liga'] = (
    comparison_f1_table['Dataset']
    .map(league_names)
)


league_order = [
    'Premier League',
    'Ligue 1',
    'Bundesliga',
    'Serie A',
    'La Liga'
]

model_order = [
    'SVM',
    'RF',
    'KNN',
    'NB',
    'DT',
    'NN'
]

comparison_f1_table['Liga'] = pd.Categorical(
    comparison_f1_table['Liga'],
    categories=league_order,
    ordered=True
)

comparison_f1_table['Model'] = pd.Categorical(
    comparison_f1_table['Model'],
    categories=model_order,
    ordered=True
)


comparison_f1_table = (
    comparison_f1_table[
        [
            'Liga',
            'Model',
            'Mean',
            'Weighted mean',
            'Różnica F1-score macro'
        ]
    ]
    .rename(
        columns={
            'Mean': 'F1-score macro — średnia arytmetyczna',
            'Weighted mean': 'F1-score macro — średnia ważona'
        }
    )
    .sort_values(
        ['Liga', 'Model']
    )
    .reset_index(drop=True)
)

f1_columns = [
    'F1-score macro — średnia arytmetyczna',
    'F1-score macro — średnia ważona',
    'Różnica F1-score macro'
]

comparison_f1_table[f1_columns] = (
    comparison_f1_table[f1_columns]
    .round(4)
)

print(comparison_f1_table.to_string(index=False))
#%%
# Tabela średnich wartości F1-score macro dla modeli
# uśrednionych ze wszystkich analizowanych lig

model_f1_summary = (
    comparison_f1_table
    .groupby(
        'Model',
        as_index=False,
        observed=True
    )[
        [
            'F1-score macro — średnia arytmetyczna',
            'F1-score macro — średnia ważona',
            'Różnica F1-score macro'
        ]
    ]
    .mean()
)


model_order = [
    'SVM',
    'RF',
    'KNN',
    'NB',
    'DT',
    'NN'
]

model_f1_summary['Model'] = pd.Categorical(
    model_f1_summary['Model'],
    categories=model_order,
    ordered=True
)

model_f1_summary = (
    model_f1_summary
    .sort_values('Model')
    .reset_index(drop=True)
)

f1_columns = [
    'F1-score macro — średnia arytmetyczna',
    'F1-score macro — średnia ważona',
    'Różnica F1-score macro'
]

model_f1_summary[f1_columns] = (
    model_f1_summary[f1_columns]
    .round(4)
)

print(
    'Średnie wartości F1-score macro modeli '
    'ze wszystkich analizowanych lig:\n'
)

print(model_f1_summary.to_string(index=False))