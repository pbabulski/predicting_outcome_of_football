
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

#for ploting
import matplotlib.pyplot as plt
import seaborn as sns

#features extracting
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%% List of datasets
datasets = ['EPL_09_10', 'EPL_10_11', 'EPL_11_12','EPL_12_13', 'EPL_13_14', 'EPL_14_15', 'EPL_15_16', 'EPL_16_17', 'EPL_17_18', 'EPL_18_19', 'EPL_19_20', 'EPL_20_21', 'EPL_21_22',
            'FL1_09_10', 'FL1_10_11', 'FL1_11_12','FL1_12_13', 'FL1_13_14', 'FL1_14_15', 'FL1_15_16', 'FL1_16_17', 'FL1_17_18', 'FL1_18_19', 'FL1_20_21', 'FL1_21_22',
            'GB1_09_10', 'GB1_10_11', 'GB1_11_12','GB1_12_13', 'GB1_13_14', 'GB1_14_15', 'GB1_15_16', 'GB1_16_17', 'GB1_17_18', 'GB1_18_19', 'GB1_19_20', 'GB1_20_21', 'GB1_21_22',
            'ISA_09_10', 'ISA_10_11', 'ISA_11_12','ISA_12_13', 'ISA_13_14', 'ISA_14_15', 'ISA_15_16', 'ISA_16_17', 'ISA_17_18', 'ISA_18_19', 'ISA_19_20', 'ISA_20_21', 'ISA_21_22',
            'SLL_09_10', 'SLL_10_11', 'SLL_11_12','SLL_12_13', 'SLL_13_14', 'SLL_14_15', 'SLL_15_16', 'SLL_16_17', 'SLL_17_18', 'SLL_18_19', 'SLL_19_20', 'SLL_20_21', 'SLL_21_22']

#%%
columns_req = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']
columns_sm = ['FTR','HST','AST','HF','AF']

#%%#*****English Premier League*****

EPL_09_10_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-0910_csv.csv")
EPL_09_10 = EPL_09_10_all[columns_req]
print(EPL_09_10.shape)

EPL_10_11_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1011_csv.csv")
EPL_10_11 = EPL_10_11_all[columns_req]
print(EPL_10_11.shape)

EPL_11_12_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1112_csv.csv")
EPL_11_12 = EPL_11_12_all[columns_req]
print(EPL_11_12.shape)

EPL_12_13_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1213_csv.csv")
EPL_12_13 = EPL_12_13_all[columns_req]
print(EPL_12_13.shape)

EPL_13_14_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1314_csv.csv")
EPL_13_14 = EPL_13_14_all[columns_req]
print(EPL_13_14.shape)

EPL_14_15_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1415_csv.csv")
EPL_14_15 = EPL_14_15_all[columns_req]
print(EPL_14_15.shape)

EPL_15_16_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1516_csv.csv")
EPL_15_16 = EPL_15_16_all[columns_req]
print(EPL_15_16.shape)

EPL_16_17_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1617_csv.csv")
EPL_16_17 = EPL_16_17_all[columns_req]
print(EPL_16_17.shape)

EPL_17_18_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1718_csv.csv")
EPL_17_18 = EPL_17_18_all[columns_req]
print(EPL_17_18.shape)

EPL_18_19_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1819_csv.csv")
EPL_18_19 = EPL_18_19_all[columns_req]
print(EPL_18_19.shape)

EPL_19_20_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-1920_csv.csv")
EPL_19_20 = EPL_19_20_all[columns_req]
print(EPL_19_20.shape)

EPL_20_21_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-2021_csv.csv")
EPL_20_21 = EPL_20_21_all[columns_req]
print(EPL_20_21.shape)

EPL_21_22_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\english-premier-league_zip\data\season-2122_csv.csv")
EPL_21_22 = EPL_21_22_all[columns_req]
print(EPL_21_22.shape)

#Connection data sets
EPL_09_22 = pd.concat([EPL_09_10, EPL_10_11, EPL_11_12, EPL_12_13, EPL_13_14, EPL_14_15, EPL_15_16, EPL_16_17, 
                       EPL_17_18, EPL_18_19, EPL_19_20, EPL_20_21, EPL_21_22])
print(EPL_09_22.shape)
EPL_09_22.to_csv('EPL_09_22.csv', index=True, header=True)

#%%*****French Ligue 1*****

FL1_09_10_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-0910_csv.csv")
FL1_09_10 = FL1_09_10_all[columns_req]
print(FL1_09_10.shape)

FL1_10_11_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1011_csv.csv")
FL1_10_11 = FL1_10_11_all[columns_req]
print(FL1_10_11.shape)

FL1_11_12_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1112_csv.csv")
FL1_11_12 = FL1_11_12_all[columns_req]
print(FL1_11_12.shape)

FL1_12_13_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1213_csv.csv")
FL1_12_13 = FL1_12_13_all[columns_req]
print(FL1_12_13.shape)

FL1_13_14_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1314_csv.csv")
FL1_13_14 = FL1_13_14_all[columns_req]
print(FL1_13_14.shape)

FL1_14_15_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1415_csv.csv")
FL1_14_15 = FL1_14_15_all[columns_req]
print(FL1_14_15.shape)

FL1_15_16_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1516_csv.csv")
FL1_15_16 = FL1_15_16_all[columns_req]
print(FL1_15_16.shape)
#%%
FL1_16_17_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1617_csv.csv")


#%% Uzupełnianie braków srednimi

# Lista kolumn statystyk meczowych do uzupełnienia
stats_columns = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

df = FL1_16_17_all[columns_req]

# # Obliczenie średnich statystyk dla każdej drużyny (osobno dla meczów u siebie i na wyjeździe)
home_stats_mean = df.groupby('HomeTeam')[stats_columns].mean().round(0)
away_stats_mean = df.groupby('AwayTeam')[stats_columns].mean().round(0)

# Funkcja do uzupełniania brakujących wartości średnimi statystykami
def fill_missing_stats(row):
    if pd.isnull(row['HS']):  # Jeśli są braki, zakładamy, że to walkower
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Uzupełnianie statystyk średnimi wartościami
        for stat in stats_columns:
            if 'H' in stat:  # Statystyka gospodarzy
                row[stat] = home_stats_mean.loc[home_team, stat]
            else:  # Statystyka gości
                row[stat] = away_stats_mean.loc[away_team, stat]
    return row

# Zastosowanie funkcji do uzupełnienia danych
df = df.apply(fill_missing_stats, axis=1)

# Sprawdzenie, czy nadal są braki
df[stats_columns].isnull().sum()

FL1_16_17 = df

#%%*****French Ligue 1*****

FL1_17_18_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1718_csv.csv")
FL1_17_18 = FL1_17_18_all[columns_req]
print(FL1_17_18.shape)

FL1_18_19_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1819_csv.csv")
FL1_18_19 = FL1_18_19_all[columns_req]
print(FL1_18_19.shape)

#FL1_19_20_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-1920_csv.csv")
#FL1_19_20 = FL1_19_20_all[columns_req]
#print(FL1_19_20.shape)

FL1_20_21_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-2021_csv.csv")
FL1_20_21 = FL1_20_21_all[columns_req]
print(FL1_20_21.shape)

FL1_21_22_all = pd.read_csv(r"D:/CODING/Projects/EPL_prediction\french-ligue-1_zip\data\season-2122_csv.csv")
FL1_21_22 = FL1_21_22_all[columns_req]
print(FL1_21_22.shape)

#Connection data sets without 19/20
FL1_09_22 = pd.concat([FL1_09_10, FL1_10_11, FL1_11_12, FL1_12_13, FL1_13_14, FL1_14_15, FL1_15_16, FL1_16_17, 
                       FL1_17_18, FL1_18_19, FL1_20_21, FL1_21_22])
print(FL1_09_22.shape)
FL1_09_22.to_csv('FL1_09_22.csv', index=True, header=True)

#%%*****German Bundesliga*****

GB1_09_10_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-0910_csv.csv")
GB1_09_10 = GB1_09_10_all[columns_req]
print(GB1_09_10.shape)

GB1_10_11_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1011_csv.csv")
GB1_10_11 = GB1_10_11_all[columns_req]
print(GB1_10_11.shape)

GB1_11_12_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1112_csv.csv")
GB1_11_12 = GB1_11_12_all[columns_req]
print(GB1_11_12.shape)

GB1_12_13_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1213_csv.csv")
GB1_12_13 = GB1_12_13_all[columns_req]
print(GB1_12_13.shape)

GB1_13_14_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1314_csv.csv")
GB1_13_14 = GB1_13_14_all[columns_req]
print(GB1_13_14.shape)

GB1_14_15_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1415_csv.csv")
GB1_14_15 = GB1_14_15_all[columns_req]
print(GB1_14_15.shape)

GB1_15_16_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1516_csv.csv")
GB1_15_16 = GB1_15_16_all[columns_req]
print(GB1_15_16.shape)

GB1_16_17_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1617_csv.csv")
GB1_16_17 = GB1_16_17_all[columns_req]
print(GB1_16_17.shape)

GB1_17_18_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1718_csv.csv")
GB1_17_18 = GB1_17_18_all[columns_req]
print(GB1_17_18.shape)

GB1_18_19_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1819_csv.csv")
GB1_18_19 = GB1_18_19_all[columns_req]
print(GB1_18_19.shape)

GB1_19_20_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-1920_csv.csv")
GB1_19_20 = GB1_19_20_all[columns_req]
print(GB1_19_20.shape)

GB1_20_21_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-2021_csv.csv")
GB1_20_21 = GB1_20_21_all[columns_req]
print(GB1_20_21.shape)

GB1_21_22_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\german-bundesliga_zip\data\season-2122_csv.csv")
GB1_21_22 = GB1_21_22_all[columns_req]
print(GB1_21_22.shape)

#Connection data sets
GB1_09_22 = pd.concat([GB1_09_10, GB1_10_11, GB1_11_12, GB1_12_13, GB1_13_14, GB1_14_15, GB1_15_16, GB1_16_17, 
                       GB1_17_18, GB1_18_19, GB1_19_20, GB1_20_21, GB1_21_22])
print(GB1_09_22.shape)
GB1_09_22.to_csv('GB1_09_22.csv', index=True, header=True)

#%%*****Italian SERIE A*****

ISA_09_10_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-0910_csv.csv")
ISA_09_10 = ISA_09_10_all[columns_req]
print(ISA_09_10.shape)

ISA_10_11_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1011_csv.csv")
ISA_10_11 = ISA_10_11_all[columns_req]
print(ISA_10_11.shape)

ISA_11_12_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1112_csv.csv")
ISA_11_12 = ISA_11_12_all[columns_req]
print(ISA_11_12.shape)

ISA_12_13_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1213_csv.csv")
# ISA_12_13 = ISA_12_13_all[columns_req]
# print(ISA_12_13.shape)

#%% Uzupełnianie braków srednimi

# Lista kolumn statystyk meczowych do uzupełnienia
stats_columns = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

df = ISA_12_13_all[columns_req]

# # Obliczenie średnich statystyk dla każdej drużyny (osobno dla meczów u siebie i na wyjeździe)
home_stats_mean = df.groupby('HomeTeam')[stats_columns].mean().round(0)
away_stats_mean = df.groupby('AwayTeam')[stats_columns].mean().round(0)

# Funkcja do uzupełniania brakujących wartości średnimi statystykami
def fill_missing_stats(row):
    if pd.isnull(row['HS']):  # Jeśli są braki, zakładamy, że to walkower
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Uzupełnianie statystyk średnimi wartościami
        for stat in stats_columns:
            if 'H' in stat:  # Statystyka gospodarzy
                row[stat] = home_stats_mean.loc[home_team, stat]
            else:  # Statystyka gości
                row[stat] = away_stats_mean.loc[away_team, stat]
    return row

# Zastosowanie funkcji do uzupełnienia danych
df = df.apply(fill_missing_stats, axis=1)

# Sprawdzenie, czy nadal są braki
df[stats_columns].isnull().sum()

ISA_12_13 = df

#%%*****Italian SERIE A*****

ISA_13_14_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1314_csv.csv")
ISA_13_14 = ISA_13_14_all[columns_req]
print(ISA_13_14.shape)

ISA_14_15_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1415_csv.csv")
ISA_14_15 = ISA_14_15_all[columns_req]
print(ISA_14_15.shape)

ISA_15_16_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1516_csv.csv")
ISA_15_16 = ISA_15_16_all[columns_req]
print(ISA_15_16.shape)

ISA_16_17_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1617_csv.csv")
#ISA_16_17 = ISA_16_17_all[columns_req]
#print(ISA_16_17.shape)

#%% Uzupełnianie braków srednimi

# Lista kolumn statystyk meczowych do uzupełnienia
stats_columns = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

df = ISA_16_17_all[columns_req]

# # Obliczenie średnich statystyk dla każdej drużyny (osobno dla meczów u siebie i na wyjeździe)
home_stats_mean = df.groupby('HomeTeam')[stats_columns].mean().round(0)
away_stats_mean = df.groupby('AwayTeam')[stats_columns].mean().round(0)

# Funkcja do uzupełniania brakujących wartości średnimi statystykami
def fill_missing_stats(row):
    if pd.isnull(row['HS']):  # Jeśli są braki, zakładamy, że to walkower
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Uzupełnianie statystyk średnimi wartościami
        for stat in stats_columns:
            if 'H' in stat:  # Statystyka gospodarzy
                row[stat] = home_stats_mean.loc[home_team, stat]
            else:  # Statystyka gości
                row[stat] = away_stats_mean.loc[away_team, stat]
    return row

# Zastosowanie funkcji do uzupełnienia danych
df = df.apply(fill_missing_stats, axis=1)

# Sprawdzenie, czy nadal są braki
df[stats_columns].isnull().sum()

ISA_16_17 = df

#%%*****Italian SERIE A*****

ISA_17_18_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1718_csv.csv")
ISA_17_18 = ISA_17_18_all[columns_req]
print(ISA_17_18.shape)

ISA_18_19_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1819_csv.csv")
ISA_18_19 = ISA_18_19_all[columns_req]
print(ISA_18_19.shape)

ISA_19_20_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-1920_csv.csv")
ISA_19_20 = ISA_19_20_all[columns_req]
print(ISA_19_20.shape)

ISA_20_21_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-2021_csv.csv")
ISA_20_21 = ISA_20_21_all[columns_req]
print(ISA_20_21.shape)

ISA_21_22_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\italian-serie-a_zip\data\season-2122_csv.csv")
ISA_21_22 = ISA_21_22_all[columns_req]
print(ISA_21_22.shape)

#Connection data sets
ISA_09_22 = pd.concat([ISA_09_10, ISA_10_11, ISA_11_12, ISA_12_13, ISA_13_14, ISA_14_15, ISA_15_16, ISA_16_17, 
                       ISA_17_18, ISA_18_19, ISA_19_20, ISA_20_21, ISA_21_22])
print(ISA_09_22.shape)
ISA_09_22.to_csv('ISA_09_22.csv', index=True, header=True)

#%%*****Spanish La Liga*****

SLL_09_10_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-0910_csv.csv")
SLL_09_10 = SLL_09_10_all[columns_req]
print(SLL_09_10.shape)

SLL_10_11_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1011_csv.csv")
SLL_10_11 = SLL_10_11_all[columns_req]
print(SLL_10_11.shape)

SLL_11_12_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1112_csv.csv")
SLL_11_12 = SLL_11_12_all[columns_req]
print(SLL_11_12.shape)

SLL_12_13_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1213_csv.csv")
SLL_12_13 = SLL_12_13_all[columns_req]
print(SLL_12_13.shape)

SLL_13_14_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1314_csv.csv")
SLL_13_14 = SLL_13_14_all[columns_req]
print(SLL_13_14.shape)

SLL_14_15_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1415_csv.csv")
SLL_14_15 = SLL_14_15_all[columns_req]
print(SLL_14_15.shape)

SLL_15_16_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1516_csv.csv")
SLL_15_16 = SLL_15_16_all[columns_req]
print(SLL_15_16.shape)

SLL_16_17_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1617_csv.csv")
SLL_16_17 = SLL_16_17_all[columns_req]
print(SLL_16_17.shape)

SLL_17_18_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1718_csv.csv")
SLL_17_18 = SLL_17_18_all[columns_req]
print(SLL_17_18.shape)

SLL_18_19_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1819_csv.csv")
SLL_18_19 = SLL_18_19_all[columns_req]
print(SLL_18_19.shape)

SLL_19_20_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-1920_csv.csv")
SLL_19_20 = SLL_19_20_all[columns_req]
print(SLL_19_20.shape)

SLL_20_21_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-2021_csv.csv")
SLL_20_21 = SLL_20_21_all[columns_req]
print(SLL_20_21.shape)

SLL_21_22_all = pd.read_csv("D:/CODING/Projects/EPL_prediction\spanish-la-liga_zip\data\season-2122_csv.csv")
SLL_21_22 = SLL_21_22_all[columns_req]
print(SLL_21_22.shape)

#Connection data sets
SLL_09_22 = pd.concat([SLL_09_10, SLL_10_11, SLL_11_12, SLL_12_13, SLL_13_14, SLL_14_15, SLL_15_16, SLL_16_17, 
                       SLL_17_18, SLL_18_19, SLL_19_20, SLL_20_21, SLL_21_22])
print(SLL_09_22.shape)
SLL_09_22.to_csv('SLL_09_22.csv', index=True, header=True)

#%%
# Wczytywanie danych ze wszystkich sezonów dla każdej z lig
EPL_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/english-premier-league_zip/data/EPL_09_22.csv")
FL1_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/french-ligue-1_zip/data/FL1_09_22.csv")
GB1_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/german-bundesliga_zip/data/GB1_09_22.csv")
ISA_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/italian-serie-a_zip/data/ISA_09_22.csv")
SLL_09_22 = pd.read_csv("D:/CODING/Projects/EPL_prediction/spanish-la-liga_zip/data/SLL_09_22.csv")

#%%
import numpy as np
df = FL1_09_22
nan_positions = np.where(df.isnull())
for row, col in zip(*nan_positions):
    print(f"NaN w wierszu {row}, kolumna '{df.columns[col]}'")


#%% Wybieranie istotnych cech ##################################################################################################
# Lista nazw zbiorów danych
datasets =  ['EPL_09_22', 'FL1_09_22', 'GB1_09_22','ISA_09_22', 'SLL_09_22']

# Słownik do przechowywania wyników dla każdego zbioru danych
results = {}

# Pętla przez każdy zbiór danych
for dataset_name in datasets:
    # Pobierz dane
    data = globals()[dataset_name]  # Użyj globals(), aby uzyskać zmienną po nazwie
    
    # Tworzenie tabeli
    table = pd.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS'))
    
    # Obliczenia średnich
    avg_home_scored = data.FTHG.sum() / 380.0
    avg_away_scored = data.FTAG.sum() / 380.0
    avg_home_conceded = avg_away_scored
    avg_away_conceded = avg_home_scored
    
    # Grupowanie danych
    res_home = data.groupby('HomeTeam')
    res_away = data.groupby('AwayTeam')
    
    # Wypełnianie tabeli
    table.Team = list(res_home.groups.keys())
    table.HGS = res_home.FTHG.sum().values
    table.HGC = res_home.FTAG.sum().values
    table.AGS = res_away.FTAG.sum().values
    table.AGC = res_away.FTHG.sum().values
    table.HAS = (table.HGS / 19.0) / avg_home_scored  # HAS home attack strength
    table.AAS = (table.AGS / 19.0) / avg_away_scored  # AAS away attack strength
    table.HDS = (table.HGC / 19.0) / avg_home_conceded  # HDS home defence strength
    table.ADS = (table.AGC / 19.0) / avg_away_conceded  # ADS away defence strength
    
    # Przygotowanie cech
    #features = data[['HomeTeam', 'AwayTeam', 'FTR']]
    #features = data[['HomeTeam','AwayTeam','FTR', 'HST', 'AST', 'HY','AY','HR','AR']] 
    features = data[['HomeTeam','AwayTeam','FTR','HST', 'AST', 'HC', 'AC','HY','AY','HR','AR']] 
    f_HAS = []
    f_HDS = []
    f_AAS = []
    f_ADS = []
    for index, row in features.iterrows():
        f_HAS.append(table[table['Team'] == row['HomeTeam']]['HAS'].values[0])
        f_HDS.append(table[table['Team'] == row['HomeTeam']]['HDS'].values[0])
        f_AAS.append(table[table['Team'] == row['AwayTeam']]['AAS'].values[0])
        f_ADS.append(table[table['Team'] == row['AwayTeam']]['ADS'].values[0])
    
    features['HAS'] = f_HAS
    features['HDS'] = f_HDS
    features['AAS'] = f_AAS
    features['ADS'] = f_ADS
    features['Result'] = features[['FTR']].replace(['H', 'A', 'D'], [1, 2, 0])
    y = features["Result"]
    
    # Zapisz wyniki do słownika
    results[dataset_name] = {
        'table': table,
        'features': features,
        'y': y
    }
#%%
# Wczytanie danych

features = results['EPL_09_22']['features']
y = results['EPL_09_22']['y']
X = features.drop(columns=['HomeTeam', 'AwayTeam', 'FTR', 'Result', 'HAS', 'AAS', 'HDS', 'ADS'])  # Usuwamy kolumny niepotrzebne do modelowania

#%%1. Scatter matrix

pd.plotting.scatter_matrix(X, figsize = (25,25)) 
plt.show() 

#%%2. Macierz korelacji

plt.figure(figsize=(20,10)) 
sns.heatmap(X.corr(), annot= True)

#%%3. Lasso

# Standaryzacja danych (LASSO wymaga danych w tej samej skali)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regresja LASSO z doborem najlepszego alpha
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
LassoCV().fit
# Pobranie istotnych cech
importance = np.abs(lasso.coef_)
selected_features = X.columns[importance > 0]

# Wyniki
print("Selected features by LASSO:", selected_features.tolist())

# Wizualizacja istotności cech
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=importance, palette="coolwarm")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.ylabel("Lasso")
plt.title("The features importance according to LASSO")
plt.show()

#%%4. VIF - variance_inflation_factor

# Obliczenie VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("VIF dla cech:")
print(vif_data)

# Usunięcie cech o wysokim VIF
selected_features = vif_data[vif_data['VIF'] < 10]['Feature']
X_selected = X[selected_features]

print("\nWybrane cechy:", selected_features.tolist())

#Wykres wartosci VIF
plt.figure(figsize=(10, 6))
sns.barplot(x=vif_data['Feature'], y=vif_data['VIF'], palette="coolwarm")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.title("VIF value for individual features")
plt.show()


#%%
#Ploting distibution of WINS, DRAWS and LOSES by host team
results = pd.DataFrame(columns=['Home', 'Draw', 'Away'], index = ['EPL', 'FL1', 'GB1', 'ISA', 'SLL'])
results.loc['EPL'] = [((EPL_09_22['FTR'] == 'H').sum())/len(EPL_09_22)*100, ((EPL_09_22['FTR'] == 'D').sum())/len(EPL_09_22)*100, ((EPL_09_22['FTR'] == 'A').sum())/len(EPL_09_22)*100]
results.loc['FL1'] = [((FL1_09_22['FTR'] == 'H').sum())/len(FL1_09_22)*100, ((FL1_09_22['FTR'] == 'D').sum())/len(FL1_09_22)*100, ((FL1_09_22['FTR'] == 'A').sum())/len(FL1_09_22)*100]
results.loc['GB1'] = [((GB1_09_22['FTR'] == 'H').sum())/len(GB1_09_22)*100, ((GB1_09_22['FTR'] == 'D').sum())/len(GB1_09_22)*100, ((GB1_09_22['FTR'] == 'A').sum())/len(GB1_09_22)*100]
results.loc['ISA'] = [((ISA_09_22['FTR'] == 'H').sum())/len(ISA_09_22)*100, ((ISA_09_22['FTR'] == 'D').sum())/len(ISA_09_22)*100, ((ISA_09_22['FTR'] == 'A').sum())/len(ISA_09_22)*100]
results.loc['SLL'] = [((SLL_09_22['FTR'] == 'H').sum())/len(SLL_09_22)*100, ((SLL_09_22['FTR'] == 'D').sum())/len(SLL_09_22)*100, ((SLL_09_22['FTR'] == 'A').sum())/len(SLL_09_22)*100]

df = pd.DataFrame(results, index = ['EPL', 'FL1', 'GB1', 'ISA', 'SLL'],
                  columns=['Home', 'Draw', 'Away'])

ax = df.plot(kind='bar', figsize=(12,6))


plt.title('Percentage of wins, draws and loses of host team from season 2009-2010 till 2021-2022')
plt.xlabel("Leagues")
plt.xticks(rotation=45)
plt.ylabel("Results[%]")
plt.grid(axis = 'y')

# Zwiększenie zakresu osi Y
max_value = df['Home'].max()  # Maksymalna wartość w kolumnie 'Home'
plt.ylim(0, max_value * 1.1)  # Zwiększenie górnego limitu o 10%

# Dodawanie wartości na słupkach dla kolumny 'Home'
for i, (index, row) in enumerate(df.iterrows()):
    ax.text(i-0.2, row['Home'] + 1, f'{row["Home"]:.1f}%', ha='center', va='bottom')

plt.show(block=True)
