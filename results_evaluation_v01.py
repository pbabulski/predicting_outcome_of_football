
import os
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
#from data import *
import numpy as np
import pandas as pd

#%%
# Wczytywanie danych ze wszystkich sezonów dla każdej z lig
all_evaluation_results_01 = pd.read_csv("D:/CODING/Projects/EPL_prediction/results/all_evaluation_results_01.csv")
all_evaluation_results_02 = pd.read_csv("D:/CODING/Projects/EPL_prediction/results/all_evaluation_results_02.csv")

p1_01 = all_evaluation_results_01.copy()
p1_02 = all_evaluation_results_02.copy()


p1_01['Approach'] = 'Podejście 1'
p1_01['Variant'] = 'Wyniki i metryki bramkowe'

p1_02['Approach'] = 'Podejście 1'
p1_02['Variant'] = 'Wyniki, metryki bramkowe i statystyki'

tables = [p1_01, p1_02]

# %%
# Wyodrębnienie ligi i sezonu

for table in tables:
    table["League"] = table["Dataset"].str.extract(
        r"^(EPL|FL1|GB1|ISA|SLL)",
        expand=False
    )

    table["Season"] = table["Dataset"].str.extract(
        r"_(\d{2}_\d{2})$",
        expand=False
    )

# %%
# Połączenie wyników

all_results = pd.concat(
    tables,
    ignore_index=True
)

print(all_results.shape)
print(all_results.columns)
print(all_results[["Approach", "Variant"]].value_counts())
print(all_results.head())


# %%
# Nazwy lig

league_names = {
    "EPL": "Premier League",
    "FL1": "Ligue 1",
    "GB1": "Bundesliga",
    "ISA": "Serie A",
    "SLL": "La Liga"
}

all_results["League_name"] = (
    all_results["League"].map(league_names)
)

league_order = [
    "Premier League",
    "Serie A",
    "Bundesliga",
    "La Liga",
    "Ligue 1"
]

all_results["League_name"] = pd.Categorical(
    all_results["League_name"],
    categories=league_order,
    ordered=True
)


# %%
# Zapis połączonych wyników

all_results.to_csv(
    "all_results_combined_approach_1.csv",
    index=False,
    encoding="utf-8-sig"
)


# %%
# Średnie według wariantu i modelu

summary_model = (
    all_results
    .groupby(
        ["Approach", "Variant", "Model"],
        as_index=False,
        observed=True
    )
    .agg(
        Accuracy=("Accuracy", "mean"),
        Precision_macro=("Precision_macro", "mean"),
        Recall_macro=("Recall_macro", "mean"),
        F1_macro=("F1_macro", "mean"),
        F1_weighted=("F1_weighted", "mean")
    )
)

print(summary_model.to_string(index=False))


# %%
# Średnie według ligi, modelu i wariantu

summary_league = (
    all_results
    .groupby(
        [
            "Approach",
            "Variant",
            "League_name",
            "Model"
        ],
        as_index=False,
        observed=True
    )
    .agg(
        Accuracy=("Accuracy", "mean"),
        Precision_macro=("Precision_macro", "mean"),
        Recall_macro=("Recall_macro", "mean"),
        F1_macro=("F1_macro", "mean"),
        F1_weighted=("F1_weighted", "mean")
    )
)

print(summary_league.to_string(index=False))


# %%
# Średnie według ligi i wariantu

summary_league_variant = (
    all_results
    .groupby(
        [
            "Approach",
            "Variant",
            "League_name"
        ],
        as_index=False,
        observed=True
    )
    .agg(
        Accuracy=("Accuracy", "mean"),
        Precision_macro=("Precision_macro", "mean"),
        Recall_macro=("Recall_macro", "mean"),
        F1_macro=("F1_macro", "mean")
    )
)

print(summary_league_variant.to_string(index=False))


# %%
# Najlepszy model w każdej lidze i wariancie

best_models = (
    summary_league
    .sort_values(
        [
            "Approach",
            "Variant",
            "League_name",
            "F1_macro"
        ],
        ascending=[True, True, True, False]
    )
    .groupby(
        [
            "Approach",
            "Variant",
            "League_name"
        ],
        as_index=False,
        observed=True
    )
    .first()
)

print(
    best_models[
        [
            "Approach",
            "Variant",
            "League_name",
            "Model",
            "F1_macro"
        ]
    ].to_string(index=False)
)


# %%
# Informacje kontrolne

print(all_results.shape)
print(all_results["Approach"].value_counts())
print(all_results["Variant"].value_counts())
print(all_results["League_name"].value_counts())


# %%
# Tabela rankingowa modeli

summary_f1 = (
    all_results
    .groupby(
        [
            "Approach",
            "Variant",
            "League_name",
            "Model"
        ],
        as_index=False,
        observed=True
    )
    .agg(
        F1_macro=("F1_macro", "mean"),
        Accuracy=("Accuracy", "mean"),
        Precision_macro=("Precision_macro", "mean"),
        Recall_macro=("Recall_macro", "mean"),
        F1_weighted=("F1_weighted", "mean")
    )
)

summary_f1 = summary_f1.sort_values(
    [
        "Approach",
        "Variant",
        "League_name",
        "F1_macro"
    ],
    ascending=[True, True, True, False]
)

print(summary_f1.to_string(index=False))


# %%
# Najlepsze modele w poszczególnych ligach

ranking_models = (
    summary_f1
    .groupby(
        [
            "Approach",
            "Variant",
            "League_name"
        ],
        as_index=False,
        observed=True
    )
    .first()
)

ranking_models = ranking_models[
    [
        "Approach",
        "Variant",
        "League_name",
        "Model",
        "F1_macro"
    ]
]

print(ranking_models.to_string(index=False))


# %%

#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plot_df = (
    all_results
    .groupby(
        ["Variant", "Model"],
        as_index=False,
        observed=True
    )["F1_macro"]
    .mean()
)

plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=plot_df,
    x="Model",
    y="F1_macro",
    hue="Variant",
    errorbar=None
)

# Dodanie wartości nad słupkami
for container in ax.containers:
    labels = [
        f"{bar.get_height():.4f}".replace(".", ",")
        for bar in container
    ]

    ax.bar_label(
        container,
        labels=labels,
        padding=3,
        fontsize=9
    )

ax.set_title(
    "Średnia wartość F1-score modeli "
    "w dwóch wariantach pierwszego podejścia"
)

ax.set_xlabel("Model")
ax.set_ylabel("Macro F1-score")
ax.set_ylim(0, 1)

plt.legend(title="Wariant danych")
plt.tight_layout()
plt.show()

# %%
# Porównanie modeli według lig

plot_df = summary_f1.copy()

g = sns.catplot(
    data=plot_df,
    x="League_name",
    y="F1_macro",
    hue="Model",
    col="Variant",
    kind="bar",
    height=6,
    aspect=1.3,
    errorbar=None,
    order=league_order
)

g.set_axis_labels(
    "Liga",
    "Średni Macro F1-score"
)

g.set_titles("{col_name}")
g.set(ylim=(0, 0.7))

for ax in g.axes.flat:
    ax.tick_params(
        axis="x",
        rotation=25
    )

g.fig.suptitle(
    "Porównanie modeli w poszczególnych ligach — podejście 1",
    y=1.04
)

plt.show()

#%%
import pandas as pd


base_variant = "Wyniki i metryki bramkowe"
extended_variant = "Wyniki, metryki bramkowe i statystyki"

# Dane tylko dla dwóch wariantów Podejścia 1
comparison = summary_f1[
    summary_f1["Variant"].isin(
        [base_variant, extended_variant]
    )
].copy()

# Zaokrąglenie wyników
comparison["F1_macro"] = comparison["F1_macro"].round(4)


# Ranking modeli ogółem w każdym wariancie
overall_ranking = (
    comparison
    .groupby(
        ["Variant", "Model"],
        as_index=False,
        observed=True
    )["F1_macro"]
    .mean()
    .sort_values(
        ["Variant", "F1_macro"],
        ascending=[True, False]
    )
)

print("RANKING MODELI OGÓŁEM")
print(overall_ranking.to_string(index=False))


# Najlepszy model ogółem w każdym wariancie
best_overall = (
    overall_ranking
    .groupby(
        "Variant",
        as_index=False,
        observed=True
    )
    .first()
)

print("\nNAJLEPSZY MODEL OGÓŁEM")
print(best_overall.to_string(index=False))


# Ranking modeli w każdej lidze i wariancie
league_ranking = (
    comparison
    .groupby(
        ["Variant", "League_name", "Model"],
        as_index=False,
        observed=True
    )["F1_macro"]
    .mean()
    .sort_values(
        ["Variant", "League_name", "F1_macro"],
        ascending=[True, True, False]
    )
)

print("\nRANKING MODELI W LIGACH")
print(league_ranking.to_string(index=False))


# Najlepszy model w każdej lidze i wariancie
best_by_league = (
    league_ranking
    .groupby(
        ["Variant", "League_name"],
        as_index=False,
        observed=True
    )
    .first()
)

print("\nNAJLEPSZY MODEL W KAŻDEJ LIDZE")
print(best_by_league.to_string(index=False))

# %%
# Porównanie średnich wartości metryk

metrics_long = summary_model.melt(
    id_vars=["Approach", "Variant", "Model"],
    value_vars=[
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "F1_weighted"
    ],
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(14, 7))

ax = sns.barplot(
    data=metrics_long,
    x="Model",
    y="Value",
    hue="Metric",
    errorbar=None
)

ax.set_title(
    "Średnie wartości metryk oceny modeli — podejście 1"
)

ax.set_xlabel("Model")
ax.set_ylabel("Średnia wartość metryki")
ax.set_ylim(0, 1)

plt.legend(title="Metryka")
plt.tight_layout()
plt.show()


#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# Wybór tabeli wyników z pierwszego podejścia:
# all_evaluation_results_01 -> np. współczynniki siły drużyn
# all_evaluation_results_02 -> np. współczynniki + dodatkowe statystyki
plot_data = all_evaluation_results_02.copy()

league_order = ['EPL', 'FL1', 'GB1', 'ISA', 'SLL']
model_order = ['SVM', 'RF', 'KNN', 'NB', 'DT', 'NN']

# Wyodrębnienie skrótu ligi, np. EPL z EPL_09_10
plot_data['League'] = (
    plot_data['Dataset']
    .str.split('_')
    .str[0]
)

for league in league_order:

    # Wyniki tylko dla aktualnej ligi
    league_data = plot_data[
        plot_data['League'] == league
    ].copy()

    if league_data.empty:
        print(f'Brak danych dla ligi: {league}')
        continue

    # Kolejność sezonów w porządku chronologicznym
    season_order = (
        league_data['Dataset']
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Średni F1-macro z wszystkich sezonów danej ligi, osobno dla każdego modelu
    mean_data = (
        league_data
        .groupby('Model', as_index=False)['F1_macro']
        .mean()
    )

    # Etykieta ostatniej grupy słupków
    mean_label = f'Średnia {league}'

    mean_data['Dataset'] = mean_label
    mean_data['League'] = league

    # Dane dla wykresu: sezony + średnia
    chart_data = pd.concat(
        [league_data, mean_data],
        ignore_index=True
    )

    x_order = season_order + [mean_label]
    x_labels = [
    dataset.replace(f'{league}_', '').replace('_', '/')
    if dataset != mean_label
    else mean_label
    for dataset in x_order
]
    # Wykres
    plt.figure(figsize=(20, 8))

    ax = sns.barplot(
        data=chart_data,
        x='Dataset',
        y='F1_macro',
        hue='Model',
        order=x_order,
        hue_order=model_order,
        errorbar=None
    )

    plt.title(
        f'Porównanie F1-score_macro modeli — {league}',
        fontsize=18
    )
    plt.xlabel('Sezon', fontsize=13)
    plt.ylabel('F1-score_macro', fontsize=13)

    # Wspólna skala dla wszystkich lig
    plt.ylim(0, 1.0)

    plt.xticks(
    ticks=range(len(x_order)),
    labels=x_labels,
    rotation=45,
    ha='right'
)

    plt.legend(
        title='Model',
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    plt.tight_layout()

    # Zapis wykresu do pliku
    plt.savefig(
        f'F1_macro_approach_1_{league}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    

#%%
# Średni Macro F1-score według wariantu i modelu

variant_model_summary = (
    all_results
    .groupby(
        ["Variant", "Model"],
        as_index=False,
        observed=True
    )
    .agg(
        F1_macro=("F1_macro", "mean")
    )
)

# Najlepszy model w każdym wariancie

best_by_variant = (
    variant_model_summary
    .sort_values(
        ["Variant", "F1_macro"],
        ascending=[True, False]
    )
    .groupby(
        "Variant",
        as_index=False,
        observed=True
    )
    .first()
)

best_by_variant["F1_macro"] = (
    best_by_variant["F1_macro"].round(4)
)

print(best_by_variant.to_string(index=False))

# Zmiana Macro F1-score po dodaniu statystyk

variant_comparison = (
    variant_model_summary
    .pivot(
        index="Model",
        columns="Variant",
        values="F1_macro"
    )
    .reset_index()
)

variant_comparison.columns.name = None

variant_comparison = variant_comparison.rename(
    columns={
        "Wyniki i metryki bramkowe": "F1_base",
        "Wyniki, metryki bramkowe i statystyki": "F1_extended"
    }
)

variant_comparison["Delta_F1"] = (
    variant_comparison["F1_extended"]
    - variant_comparison["F1_base"]
)

variant_comparison[
    ["F1_base", "F1_extended", "Delta_F1"]
] = variant_comparison[
    ["F1_base", "F1_extended", "Delta_F1"]
].round(4)

print(variant_comparison.to_string(index=False))


#%%
comparison_pivot = (
    comparison
    .pivot_table(
        index=["League_name", "Model"],
        columns="Variant",
        values="F1_macro",
        aggfunc="mean"
    )
    .reset_index()
)

comparison_pivot.columns.name = None

comparison_pivot = comparison_pivot.rename(
    columns={
        base_variant: "F1_base",
        extended_variant: "F1_extended"
    }
)

comparison_pivot["Delta_F1"] = (
    comparison_pivot["F1_extended"]
    - comparison_pivot["F1_base"]
)

comparison_pivot[
    ["F1_base", "F1_extended", "Delta_F1"]
] = comparison_pivot[
    ["F1_base", "F1_extended", "Delta_F1"]
].round(4)

print("\nRÓŻNICE MIĘDZY WARIANTAMI")
print(
    comparison_pivot
    .sort_values(["League_name", "Delta_F1"], ascending=[True, False])
    .to_string(index=False)
)

plt.show()
#%%
league_order = [
    "Premier League",
    "Serie A",
    "Bundesliga",
    "La Liga",
    "Ligue 1"
]

model_order = [
    "DT",
    "KNN",
    "NB",
    "NN",
    "RF",
    "SVM"
]

# Kopia danych
delta_plot = comparison_pivot.copy()

# Średnia zmiana dla każdego modelu
mean_by_model = (
    delta_plot
    .groupby("Model", as_index=False)["Delta_F1"]
    .mean()
)

mean_by_model["League_name"] = "Średnia"

# Połączenie danych ligowych i średnich
delta_plot_with_mean = pd.concat(
    [
        delta_plot[
            [
                "League_name",
                "Model",
                "Delta_F1"
            ]
        ],
        mean_by_model[
            [
                "League_name",
                "Model",
                "Delta_F1"
            ]
        ]
    ],
    ignore_index=True
)

plot_order = league_order + ["Średnia"]

delta_plot_with_mean["League_name"] = pd.Categorical(
    delta_plot_with_mean["League_name"],
    categories=plot_order,
    ordered=True
)

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.figure(figsize=(16, 7))

ax = sns.barplot(
    data=delta_plot_with_mean,
    x="League_name",
    y="Delta_F1",
    hue="Model",
    order=plot_order,
    hue_order=model_order,
    errorbar=None
)

ax.axhline(
    0,
    color="black",
    linewidth=1
)

ax.set_title(
    "Zmiana F1-score po dodaniu statystyk meczowych"
)

ax.set_xlabel("Liga")
ax.set_ylabel(
    "Zmiana Macro F1-score "
    "(wariant rozszerzony − wariant podstawowy)"
)

plt.xticks(rotation=25)
plt.legend(title="Model")
plt.tight_layout()
plt.show()

#%%
league_order = [
    "Premier League",
    "Serie A",
    "Bundesliga",
    "La Liga",
    "Ligue 1"
]

model_order = [
    "DT",
    "KNN",
    "NB",
    "NN",
    "RF",
    "SVM"
]

# Kopia danych
percent_plot = comparison_pivot.copy()

# Procentowa zmiana względem wariantu podstawowego
percent_plot["Percentage_change"] = (
    (
        percent_plot["F1_extended"]
        - percent_plot["F1_base"]
    )
    / percent_plot["F1_base"]
    * 100
)

# Średnia procentowa zmiana dla każdego modelu
mean_by_model = (
    percent_plot
    .groupby("Model", as_index=False)["Percentage_change"]
    .mean()
)

mean_by_model["League_name"] = "Średnia"

# Połączenie wyników ligowych i średnich
percent_plot_with_mean = pd.concat(
    [
        percent_plot[
            [
                "League_name",
                "Model",
                "Percentage_change"
            ]
        ],
        mean_by_model[
            [
                "League_name",
                "Model",
                "Percentage_change"
            ]
        ]
    ],
    ignore_index=True
)

plot_order = league_order + ["Średnia"]

percent_plot_with_mean["League_name"] = pd.Categorical(
    percent_plot_with_mean["League_name"],
    categories=plot_order,
    ordered=True
)

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.figure(figsize=(16, 7))

ax = sns.barplot(
    data=percent_plot_with_mean,
    x="League_name",
    y="Percentage_change",
    hue="Model",
    order=plot_order,
    hue_order=model_order,
    errorbar=None
)

# Linia oznaczająca brak zmiany
ax.axhline(
    0,
    color="black",
    linewidth=1
)

# Wartości procentowe na słupkach
for container in ax.containers:
    labels = []

    for bar in container:
        height = bar.get_height()

        if pd.isna(height):
            labels.append("")
        else:
            labels.append(
                f"{height:+.1f}%".replace(".", ",")
            )

    ax.bar_label(
        container,
        labels=labels,
        padding=3,
        fontsize=8
    )

ax.set_title(
    "Procentowa zmiana F1-score "
    "po dodaniu statystyk meczowych"
)

ax.set_xlabel("Liga")
ax.set_ylabel(
    "Zmiana względem wariantu podstawowego (%)"
)

plt.xticks(rotation=25)
plt.legend(title="Model")
plt.tight_layout()
plt.show()
#%%
heatmap_data = comparison_pivot.pivot(
    index="Model",
    columns="League_name",
    values="Delta_F1"
)

heatmap_data = heatmap_data.reindex(
    index=model_order,
    columns=league_order
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
    "Zmiana  F1-score po dodaniu statystyk meczowych"
)

ax.set_xlabel("Liga")
ax.set_ylabel("Model")

plt.tight_layout()
plt.show()
