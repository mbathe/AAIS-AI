
import pandas as pd


def analyze_agricultural_data(df):
    # 1. Copie du DataFrame pour éviter de modifier l'original
    df_clean = df.copy()

    # 2. Conversion des dates en datetime
    df_clean['SDate'] = pd.to_datetime(df_clean['SDate'])
    df_clean['HDate'] = pd.to_datetime(df_clean['HDate'])

    # 3. Création d'une colonne pour la durée de croissance
    df_clean['GrowthDuration'] = (
        df_clean['HDate'] - df_clean['SDate']).dt.days

    # 4. Détection et suppression des anomalies

    # 4.1 Fonction pour détecter les outliers avec IQR
    def detect_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series >= lower_bound) & (series <= upper_bound)

    numeric_cols = ['CropCoveredArea', 'CHeight', 'IrriCount', 'WaterCov', 'ExpYield',
                    'ndvi', 'evi', 'ndwi', 'gndvi', 'savi', 'msavi']

    mask = pd.DataFrame()
    for col in numeric_cols:
        mask[col] = detect_outliers(df_clean[col])

    df_clean = df_clean[mask.all(axis=1)]

    stats_dict = {
        'basic_stats': {},
        'correlations': {},
        'categorical_analysis': {},
        'temporal_analysis': {}
    }

    stats_dict['basic_stats'] = {
        col: {
            'mean': df_clean[col].mean(),
            'median': df_clean[col].median(),
            'std': df_clean[col].std(),
            'min': df_clean[col].min(),
            'max': df_clean[col].max()
        } for col in numeric_cols
    }

    # 5.2 Matrice de corrélation pour les indices de végétation
    vegetation_indices = ['ndvi', 'evi', 'ndwi', 'gndvi', 'savi', 'msavi']
    stats_dict['correlations']['vegetation_indices'] = (
        df_clean[vegetation_indices].corr().round(3)
    )

    # 5.3 Analyse des variables catégorielles
    categorical_cols = ['category', 'IrriType', 'IrriSource']
    for col in categorical_cols:
        stats_dict['categorical_analysis'][col] = df_clean[col].value_counts(
        ).to_dict()

    # 5.4 Analyse temporelle
    stats_dict['temporal_analysis'] = {
        'avg_growth_duration': df_clean['GrowthDuration'].mean(),
        'std_growth_duration': df_clean['GrowthDuration'].std(),
        'planting_months': df_clean['SDate'].dt.month.value_counts().to_dict(),
        'harvest_months': df_clean['HDate'].dt.month.value_counts().to_dict()
    }

    return df_clean, stats_dict


def generate_insights(stats_dict):
    insights = []
    insights.append(f"La durée moyenne de croissance est de {stats_dict['temporal_analysis']['avg_growth_duration']:.1f} jours "
                    f"(écart-type: {stats_dict['temporal_analysis']['std_growth_duration']:.1f} jours)")

    for index in ['ndvi', 'evi', 'ndwi', 'gndvi', 'savi']:
        mean_val = stats_dict['basic_stats'][index]['mean']
        std_val = stats_dict['basic_stats'][index]['std']
        insights.append(f"L'indice {index.upper()} présente une moyenne de {
                        mean_val:.3f} (écart-type: {std_val:.3f})")

    health_dist = stats_dict['categorical_analysis']['category']
    total = sum(health_dist.values())
    for category, count in health_dist.items():
        percentage = (count/total) * 100
        insights.append(f"{category}: {percentage:.1f}% des cultures")
    return insights

    """
    # %%
df_clean, stats = analyze_agricultural_data(data)
insights = generate_insights(stats)
# Afficher les insights
for insight in insights:
    print(insight)

# %%

df_clean, stats = analyze_agricultural_data(data)
insights = generate_insights(stats)
# Afficher les insights
for insight in insights:
    print(insight)
    
    """
