import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Prétraite un dataframe en gérant les valeurs manquantes, infinies et aberrantes,
    et en convertissant les colonnes de dates au format datetime.
    
    Parameters:
    df (pandas.DataFrame): Le dataframe d'entrée à prétraiter.
    
    Returns:
    pandas.DataFrame: Le dataframe prétraité avec les valeurs manquantes et aberrantes traitées
                     et les colonnes de dates converties au format datetime.
    dict: Dictionnaire contenant les statistiques sur les anomalies trouvées
    """
    anomalies = {}
    df_clean = df.copy()

    # Compter les valeurs manquantes initiales
    missing_values = df_clean.isnull().sum()
    anomalies["initial_missing_values"] = missing_values.to_dict()

    # Remplacer les valeurs infinies par NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Traiter les colonnes numériques
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in ['tif_path', 'dataset']:
            continue

        # Calculer la moyenne des valeurs non-NaN et non-aberrantes
        non_null_values = df_clean[col].dropna()

        if len(non_null_values) > 0:
            # Calculer les z-scores pour identifier les valeurs aberrantes
            z_scores = np.abs(stats.zscore(non_null_values))
            outlier_mask = z_scores > 3

            # Compter les outliers
            anomalies[f"{col}_outliers_count"] = outlier_mask.sum()

            # Calculer la moyenne en excluant les valeurs aberrantes
            clean_mean = non_null_values[~outlier_mask].mean()

            # Si toutes les valeurs sont des outliers, utiliser la médiane
            if pd.isna(clean_mean):
                clean_mean = non_null_values.median()

            # Remplacer uniquement les valeurs aberrantes par la moyenne
            df_clean.loc[df_clean.index[outlier_mask], col] = clean_mean

            # Remplacer les NaN par la moyenne
            df_clean[col] = df_clean[col].fillna(clean_mean)

    # Traiter les colonnes catégorielles
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in ['tif_path', 'dataset']:
            continue

        # Calculer le mode en excluant les NaN
        mode_value = df_clean[col].mode()[0]

        # Remplacer uniquement les NaN par le mode
        df_clean[col] = df_clean[col].fillna(mode_value)

        # Enregistrer le nombre de remplacements
        anomalies[f"{col}_missing_replaced"] = missing_values[col]

    # Traiter les colonnes de dates
    date_columns = ['SDate', 'HDate']
    for col in date_columns:
        if col in df_clean.columns:
            # Convertir en datetime
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

            # Calculer la médiane des dates non-NaN
            median_date = df_clean[col].dropna().median()

            # Remplacer uniquement les NaN par la médiane
            df_clean[col] = df_clean[col].fillna(median_date)

            # Extraire les composantes de la date
            df_clean[f'{col}_year'] = df_clean[col].dt.year
            df_clean[f'{col}_month'] = df_clean[col].dt.month
            df_clean[f'{col}_day'] = df_clean[col].dt.day

            # Enregistrer le nombre de dates manquantes traitées
            anomalies[f"{col}_missing_dates_replaced"] = missing_values.get(
                col, 0)

    # Compter les valeurs manquantes finales pour vérification
    final_missing_values = df_clean.isnull().sum()
    anomalies["final_missing_values"] = final_missing_values.to_dict()

    return df_clean, anomalies
