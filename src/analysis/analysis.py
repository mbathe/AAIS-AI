
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def analyze_data_quality(df):

    print("=" * 80)
    print("RAPPORT D'ANALYSE DE LA QUALITÉ DES DONNÉES")
    print("=" * 80)

    # 1. Analyse des valeurs manquantes
    print("\n1. ANALYSE DES VALEURS MANQUANTES")
    print("-" * 40)
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    if missing_values.sum() > 0:
        print("\nColonnes avec valeurs manquantes:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"{col}: {missing} valeurs manquantes ({
                  missing_percentages[col]:.2f}%)")
    else:
        print("Aucune valeur manquante trouvée dans le dataset")

    # 2. Analyse des doublons
    print("\n2. ANALYSE DES DOUBLONS")
    print("-" * 40)
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    print(f"Nombre total de lignes dupliquées: {len(duplicate_rows)}")

    print("\n3. ANALYSE DES VALEURS ABERRANTES")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        if len(outliers) > 0:
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'min': outliers.min(),
                'max': outliers.max()
            }

    if outliers_summary:
        print("\nValeurs aberrantes détectées:")
        for col, stats in outliers_summary.items():
            print(f"\n{col}:")
            print(f"  - Nombre de valeurs aberrantes: {stats['count']}")
            print(f"  - Pourcentage: {stats['percentage']:.2f}%")
            print(
                f"  - Plage des valeurs aberrantes: [{stats['min']:.2f}, {stats['max']:.2f}]")

    # 4. Analyse des incohérences dans les dates
    print("\n4. ANALYSE DES DATES")
    print("-" * 40)
    df['SDate'] = pd.to_datetime(df['SDate'])
    df['HDate'] = pd.to_datetime(df['HDate'])

    invalid_dates = df[df['HDate'] <= df['SDate']]
    if len(invalid_dates) > 0:
        print(f"\nTrouvé {len(
            invalid_dates)} lignes où la date de récolte est antérieure ou égale à la date de semis")
        print("\nExemples d'incohérences de dates:")
        print(invalid_dates[['FarmID', 'SDate', 'HDate']].head())

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(4, 4, i)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution de {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()

    print("\n5. ANALYSE DES VARIABLES CATÉGORIELLES")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        unique_values = df[col].nunique()
        value_counts = df[col].value_counts()
        print(f"\n{col}:")
        print(f"  - Nombre de valeurs uniques: {unique_values}")
        if unique_values < 10:
            print("  - Distribution:")
            for val, count in value_counts.items():
                print(f"    {val}: {count} ({count/len(df)*100:.2f}%)")

    print("\n6. VÉRIFICATION DES FORMATS")
    print("-" * 40)
    print("\nTypes de données par colonne:")
    print(df.dtypes)
    return {
        'missing_values': missing_values,
        'duplicate_count': len(duplicate_rows),
        'outliers': outliers_summary
    }


def analyze_crop_data(df):
    """
    Analyze crop data by performing statistical tests and creating visualizations.

    This function conducts Mann-Whitney U tests for numerical variables and
    chi-square tests for categorical variables to assess their relationship
    with crop health. It also generates box plots for numerical variables
    and a correlation matrix heatmap.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing crop data.
                           It should have columns for numerical and
                           categorical variables, including a 'category'
                           column indicating crop health status.

    Returns:
    dict: A dictionary containing two sub-dictionaries:
          - 'numerical_analysis': Results of Mann-Whitney U tests for numerical variables.
            Each key is a variable name, and the value is another dict with 'statistic'
            and 'p_value'.
          - 'categorical_analysis': Results of chi-square tests for categorical variables.
            Each key is a variable name, and the value is another dict with 'chi2'
            and 'p_value'.
    """
    sns.set_palette("husl")
    numerical_vars = [
        'CropCoveredArea', 'CHeight', 'IrriCount', 'WaterCov', 'ExpYield',
        'ndvi', 'evi', 'ndwi', 'gndvi', 'savi', 'msavi', 'SDate_year', 'SDate_month',
        'SDate_day', 'HDate_year', 'HDate_month', 'HDate_day'
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()

    stats_results = {}
    for idx, col in enumerate(numerical_vars):
        if idx < len(axes):
            sns.boxplot(data=df, x='category', y=col, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col} by Category')
            axes[idx].tick_params(axis='x', rotation=45)

        healthy = df[df['category'] == 'Healthy'][col]
        diseased = df[df['category'] == 'Diseased'][col]
        stat, p_value = stats.mannwhitneyu(healthy, diseased)
        stats_results[col] = {'statistic': stat, 'p_value': p_value}

    plt.tight_layout()

    correlation_matrix = df[numerical_vars].corr()

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    categorical_vars = ['Crop', 'State', 'District',
                        'IrriType', 'IrriSource', 'Season']
    chi_square_results = {}

    for cat_var in categorical_vars:
        contingency_table = pd.crosstab(df[cat_var], df['category'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        chi_square_results[cat_var] = {'chi2': chi2, 'p_value': p_value}

    report = {
        'numerical_analysis': stats_results,
        'categorical_analysis': chi_square_results,
    }

    return report


def prepare_data(df):
    df['SDate'] = pd.to_datetime(df['SDate'])
    df['HDate'] = pd.to_datetime(df['HDate'])
    df['growing_period'] = (df['HDate'] - df['SDate']).dt.days
    date_columns = ['SDate', 'HDate']
    for col in date_columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
    return df


def main_analysis(data):
    """
    Perform the main analysis on crop data, including preparation, statistical analysis,
    and identification of significant variables.

    This function prepares the data, conducts statistical analyses on both numerical
    and categorical variables, and prints out variables that are significantly
    associated with the crop category (healthy or diseased) based on a predefined
    significance threshold.

    Parameters:
    data (pandas.DataFrame): The input dataset containing crop information.

    Returns:
    dict: A dictionary containing the results of the analysis, with two keys:
        - 'numerical_analysis': Results of statistical tests on numerical variables.
        - 'categorical_analysis': Results of chi-square tests on categorical variables.

    Each sub-dictionary contains variable names as keys and their corresponding
    test statistics and p-values.
    """
    df = prepare_data(data.copy())
    results = analyze_crop_data(df)
    significance_threshold = 0.05

    print("\nVariables numériques significativement associées à la catégorie (p < 0.05):")
    for var, stats in results['numerical_analysis'].items():
        if stats['p_value'] < significance_threshold:
            print(f"- {var}: p-value = {stats['p_value']:.4f}")

    print("\nVariables catégorielles significativement associées à la catégorie (p < 0.05):")
    for var, stats in results['categorical_analysis'].items():
        if stats['p_value'] < significance_threshold:
            print(f"- {var}: p-value = {stats['p_value']:.4f}")

    return results
