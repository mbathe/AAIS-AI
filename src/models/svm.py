import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()


def build_svm_classifier(df):
    feature_columns = [
        'CropCoveredArea', 'CHeight', 'IrriCount', 'WaterCov', 'ExpYield',
        'ndvi', 'evi', 'ndwi', 'gndvi', 'savi', 'msavi'
    ]

    X = df[feature_columns]
    y = df['category']
    enc.fit(y)
    y = enc.transform(y.reshape(-1, 1)).toarray().ravel()

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(
        best_model, X_train_scaled, y_train, cv=5, scoring='f1')

    y_pred = best_model.predict(X_test_scaled)

    if best_model.kernel == 'linear':
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.abs(best_model.coef_[0])
        })
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False)
    else:
        feature_importance = None

    results = {
        'best_params': grid_search.best_params_,
        'cv_scores': {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        },
        'test_performance': {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        },
        'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else None,
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le
    }
    return results


def predict_new_samples(model_results, new_data):
    """
    Fonction pour prédire de nouveaux échantillons
    """
    scaled_data = model_results['scaler'].transform(new_data)
    predictions = model_results['model'].predict(scaled_data)
    return model_results['label_encoder'].inverse_transform(predictions)


def print_model_insights(results):
    """
    Fonction pour afficher les insights du modèle
    """
    insights = []

    insights.append(f"Meilleurs paramètres trouvés: {results['best_params']}")

    cv_mean = results['cv_scores']['mean']
    cv_std = results['cv_scores']['std']
    insights.append(
        f"Score F1 moyen en cross-validation: {cv_mean:.3f} (±{cv_std:.3f})")

    test_f1 = results['test_performance']['classification_report']['weighted avg']['f1-score']
    test_precision = results['test_performance']['classification_report']['weighted avg']['precision']
    test_recall = results['test_performance']['classification_report']['weighted avg']['recall']

    insights.append(f"Performance sur le jeu de test:")
    insights.append(f"- F1-score: {test_f1:.3f}")
    insights.append(f"- Précision: {test_precision:.3f}")
    insights.append(f"- Recall: {test_recall:.3f}")

    # Importance des features
    if results['feature_importance'] is not None:
        insights.append("\nImportance des features (top 5):")
        for feature in results['feature_importance'][:5]:
            insights.append(
                f"- {feature['feature']}: {feature['importance']:.3f}")

    return insights
