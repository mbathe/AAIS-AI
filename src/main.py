from sklearn import preprocessing
import pandas as pd
from scipy import stats
from models.svm import build_svm_classifier
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
# %%

data = pd.read_csv('../data/processed/data.csv')

# %%
data.head()

# %%
# Clen data
threshold = 1e+100
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()

y = data["category"]
y.unique()
# %%
# result = build_svm_classifier(data)

# %%

enc = preprocessing.OneHotEncoder()
feature_columns = [
    'Healthy', 'Diseased', 'Pests', 'Stressed'
]

le = LabelEncoder()
y = le.fit_transform(y)

feature_columns = [
    'CropCoveredArea', 'CHeight', 'IrriCount', 'WaterCov', 'ExpYield',
    'ndvi', 'evi', 'ndwi', 'gndvi', 'savi', 'msavi'
]


X = data[feature_columns]
y = data['category']
print(y.shape)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(y.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}

# Initialize the SVC model
svm = SVC(random_state=42)

# Set up the grid search with cross-validation
grid_search = GridSearchCV(
    # Use 'f1_macro' for multiclass
    svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Cross-validation scores for the best model
cv_scores = cross_val_score(
    # Ensure consistency in scoring
    best_model, X_train_scaled, y_train, cv=5, scoring='f1_macro'
)

# %%


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
# %%

np.mean(y_test==y_pred)
# %%