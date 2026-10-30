from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
from IPython.display import display
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay, make_scorer)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 25)
print('Python:', sys.version.split()[0], '| scikit-learn:', sklearn.__version__)

TRAIN_PATH = Path('DATA/Train2021.csv')
TEST_PATH = Path('DATA/Test Data.csv')
RESULTS_DIR = Path('heatguard_results')
TARGET = 'Thermal sensation'
CLASSES = [
    'neutral', 
    'slightly warm',
    'warm', 
    'hot']
FEATURES = ['Temperature', 
            'Humidity', 
            'hrv_mean_hr', 
            'hrv_rmssd',
            'hrv_sdnn',
             'hrv_lf_hf_ratio']

def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f'Update the CSV path: {path}')
    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET, 'subject', 'experiment'])
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    for col in ['subject', 'experiment', TARGET]:
        if df[col].isna().any():
            raise ValueError(f'Missing grouping/target values in {col}')
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq('').any():
            raise ValueError(f'Empty values in {col}')
    if not df[TARGET].isin(CLASSES).all():
        raise ValueError('Unexpected thermal-sensation labels')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df

train_raw = load_data(TRAIN_PATH)
test_raw = load_data(TEST_PATH)
overlap = set(train_raw.subject) & set(test_raw.subject)
assert not overlap, f'Overlapping participant names: {overlap}'
display(pd.DataFrame({'file': ['Train2021', 'Test Data'],
    'rows': [len(train_raw), len(test_raw)],
    'columns': [len(train_raw.columns), len(test_raw.columns)],
    'participants': [train_raw.subject.nunique(), test_raw.subject.nunique()]}))
display(train_raw[FEATURES + [TARGET]].head())


def summarize_sessions(df):
    keys = ['subject', 'experiment']
    counts = df.groupby(keys)[TARGET].nunique()
    assert counts.eq(1).all(), 'Mixed labels: revise the session definition.'
    return df.groupby(keys, as_index=False).agg(
        **{c: (c, 'median') for c in FEATURES},
        actual=(TARGET, 'first'), n_rows=(TARGET, 'size'))

train = summarize_sessions(train_raw)
test = summarize_sessions(test_raw)

X_train = train[FEATURES].copy()
y_train = train.actual.copy()
groups = train.subject.copy()
X_test = test[FEATURES].copy()
y_test = test.actual.copy()

assert set(y_train) == set(CLASSES)
assert not X_train.isna().all().any(), 'At least one feature is entirely missing.'

print('Training groups:', len(train), '| Test groups:', len(test))
print('Training participants:', groups.nunique())

display(X_train.isna().sum().to_frame('missing_training_session_values'))
display(y_train.value_counts().reindex(CLASSES).to_frame('training_sessions'))
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(data=train, x='actual', order=CLASSES, ax=axes[0], color='steelblue')
axes[0].tick_params(axis='x', rotation=25)
axes[0].set_title('Training class balance (sessions)')
sns.scatterplot(data=train, x='Temperature', y='Humidity', hue='actual', ax=axes[1])
axes[1].set_title('Limited environmental coverage')
sns.boxplot(data=train, x='actual', y='hrv_mean_hr', order=CLASSES, ax=axes[2])
axes[2].tick_params(axis='x', rotation=25)
axes[2].set_title('Mean heart rate by recorded label')
plt.tight_layout()
plt.show()
display(train_raw.groupby('experiment')[TARGET].agg(['nunique', 'first']))
display(train_raw.groupby(['Temperature', 'Humidity', TARGET]).size().to_frame('rows'))

def make_pipeline(estimator):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', clone(estimator)),
    ])

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train, groups))
for i, (fit_idx, val_idx) in enumerate(folds, start=1):
    assert not set(groups.iloc[fit_idx]) & set(groups.iloc[val_idx])
    assert set(y_train.iloc[fit_idx]) == set(CLASSES), 'Training fold lacks a class.'
    assert set(y_train.iloc[val_idx]) == set(CLASSES), 'Validation fold lacks a class.'
    print(f'Fold {i}: {groups.iloc[fit_idx].nunique()} training participants, '
          f'{groups.iloc[val_idx].nunique()} validation participants')

classifiers = {
    'Dummy (majority)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(max_iter=3000, random_state=SEED),
    'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=5,
                    min_samples_leaf=2, random_state=SEED, n_jobs=1),
    'Support Vector Machine': SVC(C=1.0, kernel='rbf'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=60,
                         max_depth=2, random_state=SEED),
}
macro_f1 = make_scorer(f1_score, labels=CLASSES, average='macro', pos_label=None, zero_division=0)
scoring = {'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy', 'macro_f1': macro_f1}
rows = []
for name, estimator in classifiers.items():
    scores = cross_validate(make_pipeline(estimator), X_train, y_train,
        cv=folds, scoring=scoring, n_jobs=1, error_score='raise')
    rows.append({'model': name, 'macro_f1': scores['test_macro_f1'].mean(),
        'fold_sd': scores['test_macro_f1'].std(),
        'accuracy': scores['test_accuracy'].mean(),
        'balanced_accuracy': scores['test_balanced_accuracy'].mean()})
comparison = pd.DataFrame(rows).sort_values('macro_f1', ascending=False).reset_index(drop=True)
display(comparison.round(3))
order = comparison.iloc[::-1]
plt.figure(figsize=(9, 5))
plt.barh(order.model, order.macro_f1, xerr=order.fold_sd, capsize=3)
plt.xlabel('Training CV macro F1 (error bars: fold SD)')
plt.title('Classical classifier comparison')
plt.tight_layout()
plt.show()
small_tree = make_pipeline(DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=SEED))
small_tree.fit(X_train, y_train)
plt.figure(figsize=(17, 7))
plot_tree(small_tree.named_steps['model'], feature_names=FEATURES,
          class_names=small_tree.named_steps['model'].classes_.tolist(),
          filled=True, rounded=True, fontsize=8)
plt.title('Training tree: thresholds are in standardized units')
plt.tight_layout()
plt.show()

search = GridSearchCV(make_pipeline(classifiers['Random Forest']),
    param_grid={'model__max_depth': [3, 5, None], 'model__min_samples_leaf': [2, 5]},
    scoring=macro_f1, cv=folds, n_jobs=1, refit=True, error_score='raise')
search.fit(X_train, y_train)
print('Best Random Forest settings:', search.best_params_)
print('Best Random Forest CV macro F1:', round(search.best_score_, 3))
display(pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
        .sort_values('mean_test_score', ascending=False))

best_name = comparison.iloc[0]['model']
if search.best_score_ > comparison.iloc[0]['macro_f1']:
    selected_name = 'Tuned Random Forest'
    final_model = clone(search.best_estimator_)
else:
    selected_name = best_name
    final_model = make_pipeline(classifiers[best_name])
final_model.fit(X_train, y_train)
pred = final_model.predict(X_test)
test_metrics = {
    'accuracy': accuracy_score(y_test, pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, pred),
    'macro_f1': f1_score(y_test, pred, labels=CLASSES, average='macro', zero_division=0),
}
print('Selected using training CV:', selected_name)
display(pd.Series(test_metrics, name='test_session_metric').round(3))
print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))

environment_lookup = train.groupby(['Temperature', 'Humidity']).actual.agg(lambda s: s.mode().iloc[0])
fallback = y_train.mode().iloc[0]
environment_pred = [environment_lookup.get((t, h), fallback)
                    for t, h in zip(test.Temperature, test.Humidity)]
print('Environment-only accuracy:', round(accuracy_score(y_test, environment_pred), 3))
print('Environment-only macro F1:', round(f1_score(y_test, environment_pred,
      labels=CLASSES, average='macro', zero_division=0), 3))

ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASSES,
    cmap='Blues', xticks_rotation=30, colorbar=False)
plt.title('Test sessions: thermal-sensation confusion matrix')
plt.tight_layout()
plt.show()

diagnostics = test[FEATURES + ['experiment']].copy()
diagnostics['participant_code'] = pd.factorize(test.subject)[0] + 1
diagnostics['actual'] = y_test.to_numpy()
diagnostics['predicted'] = pred
diagnostics['correct'] = diagnostics.actual.eq(diagnostics.predicted)
display(diagnostics.loc[~diagnostics.correct].head(10))
participant_results = diagnostics.groupby('participant_code').correct.agg(['count', 'mean'])
display(participant_results.rename(columns={'count': 'sessions', 'mean': 'accuracy'}))
print('Mean participant accuracy:', round(participant_results['mean'].mean(), 3))

importance = permutation_importance(final_model, X_test, y_test, scoring=macro_f1,
    n_repeats=10, random_state=SEED, n_jobs=1)
importance_table = pd.DataFrame({'feature': FEATURES, 'importance': importance.importances_mean,
    'sd': importance.importances_std}).sort_values('importance')
plt.figure(figsize=(8, 4))
plt.barh(importance_table.feature, importance_table.importance, xerr=importance_table.sd)
plt.xlabel('Decrease in macro F1 after shuffling (exploratory)')
plt.tight_layout()
plt.show()
