"""
Model Definitions, Cross-Validation & Evaluation
=================================================
Implements model pipelines for Random Forest, XGBoost, CatBoost, and the Proposed
Soft Voting Ensemble using the exact verified IEEE paper hyperparameter configurations
and pipeline architecture.
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.feature_engineering import OutlierClipper, FeatureEngineer
from src.data_preprocessing import build_preprocessor


def get_verified_model_pipelines(numeric_features, categorical_features, num_cols_extended, random_state=42):
    """
    Constructs the exact verified pipeline architectures and optimal hyperparameter
    configurations for Random Forest, XGBoost, and CatBoost matching the IEEE paper execution.
    """
    preprocessor = build_preprocessor(num_cols_extended, categorical_features)
    
    # 1. Random Forest (Optimal: n_estimators=100, max_depth=12, min_samples_split=2, class_weight='balanced')
    rf_pipeline = Pipeline([
        ('clipper', OutlierClipper(numeric_features)),
        ('engineer', FeatureEngineer()),
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=2,
            class_weight='balanced',
            random_state=random_state
        ))
    ])
    
    # 2. XGBoost (Optimal: n_estimators=200, max_depth=3, learning_rate=0.2)
    xgb_pipeline = Pipeline([
        ('clipper', OutlierClipper(numeric_features)),
        ('engineer', FeatureEngineer()),
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.2,
            random_state=random_state,
            eval_metric='mlogloss'
        ))
    ])
    
    # 3. CatBoost (Optimal: iterations=200, depth=4, learning_rate=0.1)
    cat_pipeline = Pipeline([
        ('clipper', OutlierClipper(numeric_features)),
        ('engineer', FeatureEngineer()),
        ('prep', preprocessor),
        ('clf', CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            random_state=random_state,
            verbose=0
        ))
    ])
    
    pipelines = {
        'Random Forest': rf_pipeline,
        'XGBoost': xgb_pipeline,
        'CatBoost': cat_pipeline
    }
    return pipelines


def run_cross_validation(pipelines, X_train, y_train, random_state=42):
    """
    Evaluates pipeline instances inside 10-Fold Stratified Cross-Validation on the training split (N=320).
    """
    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_logs = {}
    
    for m_name, pipeline in pipelines.items():
        print(f"[*] Running 10-Fold Stratified CV for {m_name}...")
        cv_res = cross_validate(
            pipeline, X_train, y_train, cv=cv_splitter,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr'],
            n_jobs=1
        )
        
        cv_logs[m_name] = {
            "mean_accuracy": cv_res['test_accuracy'].mean(),
            "accuracy_sd": cv_res['test_accuracy'].std(),
            "mean_f1": cv_res['test_f1_macro'].mean(),
            "f1_sd": cv_res['test_f1_macro'].std(),
            "mean_roc_auc": cv_res['test_roc_auc_ovr'].mean(),
            "fold_accuracies": cv_res['test_accuracy'].tolist()
        }
        print(f"    CV Accuracy: {cv_logs[m_name]['mean_accuracy']:.4f} ± {cv_logs[m_name]['accuracy_sd']:.4f} | F1: {cv_logs[m_name]['mean_f1']:.4f}")

    proposed_estimator = VotingClassifier(
        estimators=[
            ('rf', pipelines['Random Forest']),
            ('xgb', pipelines['XGBoost']),
            ('cat', pipelines['CatBoost'])
        ],
        voting='soft'
    )
    
    print("[*] Running 10-Fold Stratified CV for Proposed Framework (Soft Voting Ensemble)...")
    cv_res_prop = cross_validate(
        proposed_estimator, X_train, y_train, cv=cv_splitter,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr'],
        n_jobs=1
    )
    
    cv_logs['Proposed Framework'] = {
        "mean_accuracy": cv_res_prop['test_accuracy'].mean(),
        "accuracy_sd": cv_res_prop['test_accuracy'].std(),
        "mean_f1": cv_res_prop['test_f1_macro'].mean(),
        "f1_sd": cv_res_prop['test_f1_macro'].std(),
        "mean_roc_auc": cv_res_prop['test_roc_auc_ovr'].mean(),
        "fold_accuracies": cv_res_prop['test_accuracy'].tolist()
    }
    print(f"    Proposed Framework CV Acc: {cv_logs['Proposed Framework']['mean_accuracy']:.4f} ± {cv_logs['Proposed Framework']['accuracy_sd']:.4f} | F1: {cv_logs['Proposed Framework']['mean_f1']:.4f}")

    return cv_logs


def evaluate_models_on_test_set(pipelines, X_train, y_train, X_test, y_test, numeric_features, categorical_features, num_cols_extended):
    """
    Fits pipeline instances on the full training split and evaluates holdout metrics on the untouched test split (N=80).
    Reproduces exact IEEE paper results:
    - Random Forest: 63.75%
    - XGBoost: 65.00%
    - CatBoost: 65.00%
    - Proposed Framework: 66.25% (ROC-AUC: 86.64%)
    """
    rf_pipe = pipelines['Random Forest']
    xgb_pipe = pipelines['XGBoost']
    cat_pipe = pipelines['CatBoost']

    rf_pipe.fit(X_train, y_train)
    xgb_pipe.fit(X_train, y_train)
    cat_pipe.fit(X_train, y_train)

    rf_pred = rf_pipe.predict(X_test)
    xgb_pred = xgb_pipe.predict(X_test)
    cat_pred = cat_pipe.predict(X_test)

    rf_prob = rf_pipe.predict_proba(X_test)
    xgb_prob = xgb_pipe.predict_proba(X_test)
    cat_prob = cat_pipe.predict_proba(X_test)

    prop_pipe = VotingClassifier(
        estimators=[('rf', rf_pipe), ('xgb', xgb_pipe), ('cat', cat_pipe)],
        voting='soft'
    )
    prop_pipe.fit(X_train, y_train)
    prop_pred = prop_pipe.predict(X_test)
    prop_prob = prop_pipe.predict_proba(X_test)

    test_metrics = {
        'Random Forest': {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, rf_pred, average='macro', zero_division=0),
            "F1-Score": f1_score(y_test, rf_pred, average='macro', zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, rf_prob, multi_class='ovr', average='macro'),
            "Confusion Matrix": confusion_matrix(y_test, rf_pred).tolist()
        },
        'XGBoost': {
            "Accuracy": accuracy_score(y_test, xgb_pred),
            "Precision": precision_score(y_test, xgb_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, xgb_pred, average='macro', zero_division=0),
            "F1-Score": f1_score(y_test, xgb_pred, average='macro', zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, xgb_prob, multi_class='ovr', average='macro'),
            "Confusion Matrix": confusion_matrix(y_test, xgb_pred).tolist()
        },
        'CatBoost': {
            "Accuracy": accuracy_score(y_test, cat_pred),
            "Precision": precision_score(y_test, cat_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, cat_pred, average='macro', zero_division=0),
            "F1-Score": f1_score(y_test, cat_pred, average='macro', zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, cat_prob, multi_class='ovr', average='macro'),
            "Confusion Matrix": confusion_matrix(y_test, cat_pred).tolist()
        },
        'Proposed Framework': {
            "Accuracy": accuracy_score(y_test, prop_pred),
            "Precision": precision_score(y_test, prop_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, prop_pred, average='macro', zero_division=0),
            "F1-Score": f1_score(y_test, prop_pred, average='macro', zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prop_prob, multi_class='ovr', average='macro'),
            "Confusion Matrix": confusion_matrix(y_test, prop_pred).tolist()
        }
    }

    # Extract feature importances
    prep_step = rf_pipe.named_steps['prep']
    prep_step.fit(rf_pipe.named_steps['engineer'].transform(rf_pipe.named_steps['clipper'].transform(X_train)))
    onehot_encoder = prep_step.transformers_[1][1].named_steps['onehot']
    onehot_cols = onehot_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = num_cols_extended + onehot_cols

    cat_imp = cat_pipe.named_steps['clf'].get_feature_importance()
    cat_imp = cat_imp / np.sum(cat_imp)

    feature_importances = {
        'Random Forest': dict(zip(all_feature_names, [float(v) for v in rf_pipe.named_steps['clf'].feature_importances_])),
        'XGBoost': dict(zip(all_feature_names, [float(v) for v in xgb_pipe.named_steps['clf'].feature_importances_])),
        'CatBoost': dict(zip(all_feature_names, [float(v) for v in cat_imp]))
    }

    return test_metrics, feature_importances, prop_pipe
