"""
Data Preparation & Risk Label Construction
===========================================
Handles dataset loading, target label generation (5 risk classes), target leakage isolation,
and preprocessor pipeline construction.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(data_path="data/raw/student_mental_health.csv"):
    """
    Loads raw dataset and handles basic row-level deduplication.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at path: {data_path}")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    return df


def generate_risk_labels(df):
    """
    Constructs the 5 psychological risk target classes programmatically based on
    distress and protective factors to simulate survey risk stratification.
    
    Target Categories:
    - 0: Low Risk (score <= -3.0)
    - 1: Mild Risk (-3.0 < score <= -2.0)
    - 2: Moderate Risk (-2.0 < score <= -1.0)
    - 3: High Risk (-1.0 < score <= 0.0)
    - 4: Critical Risk (score > 0.0)
    """
    df_out = df.copy()
    
    # Missing values imputation for target calculation and baseline feature alignment
    num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df_out[col] = df_out[col].fillna(df_out[col].median())
    cat_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        df_out[col] = df_out[col].fillna(df_out[col].mode()[0])

    # Distress score formulation
    distress = (
        df_out['well_depressive_mood'] / 3.0 + 
        df_out['well_anxious_mood'] / 3.0 + 
        df_out['well_loneliness'] / 3.0 + 
        df_out['well_overwhelmed'] / 3.0 + 
        (df_out['fam_conflict_stress'] - 1.0) / 4.0 + 
        (df_out['acad_workload_stress'] - 1.0) / 4.0 + 
        (df_out['acad_exam_anxiety'] - 1.0) / 4.0 + 
        (df_out['acad_grade_pressure'] - 1.0) / 4.0 + 
        (df_out['fin_expense_worry'] - 1.0) / 4.0 + 
        (df_out['sleep_disturbance'] - 1.0) / 4.0
    )

    # Protective score formulation
    protective = (
        df_out['well_optimism'] / 3.0 + 
        df_out['well_emotional_coping'] / 3.0 + 
        (df_out['fam_communication'] - 1.0) / 4.0 + 
        (df_out['fam_support_emotional'] - 1.0) / 4.0 + 
        (df_out['fam_contact_satisfaction'] - 1.0) / 4.0 + 
        (df_out['env_belonging'] - 1.0) / 4.0 + 
        (df_out['env_safety'] - 1.0) / 4.0 + 
        (df_out['phys_stress_relief'] - 1.0) / 4.0 + 
        (df_out['sleep_restfulness'] - 1.0) / 4.0
    )

    prs = distress - protective

    def bin_risk_level(score):
        if score <= -3.0:
            return 0  # Low Risk
        elif score <= -2.0:
            return 1  # Mild Risk
        elif score <= -1.0:
            return 2  # Moderate Risk
        elif score <= 0.0:
            return 3  # High Risk
        else:
            return 4  # Critical Risk

    df_out['Risk_Class'] = prs.apply(bin_risk_level)
    return df_out


def prepare_features_and_target(df, test_size=0.2, random_state=42):
    """
    Isolates predictor variables from target variables and explicitly drops
    wellbeing subscale features to eliminate target leakage.
    Returns stratified train and test partitions.
    """
    # Explicitly drop wellbeing subscales used in target definition to prevent target leakage
    excluded_wellbeing = ['well_depressive_mood', 'well_anxious_mood', 'well_emotional_coping', 'well_overwhelmed']
    drop_cols = ['student_id', 'Risk_Level', 'Risk_Class'] + [col for col in excluded_wellbeing if col in df.columns]
    
    X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Risk_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric_features, categorical_features):
    """
    Constructs a ColumnTransformer for median imputation and standard scaling on numerical features,
    and mode imputation and one-hot encoding on categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ]
    )
    return preprocessor
