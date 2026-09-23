"""
Feature Engineering Transformers
=================================
Provides custom scikit-learn transformers for outlier handling (IQR bounds)
and domain-specific feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric features using IQR thresholds calculated strictly on the training set
    to prevent distribution leakage across train and test splits.
    """
    def __init__(self, cols, threshold=3.0):
        self.cols = cols
        self.threshold = threshold
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        self.bounds_ = {}
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.cols)
        for col in self.cols:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = (q1 - self.threshold * iqr, q3 + self.threshold * iqr)
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X_out = pd.DataFrame(X, columns=self.cols)
        else:
            X_out = X.copy()
        for col in self.cols:
            if col in X_out.columns and col in self.bounds_:
                lower, upper = self.bounds_[col]
                X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes 5 domain-specific composite indicators from academic, lifestyle,
    social, and environmental factors.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input to FeatureEngineer must be a pandas DataFrame")
        X_out = X.copy()
        
        # 1. Academic Stress Index
        X_out['Academic_Stress_Index'] = (
            X_out['acad_workload_stress'] + 
            X_out['acad_exam_anxiety'] + 
            X_out['acad_grade_pressure']
        ) / 3.0

        # 2. Lifestyle Score
        X_out['Lifestyle_Score'] = (
            X_out['phys_exercise_frequency'] + 
            X_out['phys_stress_relief'] + 
            (6.0 - X_out['sleep_disturbance']) + 
            X_out['sleep_restfulness']
        ) / 4.0

        # 3. Social Support Score
        X_out['Social_Support_Score'] = (
            X_out['fam_communication'] + 
            X_out['fam_support_emotional'] + 
            X_out['fam_contact_satisfaction'] + 
            X_out['env_belonging'] + 
            X_out['study_peer_collaboration']
        ) / 5.0

        # 4. Sleep Quality Indicator
        X_out['Sleep_Quality_Indicator'] = (
            X_out['sleep_restfulness'] + 
            (6.0 - X_out['sleep_disturbance']) + 
            X_out['sleep_duration']
        ) / 3.0

        # 5. Behavioral Consistency Score
        X_out['Behavioral_Consistency_Score'] = (
            X_out['Attendance_Rate'] * (1.0 - X_out['study_procrastination'] / 6.0)
        )
        
        return X_out
