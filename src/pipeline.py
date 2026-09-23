"""
End-to-End Execution Pipeline
==============================
Main orchestration script for dataset processing, holdout testing, 10-fold cross-validation,
metric export, figure plotting, and recommendation generation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_preprocessing import load_dataset, generate_risk_labels, prepare_features_and_target
from src.models import get_verified_model_pipelines, run_cross_validation, evaluate_models_on_test_set
from src.recommendation import format_student_assessment_report


def run_pipeline():
    print("==================================================")
    print("1. DATA LOADING & RISK LABEL GENERATION (5 CLASSES)")
    print("==================================================")
    
    data_path = "data/raw/student_mental_health.csv"
    df = load_dataset(data_path)
    print(f"[*] Loaded raw dataset: {df.shape[0]} records, {df.shape[1]} features.")
    
    df = generate_risk_labels(df)
    class_counts = df['Risk_Class'].value_counts().sort_index()
    class_names = ['Low Risk', 'Mild Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
    
    print("[*] Target distribution across 5 risk classes:")
    for idx, count in class_counts.items():
        print(f"    - {class_names[idx]} (Class {idx}): {count} records ({count/len(df)*100:.2f}%)")

    # Split train and holdout test set (80/20 stratified)
    X_train, X_test, y_train, y_test = prepare_features_and_target(df, test_size=0.2, random_state=42)
    print(f"[*] Training partition shape: {X_train.shape}")
    print(f"[*] Holdout test partition shape: {X_test.shape}")
    
    categorical_features = ['Gender', 'Living_Status', 'fin_work_status', 'Coping_Mechanism', 'env_counseling_use']
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]
    
    num_cols_extended = numeric_features + [
        'Academic_Stress_Index', 'Lifestyle_Score', 
        'Social_Support_Score', 'Sleep_Quality_Indicator', 
        'Behavioral_Consistency_Score'
    ]

    print("\n==================================================")
    print("2. HOLDOUT TEST SET EVALUATION (FROZEN TEST PARTITION)")
    print("==================================================")
    
    pipelines = get_verified_model_pipelines(numeric_features, categorical_features, num_cols_extended, random_state=42)
    
    test_metrics, feature_importances, proposed_estimator = evaluate_models_on_test_set(
        pipelines, X_train, y_train, X_test, y_test,
        numeric_features, categorical_features, num_cols_extended
    )
    
    print("\nHoldout Test Set Results Summary:")
    print("--------------------------------------------------")
    for m_name in ['Random Forest', 'XGBoost', 'CatBoost', 'Proposed Framework']:
        m = test_metrics[m_name]
        print(f"{m_name:20s} | Acc: {m['Accuracy']*100:6.2f}% | F1: {m['F1-Score']*100:6.2f}% | ROC-AUC: {m['ROC-AUC']*100:6.2f}%")

    print("\n==================================================")
    print("3. EXPORTING PERFORMANCE TABLES")
    print("==================================================")
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Table 1: Dataset Composition
    t1_data = []
    for idx, count in class_counts.items():
        t1_data.append({
            "Risk Class": idx,
            "Risk Category": class_names[idx],
            "Student Count": count,
            "Percentage": f"{count / len(df) * 100:.2f}%"
        })
    pd.DataFrame(t1_data).to_csv("results/tables/table1_dataset_composition.csv", index=False)
    
    # Table 2: Model Performance Comparison
    t2_data = []
    for m_name in ['Random Forest', 'XGBoost', 'CatBoost', 'Proposed Framework']:
        t2_data.append({
            "Model": m_name,
            "Accuracy": f"{test_metrics[m_name]['Accuracy']*100:.2f}%",
            "Precision (Macro)": f"{test_metrics[m_name]['Precision']*100:.2f}%",
            "Recall (Macro)": f"{test_metrics[m_name]['Recall']*100:.2f}%",
            "F1-Score (Macro)": f"{test_metrics[m_name]['F1-Score']*100:.2f}%",
            "ROC-AUC (Macro)": f"{test_metrics[m_name]['ROC-AUC']*100:.2f}%"
        })
    pd.DataFrame(t2_data).to_csv("results/tables/table2_model_performance.csv", index=False)
    
    # Table 3: Accuracy Analysis
    t3_data = [
        {"Model": m_name, "Accuracy": f"{test_metrics[m_name]['Accuracy']*100:.2f}%"}
        for m_name in ['Random Forest', 'XGBoost', 'CatBoost', 'Proposed Framework']
    ]
    pd.DataFrame(t3_data).to_csv("results/tables/table3_accuracy_analysis.csv", index=False)
    print("[+] Saved tables to results/tables/")

    print("\n==================================================")
    print("4. GENERATING PUBLICATION-GRADE FIGURES")
    print("==================================================")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 9
    
    def clean_label(f):
        mapper = {
            'acad_workload_stress': 'Workload Stress',
            'acad_grade_pressure': 'Grade Pressure',
            'acad_exam_anxiety': 'Exam Anxiety',
            'fam_support_emotional': 'Emotional Support',
            'fam_communication': 'Family Communication',
            'fin_expense_worry': 'Financial Worry',
            'sleep_disturbance': 'Sleep Disturbance',
            'phys_stress_relief': 'Stress Relief Activity',
            'Academic_Stress_Index': 'Academic Stress Index',
            'Lifestyle_Score': 'Lifestyle Score',
            'Social_Support_Score': 'Social Support Score',
            'Sleep_Quality_Indicator': 'Sleep Quality Indicator',
            'Behavioral_Consistency_Score': 'Behavioral Consistency'
        }
        return mapper.get(f, f.replace('_', ' ').title())

    # Figure 1: Feature Importance Analysis
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    models_to_plot = ['Random Forest', 'XGBoost', 'CatBoost']
    
    for i, m_name in enumerate(models_to_plot):
        sorted_imp = sorted(feature_importances[m_name].items(), key=lambda x: x[1], reverse=True)[:10]
        feats, vals = zip(*sorted_imp)
        cleaned_feats = [clean_label(f) for f in feats]
        
        y_pos = np.arange(len(cleaned_feats))
        axes[i].barh(y_pos, vals, color='#2F5597', edgecolor='black', height=0.6)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(cleaned_feats, fontsize=8)
        axes[i].invert_yaxis()
        axes[i].set_title(f"{m_name} Top 10 Features", fontweight='bold', fontsize=10)
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig("results/figures/figure1_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Model Accuracy Comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    models_list = ['Random Forest', 'XGBoost', 'CatBoost', 'Proposed Framework']
    accuracies = [test_metrics[m]['Accuracy'] * 100 for m in models_list]
    colors = ['#b0c4de', '#8ba8c7', '#5f84a2', '#1f4e79']
    
    bars = ax.bar(models_list, accuracies, color=colors, edgecolor='black', width=0.45)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Comparison of Accuracy Across Models', fontweight='bold', pad=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig("results/figures/figure2_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: Confusion Matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_arr = np.array(test_metrics['Proposed Framework']['Confusion Matrix'])
    cm_pct = cm_arr.astype('float') / cm_arr.sum(axis=1)[:, np.newaxis]
    short_classes = ['Low', 'Mild', 'Moderate', 'High', 'Critical']
    
    sns.heatmap(cm_pct, annot=True, fmt=".2%", cmap='Blues', xticklabels=short_classes, yticklabels=short_classes, cbar=True, ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted Class', fontweight='bold')
    ax.set_ylabel('True Class', fontweight='bold')
    ax.set_title('Confusion Matrix: Proposed Framework', fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig("results/figures/figure3_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[+] Saved figures to results/figures/")

    print("\n==================================================")
    print("5. SAMPLE RECOMMENDATION MODULE OUTPUT")
    print("==================================================")
    
    sample_student_id = "STU-2026-0042"
    sample_probs = proposed_estimator.predict_proba(X_test.iloc[0:1])[0]
    sample_pred = np.argmax(sample_probs)
    
    sample_report = format_student_assessment_report(sample_student_id, sample_pred, sample_probs)
    print(sample_report)


if __name__ == '__main__':
    run_pipeline()
