"""
Machine Learning-Based Multi-Level Psychological Risk Assessment Framework for University Students
==================================================================================================

Main entry point for executing the full end-to-end Machine Learning pipeline:
- Data Loading & 5-Class Target Risk Label Generation
- Stratified 80/20 Train-Test Split (Target Leakage Prevention)
- 10-Fold Stratified Cross-Validation Hyperparameter Tuning
- Model Evaluation (Random Forest, XGBoost, CatBoost, Proposed Soft Voting Ensemble Framework)
- Performance Metrics & Confusion Matrix Export
- Early Warning Recommendation Module Demonstration

Usage:
    python main.py
"""

import sys
from src.pipeline import run_pipeline

if __name__ == '__main__':
    print("Launching Psychological Risk Assessment Framework Pipeline...\n")
    try:
        run_pipeline()
        print("\n[SUCCESS] Pipeline executed cleanly. Results and figures updated in results/")
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)
