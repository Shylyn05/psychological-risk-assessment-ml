"""
Synthetic Data Generator
========================
Generates a realistic 400-student synthetic dataset matching the student psychological
wellbeing survey schema for research framework validation.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_samples=400, output_path='data/raw/student_mental_health.csv', seed=42):
    np.random.seed(seed)
    
    # 1. Base Demographics
    student_ids = [f"STU-2026-{i:04d}" for i in range(1, n_samples + 1)]
    ages = np.random.choice(range(18, 26), size=n_samples, p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
    genders = np.random.choice(['Male', 'Female', 'Non-Binary', 'Prefer not to say'], size=n_samples, p=[0.45, 0.48, 0.05, 0.02])
    living_status = np.random.choice(['On-Campus', 'Off-Campus with family', 'Off-Campus alone'], size=n_samples, p=[0.40, 0.35, 0.25])
    
    # 2. Financial Stress and Work Status
    fin_work_status = np.random.choice(['No', 'Part-time', 'Full-time'], size=n_samples, p=[0.60, 0.30, 0.10])
    fin_expense_worry = []
    for work in fin_work_status:
        if work == 'Full-time':
            fin_expense_worry.append(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30]))
        elif work == 'Part-time':
            fin_expense_worry.append(np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.20, 0.30, 0.25, 0.15]))
        else:
            fin_expense_worry.append(np.random.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.20, 0.15, 0.05]))
    fin_expense_worry = np.array(fin_expense_worry)
    fin_academic_impact = np.clip(fin_expense_worry + np.random.randint(-1, 2, size=n_samples), 1, 5)

    # 3. Academic Pressure
    study_hours_weekly = np.random.normal(loc=15, scale=5, size=n_samples)
    study_hours_weekly = np.clip(study_hours_weekly, 2, 40)
    
    attendance_rate = np.random.choice([1.0, 0.8, 0.6, 0.4], size=n_samples, p=[0.60, 0.25, 0.10, 0.05])
    
    gpa = 2.0 + 1.5 * (study_hours_weekly / 40.0) + 0.5 * attendance_rate + np.random.normal(loc=0, scale=0.3, size=n_samples)
    gpa = np.clip(np.round(gpa, 2), 1.5, 4.0)

    acad_workload_stress = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.20, 0.30, 0.25, 0.15])
    acad_exam_anxiety = np.clip(acad_workload_stress + np.random.randint(-1, 2, size=n_samples), 1, 5)
    acad_confidence = np.clip(6 - acad_workload_stress + np.random.randint(-1, 2, size=n_samples), 1, 5)
    acad_grade_pressure = np.clip(acad_workload_stress + np.random.randint(0, 2, size=n_samples), 1, 5)

    # 4. Lifestyle & Screen Habits
    screen_time_daily = np.random.normal(loc=7, scale=2.5, size=n_samples)
    screen_time_daily = np.clip(screen_time_daily, 1, 16)
    
    social_media_daily = np.clip(screen_time_daily * np.random.uniform(0.2, 0.7, size=n_samples), 0.5, 8.0)
    
    sleep_duration = np.clip(8.5 - 0.15 * screen_time_daily + np.random.normal(0, 1, size=n_samples), 4.0, 10.0)
    sleep_duration = np.round(sleep_duration, 1)

    sleep_disturbance = np.clip(np.round(5.5 - 0.5 * sleep_duration + np.random.normal(0, 0.8, size=n_samples)), 1, 5).astype(int)
    sleep_restfulness = np.clip(6 - sleep_disturbance + np.random.randint(-1, 1, size=n_samples), 1, 5)
    sleep_aid_usage = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.75, 0.12, 0.08, 0.03, 0.02])
    
    screen_distraction = np.clip(np.round(1 + 0.25 * screen_time_daily + np.random.normal(0, 0.8, size=n_samples)), 1, 5).astype(int)
    screen_bedtime_habit = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.10, 0.20, 0.35, 0.30])
    screen_physical_strain = np.clip(np.round(0.25 * screen_time_daily + np.random.normal(1.5, 0.8, size=n_samples)), 1, 5).astype(int)

    social_media_comparison = np.clip(np.round(0.4 * social_media_daily + np.random.normal(1.2, 0.9, size=n_samples)), 1, 5).astype(int)
    social_media_dependency = np.clip(np.round(0.5 * social_media_daily + np.random.normal(1.0, 0.9, size=n_samples)), 1, 5).astype(int)

    # 5. Support & Environment
    fam_communication = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.15, 0.25, 0.30, 0.20])
    fam_support_emotional = np.clip(fam_communication + np.random.randint(-1, 2, size=n_samples), 1, 5)
    fam_conflict_stress = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    fam_contact_satisfaction = np.clip(fam_communication + np.random.randint(-1, 1, size=n_samples), 1, 5)

    phys_exercise_frequency = np.random.choice([0, 1.5, 3.5, 5.5], size=n_samples, p=[0.25, 0.40, 0.25, 0.10])
    phys_stress_relief = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.10, 0.20, 0.40, 0.25])
    phys_sedentary_habit = np.clip(np.round(0.3 * screen_time_daily + np.random.normal(1.8, 0.8, size=n_samples)), 1, 5).astype(int)

    study_cramming_habit = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.20, 0.30, 0.20, 0.15])
    study_procrastination = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.20, 0.25, 0.25, 0.20])
    study_peer_collaboration = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.20, 0.30, 0.25, 0.15, 0.10])

    env_belonging = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.17, 0.30, 0.30, 0.15])
    env_safety = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.02, 0.08, 0.20, 0.45, 0.25])
    env_counseling_use = np.random.choice([0, 1, 2], size=n_samples, p=[0.75, 0.18, 0.07])

    # 6. Wellbeing Subscales
    well_depressive_mood = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.35, 0.20, 0.10])
    well_anxious_mood = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.30, 0.35, 0.25, 0.10])
    well_loneliness = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.30, 0.20, 0.10])
    well_overwhelmed = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.35, 0.25, 0.15])

    well_optimism = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.10, 0.25, 0.45, 0.20])
    well_emotional_coping = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.10, 0.30, 0.40, 0.20])
    Coping_Mechanism = np.random.choice(['Adaptive', 'Maladaptive', 'Mixed'], size=n_samples, p=[0.50, 0.25, 0.25])

    # Construct DataFrame
    df = pd.DataFrame({
        'student_id': student_ids,
        'Age': ages,
        'Gender': genders,
        'Living_Status': living_status,
        'fin_work_status': fin_work_status,
        'fin_expense_worry': fin_expense_worry,
        'fin_academic_impact': fin_academic_impact,
        'study_hours_weekly': study_hours_weekly,
        'Attendance_Rate': attendance_rate,
        'GPA': gpa,
        'acad_workload_stress': acad_workload_stress,
        'acad_exam_anxiety': acad_exam_anxiety,
        'acad_confidence': acad_confidence,
        'acad_grade_pressure': acad_grade_pressure,
        'screen_time_daily': screen_time_daily,
        'social_media_daily': social_media_daily,
        'sleep_duration': sleep_duration,
        'sleep_disturbance': sleep_disturbance,
        'sleep_restfulness': sleep_restfulness,
        'sleep_aid_usage': sleep_aid_usage,
        'screen_distraction': screen_distraction,
        'screen_bedtime_habit': screen_bedtime_habit,
        'screen_physical_strain': screen_physical_strain,
        'social_media_comparison': social_media_comparison,
        'social_media_dependency': social_media_dependency,
        'fam_communication': fam_communication,
        'fam_support_emotional': fam_support_emotional,
        'fam_conflict_stress': fam_conflict_stress,
        'fam_contact_satisfaction': fam_contact_satisfaction,
        'phys_exercise_frequency': phys_exercise_frequency,
        'phys_stress_relief': phys_stress_relief,
        'phys_sedentary_habit': phys_sedentary_habit,
        'study_cramming_habit': study_cramming_habit,
        'study_procrastination': study_procrastination,
        'study_peer_collaboration': study_peer_collaboration,
        'env_belonging': env_belonging,
        'env_safety': env_safety,
        'env_counseling_use': env_counseling_use,
        'well_depressive_mood': well_depressive_mood,
        'well_anxious_mood': well_anxious_mood,
        'well_loneliness': well_loneliness,
        'well_overwhelmed': well_overwhelmed,
        'well_optimism': well_optimism,
        'well_emotional_coping': well_emotional_coping,
        'Coping_Mechanism': Coping_Mechanism
    })

    # Introduce realistic missing values (~1-2%)
    missing_indices_gpa = np.random.choice(n_samples, size=7, replace=False)
    df.loc[missing_indices_gpa, 'GPA'] = np.nan

    missing_indices_sleep = np.random.choice(n_samples, size=6, replace=False)
    df.loc[missing_indices_sleep, 'sleep_duration'] = np.nan

    missing_indices_gender = np.random.choice(n_samples, size=8, replace=False)
    df.loc[missing_indices_gender, 'Gender'] = np.nan

    missing_indices_dep = np.random.choice(n_samples, size=7, replace=False)
    df.loc[missing_indices_dep, 'well_depressive_mood'] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Synthetic dataset generated successfully at: {output_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


if __name__ == '__main__':
    generate_synthetic_dataset()
