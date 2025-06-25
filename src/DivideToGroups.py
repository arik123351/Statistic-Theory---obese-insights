import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

# Assuming df is already loaded and cleaned

def best_subgroup_with_the_best_8_cols(df):


    # Select and encode features
    cols = ['SCC', 'family_history_with_overweight', 'MTRANS_Walking', 'FAVC', 'FAF', 'Age', 'CAEC', 'FCVC']
    X = df[cols]
    y = df['is_obese']

    # One-hot encode all features (chi2 requires non-negative values)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Scale to [0, 1] (needed for numeric columns with chi2)
    X_scaled = MinMaxScaler().fit_transform(X_encoded)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    # Baseline Chi2 for 'family_history_with_overweight'
    baseline_cols = [col for col in X_encoded.columns if 'family_history_with_overweight' in col]
    chi2_scores, _ = chi2(X_scaled_df[baseline_cols], y)
    chi2_baseline = chi2_scores.sum()

    print("🎯 Baseline Chi2 (family_history_with_overweight):", chi2_baseline)

    # Try combinations of 2+ features excluding the baseline
    feature_list = [col for col in X_encoded.columns if col not in baseline_cols]
    results = []

    for r in range(1,4):
        for combo in combinations(feature_list, r):
            chi2_score = chi2(X_scaled_df[list(combo)], y)[0].sum()
            if chi2_score > chi2_baseline:
                results.append((combo, chi2_score))

    # Top 5 combinations
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    print("\n Top 5 Feature Combinations with Higher Chi2 than 'family_history_with_overweight':")
    for i, (combo, score) in enumerate(results, 1):
        print(f"{i}. {combo} => Chi2: {score:.4f}")

    return results

