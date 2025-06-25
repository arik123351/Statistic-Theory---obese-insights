import pandas as pd
from sklearn.feature_selection import chi2 as chi2_1
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2 as chi2_2
from itertools import combinations
import numpy as np

# Assuming df is already loaded and cleaned

def chi2_best_subgroup_with_the_best_8_cols(df):

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
    chi2_scores, _ = chi2_1(X_scaled_df[baseline_cols], y)
    chi2_baseline = chi2_scores.sum()

    print("🎯 Baseline Chi2 (family_history_with_overweight):", chi2_baseline)

    # Try combinations of 2+ features excluding the baseline
    feature_list = [col for col in X_encoded.columns if col not in baseline_cols]
    results = []

    for r in range(1,4):
        for combo in combinations(feature_list, r):
            chi2_score = chi2_1(X_scaled_df[list(combo)], y)[0].sum()
            if chi2_score > chi2_baseline:
                results.append((combo, chi2_score))

    # Top 5 combinations
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    print("\n Top 5 Feature Combinations with Higher Chi2 than 'family_history_with_overweight':")
    for i, (combo, score) in enumerate(results, 1):
        print(f"{i}. {combo} => Chi2: {score:.4f}")

    return results



def wald_best_subgroup_with_the_best_8_cols(df):

    # Setup
    cols = ['SCC', 'family_history_with_overweight', 'MTRANS_Walking', 'FAVC', 'FAF', 'Age', 'CAEC', 'FCVC']
    X = df[cols]
    y = df['is_obese']

    # Encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Baseline: Wald p-value for 'family_history_with_overweight'
    baseline_cols = [col for col in X_encoded.columns if 'family_history_with_overweight' in col]
    X_base = X_encoded[baseline_cols]
    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=0)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Wald test statistic and p-value
    coef = model.coef_[0]
    se = np.sqrt(np.diag(np.linalg.inv(np.dot(X_train.T, X_train))))  # approx. std errors
    wald_stats = (coef / se) ** 2
    p_values = chi2_2.sf(wald_stats, df=1)

    print("🎯 Baseline Wald p-value (family_history_with_overweight):", p_values[0])

    # Now test all other combinations
    results = []

    feature_list = list(X_encoded.columns)
    feature_list = [f for f in feature_list if f not in baseline_cols]

    for r in range(1,4):
        for combo in combinations(feature_list, r):
            X_sub = X_encoded[list(combo)]
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.2, random_state=0)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            coef = model.coef_[0]
            try:
                XtX = np.dot(X_train.values.astype(float).T, X_train.values.astype(float))
                se = np.sqrt(np.diag(np.linalg.pinv(XtX)))

                wald_stats = (coef / se) ** 2
                p_vals = chi2_2.sf(wald_stats, df=1)

                # Average p-value or min p-value as metric (lower is better)
                avg_p = np.mean(p_vals)

                # You can change this to a specific threshold like 0.05 if preferred
                if avg_p < p_values[0]:  # Compare with baseline
                    results.append((combo, avg_p))

            except np.linalg.LinAlgError:
                # Singular matrix, skip
                continue

    # Sort and print best results
    results = sorted(results, key=lambda x: x[1])#[:5]

    print("Feature combinations with better Wald p-value than 'family_history_with_overweight':")
    for combo, p_val in results:
        print(f"{combo} => Avg p-value: {p_val:.4e}")

    return results
