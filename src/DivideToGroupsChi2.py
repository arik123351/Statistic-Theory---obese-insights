import pandas as pd
from sklearn.feature_selection import chi2 as chi2_1
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.model_selection import train_test_split
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




from sklearn.linear_model import LogisticRegression
from scipy import stats


def wald_best_subgroup_with_the_best_8_cols(df):
    def wald_test(model, X_train):
        coef = model.coef_[0]
        X_array = np.array(X_train, dtype=np.float64)
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        y_pred_proba = np.clip(y_pred_proba, 1e-8, 1 - 1e-8)

        weights = y_pred_proba * (1 - y_pred_proba)
        W = np.diag(weights)
        H = X_array.T @ W @ X_array

        if np.linalg.cond(H) > 1e12:
            H += np.eye(H.shape[0]) * 1e-8

        try:
            cov_matrix = np.linalg.inv(H)
            std_errors = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            n_samples = X_array.shape[0]
            std_errors = np.sqrt(np.abs(coef) / n_samples + 1e-8)

        std_errors = np.maximum(std_errors, 1e-8)
        z_scores = coef / std_errors

        # Use survival function for better numerical precision with very small p-values
        p_values = 2 * stats.norm.sf(np.abs(z_scores))

        # Set minimum p-value threshold to avoid numerical issues
        min_p_threshold = 1e-300
        p_values = np.maximum(p_values, min_p_threshold)

        return p_values, z_scores, std_errors

    def format_p_value(p_val):
        """Format p-values for better readability"""
        if p_val < 1e-10:
            return f"{p_val:.2e}"
        else:
            return f"{p_val:.6f}"

    def get_significance_tier(p_val):
        """Categorize p-values into significance tiers"""
        if p_val < 1e-10:
            return "extremely_significant"
        elif p_val < 0.001:
            return "highly_significant"
        elif p_val < 0.01:
            return "very_significant"
        elif p_val < 0.05:
            return "significant"
        else:
            return "not_significant"

    cols = ['SCC', 'family_history_with_overweight', 'MTRANS_Walking', 'FAVC', 'FAF', 'Age', 'CAEC', 'FCVC']
    X = df[cols]
    y = df['is_obese']
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Baseline
    baseline_cols = [col for col in X_encoded.columns if 'family_history_with_overweight' in col]
    X_base = X_encoded[baseline_cols]
    X_train, _, y_train, _ = train_test_split(X_base, y, test_size=0.2, random_state=0)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train, y_train)

    baseline_p_values, baseline_z_scores, _ = wald_test(baseline_model, X_train)
    baseline_min_p = float(np.min(baseline_p_values))
    baseline_tier = get_significance_tier(baseline_min_p)

    print(f"\n🎯 Baseline Feature: {baseline_cols}")
    for i, col in enumerate(baseline_cols):
        p_formatted = format_p_value(baseline_p_values[i])
        print(f"   {col}: p = {p_formatted}, z = {baseline_z_scores[i]:.3f}")
    print(f"Baseline Min p-value: {format_p_value(baseline_min_p)} ({baseline_tier})")

    # Test combinations
    feature_list = [col for col in X_encoded.columns if col not in baseline_cols]
    results = []

    for r in range(1, 4):
        for combo in combinations(feature_list, r):
            try:
                X_sub = X_encoded[list(combo)]
                X_train, _, y_train, _ = train_test_split(X_sub, y, test_size=0.2, random_state=0)

                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                p_values, z_scores, _ = wald_test(model, X_train)
                min_p = float(np.min(p_values))

                print(f'Combo: {combo}')
                print(f'p_values: {[format_p_value(p) for p in p_values]}')
                print(f'min_p: {format_p_value(min_p)}')
                print(f'Significance tier: {get_significance_tier(min_p)}')
                print('---')

                # Modified comparison logic
                # Instead of strict numerical comparison, use significance tiers
                combo_tier = get_significance_tier(min_p)

                # Consider a combination "better" if:
                # 1. It has a lower p-value (traditional approach), OR
                # 2. It's in the same extreme significance tier but has more significant features, OR
                # 3. It's in the same tier but has better average significance
                avg_p = float(np.mean(p_values))
                n_significant = int(np.sum(p_values < 0.05))

                # Determine if this combo is "better" than baseline
                if baseline_tier == "extremely_significant" and combo_tier == "extremely_significant":
                    # Both are extremely significant, compare by other metrics
                    better_than_baseline = (avg_p < np.mean(baseline_p_values)) or (n_significant > 0)
                else:
                    # Traditional p-value comparison
                    better_than_baseline = min_p < baseline_min_p

                results.append({
                    'combo': combo,
                    'min_p_value': min_p,
                    'avg_p_value': avg_p,
                    'n_significant': n_significant,
                    'better_than_baseline': better_than_baseline,
                    'significance_tier': combo_tier,
                    'p_values': dict(zip(combo, p_values)),
                    'z_scores': dict(zip(combo, z_scores)),
                })
            except Exception as e:
                print(f"Error with combo {combo}: {e}")
                continue

    # Sort by significance tier first, then by p-value
    tier_order = {"extremely_significant": 0, "highly_significant": 1, "very_significant": 2, "significant": 3,
                  "not_significant": 4}
    results.sort(key=lambda x: (tier_order.get(x['significance_tier'], 5), x['min_p_value']))

    # Get results that are potentially interesting
    if baseline_tier == "extremely_significant":
        # If baseline is extremely significant, show all extremely significant results
        interesting_results = [r for r in results if r['significance_tier'] == "extremely_significant"][:10]
    else:
        # Traditional approach for less extreme baselines
        interesting_results = [r for r in results if r['min_p_value'] < baseline_min_p][:10]

    print(f"\n📊 Top Results (Baseline tier: {baseline_tier}):")
    if not interesting_results:
        print("   No combinations found in the same or better significance tier.")
        # Show top 5 overall results
        print("\n   Top 5 overall results:")
        for i, res in enumerate(results[:5]):
            combo_str = ', '.join(res['combo'])
            print(
                f"{i + 1}. ({combo_str}) → p = {format_p_value(res['min_p_value'])} ({res['significance_tier']}) | significant: {res['n_significant']}/{len(res['combo'])}")
    else:
        for i, res in enumerate(interesting_results):
            combo_str = ', '.join(res['combo'])
            better_indicator = "✓" if res['better_than_baseline'] else "="
            print(
                f"{i + 1}. {better_indicator} ({combo_str}) → p = {format_p_value(res['min_p_value'])} ({res['significance_tier']}) | significant: {res['n_significant']}/{len(res['combo'])}")

    print(f"\n📈 Summary:")
    print(f"   ➤ Total combinations tested: {len(results)}")
    print(f"   ➤ Baseline significance tier: {baseline_tier}")

    # Count by significance tiers
    tier_counts = {}
    for tier in ["extremely_significant", "highly_significant", "very_significant", "significant", "not_significant"]:
        count = sum(1 for r in results if r['significance_tier'] == tier)
        if count > 0:
            tier_counts[tier] = count

    print(f"   ➤ Results by significance tier:")
    for tier, count in tier_counts.items():
        print(f"      - {tier.replace('_', ' ').title()}: {count}")

    return {
        'baseline': {
            'features': baseline_cols,
            'min_p_value': baseline_min_p,
            'significance_tier': baseline_tier,
            'p_values': baseline_p_values,
            'z_scores': baseline_z_scores
        },
        'best_combinations': results[:10],
        'tier_counts': tier_counts,
        'total_tested_combinations': len(results)
    }