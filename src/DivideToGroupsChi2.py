import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import numpy as np


def chi2_best_combinations_excluding_weight_family(df, max_features=3):

    del_col = ['NObeyesdad_Overweight_Level_II', 'NObeyesdad_Overweight_Level_I', 'NObeyesdad_Obesity_Type_III',
               'NObeyesdad_Obesity_Type_II', 'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Normal_Weight',
               'NObeyesdad_Insufficient_Weight', 'Weight', 'is_obese']
    # Available features (excluding Weight and family_history_with_overweight)
    available_features = [col for col in df.columns
                          if col not in del_col + ['family_history_with_overweight']]

    print("🔍 Available features for combination:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i}. {feat}")

    # Prepare target variable
    y = df['is_obese']

    # Store results for all combinations
    all_results = []

    # Test combinations of different sizes
    for combo_size in range(1, min(max_features + 1, len(available_features) + 1)):
        print(f"\n📊 Testing combinations of {combo_size} feature(s)...")

        combo_results = []

        for combo in combinations(available_features, combo_size):
            try:
                # Select features for this combination
                X = df[list(combo)]

                # One-hot encode categorical features
                X_encoded = pd.get_dummies(X, drop_first=True)

                # Handle case where all features are categorical and become binary
                if X_encoded.shape[1] == 0:
                    continue

                # Scale to [0, 1] for chi2 (requires non-negative values)
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_encoded)

                # Calculate Chi2 score
                chi2_scores, p_values = chi2(X_scaled, y)
                total_chi2 = chi2_scores.sum()

                # Store results
                combo_results.append({
                    'features': combo,
                    'chi2_score': total_chi2,
                    'feature_count': len(combo),
                    'encoded_features': list(X_encoded.columns),
                    'avg_chi2_per_feature': total_chi2 / len(combo)
                })

            except Exception as e:
                print(f"⚠️  Error with combination {combo}: {e}")
                continue

        # Sort by chi2 score for this combination size
        combo_results.sort(key=lambda x: x['chi2_score'], reverse=True)

        # Show top 3 for this combination size
        print(f"\n🏆 Top 3 combinations with {combo_size} feature(s):")
        for i, result in enumerate(combo_results[:3], 1):
            print(f"  {i}. {result['features']}")
            print(f"     Chi2: {result['chi2_score']:.4f} | Avg per feature: {result['avg_chi2_per_feature']:.4f}")

        all_results.extend(combo_results)

    # Sort all results by chi2 score
    all_results.sort(key=lambda x: x['chi2_score'], reverse=True)

    print(f"\n🎯 OVERALL TOP 10 FEATURE COMBINATIONS:")
    print("=" * 80)

    for i, result in enumerate(all_results[:10], 1):
        print(f"{i:2d}. Features: {result['features']}")
        print(f"    Chi2 Score: {result['chi2_score']:.4f}")
        print(f"    Feature Count: {result['feature_count']}")
        print(f"    Avg Chi2/Feature: {result['avg_chi2_per_feature']:.4f}")
        print(f"    Encoded as: {result['encoded_features']}")
        print("-" * 60)

    return all_results


def compare_combinations_detailed(df, top_combinations=5):
    """
    Detailed comparison of top feature combinations with statistical analysis
    """

    results = chi2_best_combinations_excluding_weight_family(df)

    print(f"\n📈 DETAILED ANALYSIS OF TOP {top_combinations} COMBINATIONS:")
    print("=" * 90)

    for i, result in enumerate(results[:top_combinations], 1):
        print(f"\n🔍 COMBINATION #{i}: {result['features']}")

        # Get the actual data for this combination
        X = df[list(result['features'])]
        y = df['is_obese']

        # Encode and scale
        X_encoded = pd.get_dummies(X, drop_first=True)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Calculate individual chi2 scores
        chi2_scores, p_values = chi2(X_scaled, y)

        print(f"   Overall Chi2: {result['chi2_score']:.4f}")
        print(f"   Features: {len(result['features'])}")
        print(f"   Efficiency (Chi2/Feature): {result['avg_chi2_per_feature']:.4f}")

        print(f"   Individual feature contributions:")
        for j, (feature, score, p_val) in enumerate(zip(X_encoded.columns, chi2_scores, p_values)):
            print(f"     - {feature}: Chi2={score:.4f}, p-value={p_val:.4f}")

        print("-" * 60)

    return results[:top_combinations]

# Usage example:
# results = chi2_best_combinations_excluding_weight_family(df, max_features=4)
# detailed_results = compare_combinations_detailed(df, top_combinations=5)