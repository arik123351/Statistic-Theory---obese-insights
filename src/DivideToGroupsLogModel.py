import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import itertools
import warnings

warnings.filterwarnings('ignore')


def prepare_data(df):
    """
    Prepare the data by encoding categorical variables and handling missing values
    """
    df_processed = df.copy()

    # Identify categorical columns (excluding target)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'is_obese' in categorical_cols:
        categorical_cols.remove('is_obese')

    # Label encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # Encode target variable if it's categorical
    if df_processed['is_obese'].dtype == 'object':
        le_target = LabelEncoder()
        df_processed['is_obese'] = le_target.fit_transform(df_processed['is_obese'])

    return df_processed, label_encoders


def wald_test(model, X, feature_indices):
    """
    Perform Wald test for a subset of features using statsmodels approach
    Returns: test statistic, p-value, degrees of freedom
    """
    try:
        # Get coefficients
        coef = model.coef_[0]

        # Calculate Fisher Information Matrix (more stable approach)
        probas = model.predict_proba(X)[:, 1]
        weights = probas * (1 - probas)

        # Add small regularization to avoid singularity
        reg_term = 1e-8 * np.eye(X.shape[1])

        # Fisher Information Matrix: X'WX
        W = np.diag(weights)
        fisher_info = X.T @ W @ X + reg_term

        # Covariance matrix (inverse of Fisher Information)
        cov_matrix = np.linalg.inv(fisher_info)

        # Extract submatrix for the features of interest
        cov_sub = cov_matrix[np.ix_(feature_indices, feature_indices)]
        coef_sub = coef[feature_indices]

        # Wald test statistic: β'(Cov^-1)β
        wald_stat = coef_sub.T @ np.linalg.inv(cov_sub) @ coef_sub
        df = len(feature_indices)
        p_value = 1 - stats.chi2.cdf(wald_stat, df)

        return wald_stat, p_value, df

    except Exception as e:
        print(f"Warning: Wald test failed - {str(e)}")
        # Fallback: use individual t-tests and combine
        try:
            # Calculate standard errors from diagonal of covariance matrix
            probas = model.predict_proba(X)[:, 1]
            weights = probas * (1 - probas)
            W = np.diag(weights)
            fisher_info = X.T @ W @ X + 1e-8 * np.eye(X.shape[1])
            cov_matrix = np.linalg.inv(fisher_info)

            # Individual z-statistics
            coef = model.coef_[0]
            std_errors = np.sqrt(np.diag(cov_matrix))
            z_stats = coef[feature_indices] / std_errors[feature_indices]

            # Sum of squared z-statistics approximates chi-squared
            wald_stat = np.sum(z_stats ** 2)
            df = len(feature_indices)
            p_value = 1 - stats.chi2.cdf(wald_stat, df)

            return wald_stat, p_value, df

        except:
            return np.nan, np.nan, len(feature_indices)


def get_coefficient_stats(model, X, feature_index):
    """
    Get the z-statistic and p-value for a single coefficient using Wald test
    Returns: z_statistic, p_value
    """
    try:
        # Get coefficient
        coef = model.coef_[0, feature_index]

        # Calculate standard error
        probas = model.predict_proba(X)[:, 1]
        weights = probas * (1 - probas)
        W = np.diag(weights)
        fisher_info = X.T @ W @ X + 1e-8 * np.eye(X.shape[1])
        cov_matrix = np.linalg.inv(fisher_info)

        std_error = np.sqrt(cov_matrix[feature_index, feature_index])

        # Calculate z-statistic and p-value
        z_stat = coef / std_error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test

        return z_stat, p_value
    except:
        return np.nan, np.nan


def fit_logistic_model(X, y):
    """
    Fit logistic regression model with better numerical stability
    """
    # Standardize features for better numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000,
                               solver='lbfgs', C=1.0)  # L2 regularization helps stability
    model.fit(X_scaled, y)

    # Store scaler for later use
    model.scaler = scaler

    return model


def evaluate_feature_subgroups(df, base_feature='family_history_with_overweight',
                               target='is_obese', max_features=3):
    """
    Evaluate different feature subgroups and compare them to the base model
    """
    # Prepare data
    df_processed, label_encoders = prepare_data(df)
    del_col = ['NObeyesdad_Overweight_Level_II', 'NObeyesdad_Overweight_Level_I', 'NObeyesdad_Obesity_Type_III',
               'NObeyesdad_Obesity_Type_II', 'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Normal_Weight',
               'NObeyesdad_Insufficient_Weight', 'Weight']

    # Get feature columns (excluding target and base feature)
    all_features = [col for col in df_processed.columns if col not in [target, base_feature] + del_col]

    # Prepare base model data
    X_base = df_processed[[base_feature]].values
    y = df_processed[target].values

    # Fit base model
    base_model = fit_logistic_model(X_base, y)
    X_base_scaled = base_model.scaler.transform(X_base)
    base_accuracy = accuracy_score(y, base_model.predict(X_base_scaled))
    base_auc = roc_auc_score(y, base_model.predict_proba(X_base_scaled)[:, 1])

    # Get z-statistic and p-value for the base feature coefficient
    base_z_stat, base_p_value = get_coefficient_stats(base_model, X_base_scaled, 0)

    print("=" * 60)
    print("FEATURE SUBGROUP ANALYSIS FOR OBESITY PREDICTION")
    print("=" * 60)
    print(f"Base Model: Using only '{base_feature}'")
    print(f"Base Accuracy: {base_accuracy:.4f}")
    print(f"Base AUC: {base_auc:.4f}")
    print(f"Base Feature Z-statistic: {base_z_stat:.4f}")
    print(f"Base Feature P-value: {base_p_value:.6f}")
    print("=" * 60)

    results = []

    # Test different combinations of features
    for r in range(1, min(len(all_features) + 1, max_features + 1)):
        print(f"\nTesting combinations with {r} additional feature(s):")
        print("-" * 50)

        for combo in itertools.combinations(all_features, r):
            # Create feature set: base feature + additional features
            feature_set = list(combo)
            X_combo = df_processed[feature_set].values

            # Fit model with additional features
            combo_model = fit_logistic_model(X_combo, y)

            # Use scaled data for predictions and Wald test
            X_combo_scaled = combo_model.scaler.transform(X_combo)
            combo_accuracy = accuracy_score(y, combo_model.predict(X_combo_scaled))
            combo_auc = roc_auc_score(y, combo_model.predict_proba(X_combo_scaled)[:, 1])

            # Perform Wald test for the additional features
            # (test if coefficients of additional features are jointly significant)
            additional_feature_indices = list(range(1, len(feature_set)))  # exclude base feature
            wald_stat, p_value, df_wald = wald_test(combo_model, X_combo_scaled, additional_feature_indices)

            # Calculate improvement metrics
            accuracy_improvement = combo_accuracy - base_accuracy
            auc_improvement = combo_auc - base_auc

            # For comparison purposes, use test statistics when p-values underflow
            # Higher test statistic = more significant = better
            combo_more_significant = False
            if not np.isnan(wald_stat) and not np.isnan(base_z_stat):
                # Convert base z-stat to chi-square equivalent for fair comparison
                base_chi_square_equiv = base_z_stat ** 2
                combo_more_significant = wald_stat > base_chi_square_equiv

            result = {
                'features': feature_set,
                'additional_features': list(combo),
                'accuracy': combo_accuracy,
                'auc': combo_auc,
                'accuracy_improvement': accuracy_improvement,
                'auc_improvement': auc_improvement,
                'wald_statistic': wald_stat,
                'wald_p_value': p_value,
                'wald_df': df_wald,
                'base_p_value': base_p_value,
                'base_z_stat': base_z_stat,
                'base_chi_square_equiv': base_z_stat ** 2 if not np.isnan(base_z_stat) else np.nan,
                'combo_more_significant': combo_more_significant,
                'combo_better_than_base': p_value < base_p_value if (not np.isnan(p_value) and not np.isnan(
                    base_p_value) and p_value > 0 and base_p_value > 0) else combo_more_significant,
                'significant_improvement': p_value < 0.05 if not np.isnan(p_value) else False
            }
            results.append(result)

    # Sort results by statistical significance (highest Wald statistics first)
    # Filter out NaN Wald statistics first
    valid_results = [r for r in results if not np.isnan(r['wald_statistic'])]
    valid_results.sort(key=lambda x: x['wald_statistic'], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 5 MOST STATISTICALLY SIGNIFICANT COMBINATIONS")
    print("(Ranked by Wald test statistic - highest statistics first)")
    print("=" * 60)

    top_5_significant = valid_results[:5]
    for i, result in enumerate(top_5_significant):
        # Format p-values with scientific notation for very small values
        base_p_str = f"{result['base_p_value']:.2e}" if result[
                                                            'base_p_value'] < 0.001 else f"{result['base_p_value']:.6f}"
        combo_p_str = f"{result['wald_p_value']:.2e}" if result[
                                                             'wald_p_value'] < 0.001 else f"{result['wald_p_value']:.6f}"

        # Handle p-value ratio calculation
        if result['base_p_value'] > 0 and result['wald_p_value'] > 0:
            p_ratio_str = f"{result['wald_p_value'] / result['base_p_value']:.2e}"
        else:
            p_ratio_str = "N/A (p-values too small)"

        print(f"{i + 1}. Additional Features: {result['additional_features']}")
        print(f"   Wald Test: χ² = {result['wald_statistic']:.4f}, p-value = {combo_p_str}")
        print(f"   AUC: {result['auc']:.4f} (Improvement: {result['auc_improvement']:+.4f})")
        print(f"   Accuracy: {result['accuracy']:.4f} (Improvement: {result['accuracy_improvement']:+.4f})")
        print(f"   Statistically Significant: {'Yes' if result['significant_improvement'] else 'No'}")
        print(f"   Base Z-statistic: {result['base_z_stat']:.4f} (χ² equiv: {result['base_chi_square_equiv']:.4f})")
        print(f"   Base p-value: {base_p_str}")
        print(f"   Combo p-value: {combo_p_str}")
        print(f"   P-value ratio (combo/base): {p_ratio_str}")
        print(
            f"   Test Statistic Comparison: Combo χ² ({result['wald_statistic']:.4f}) > Base χ² ({result['base_chi_square_equiv']:.4f}): {'Yes' if result['combo_more_significant'] else 'No'}")
        print(f"   Combo More Significant: {'Yes' if result['combo_better_than_base'] else 'No'}")
        print()

    # Also show top by AUC improvement for comparison
    results_by_auc = sorted(results, key=lambda x: x['auc_improvement'], reverse=True)

    print("=" * 60)
    print("TOP 5 BY AUC IMPROVEMENT (for comparison)")
    print("=" * 60)

    for i, result in enumerate(results_by_auc[:5]):
        # Format p-values with scientific notation for very small values
        base_p_str = f"{result['base_p_value']:.2e}" if result[
                                                            'base_p_value'] < 0.001 else f"{result['base_p_value']:.6f}"
        combo_p_str = f"{result['wald_p_value']:.2e}" if result[
                                                             'wald_p_value'] < 0.001 else f"{result['wald_p_value']:.6f}"

        # Handle p-value ratio calculation
        if result['base_p_value'] > 0 and result['wald_p_value'] > 0:
            p_ratio_str = f"{result['wald_p_value'] / result['base_p_value']:.2e}"
        else:
            p_ratio_str = "N/A (p-values too small)"

        print(f"{i + 1}. Additional Features: {result['additional_features']}")
        print(f"   AUC Improvement: {result['auc_improvement']:+.4f}")
        print(f"   Combo χ²: {result['wald_statistic']:.4f}, Base χ² equiv: {result['base_chi_square_equiv']:.4f}")
        print(f"   Combo p-value: {combo_p_str}")
        print(f"   Base p-value: {base_p_str}")
        print(f"   P-value ratio (combo/base): {p_ratio_str}")
        print(f"   Combo More Significant: {'Yes' if result['combo_better_than_base'] else 'No'}")
        print(f"   Statistically Significant: {'Yes' if result['significant_improvement'] else 'No'}")
        print()

    return results, base_model, top_5_significant


import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def plot_top_wald_statistics(top_5_significant, save_path=None):
    """
    Plot top 5 feature combinations by Wald statistic with horizontal bars.
    Color intensity encodes significance (darker = more significant).
    """

    # Extract data
    feature_combinations = []
    wald_stats = []
    p_values = []

    for i, result in enumerate(top_5_significant):
        features_str = ', '.join(result['additional_features'][:3])
        label = f"Combo {i + 1}: {features_str}"
        feature_combinations.append(label)
        wald_stats.append(result['wald_statistic'])
        p_values.append(result['wald_p_value'])

    # Normalize p-values for color mapping (invert to make lower p darker)
    norm = mpl.colors.Normalize(vmin=min(p_values), vmax=max(p_values))
    cmap = plt.cm.viridis_r  # reversed so lower p is darker
    colors = [cmap(norm(p)) for p in p_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feature_combinations))

    bars = ax.barh(y_pos, wald_stats, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_combinations, fontsize=11)
    ax.invert_yaxis()

    ax.set_xlabel('Wald Test Statistic (χ²)', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Feature Combinations by Wald Statistic',
                 fontsize=14, fontweight='bold', pad=15)

    # Add value annotations
    for bar, stat, pval in zip(bars, wald_stats, p_values):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{stat:.2f}\n(p={pval:.1e})',
                va='center', ha='left', fontsize=9, color='black')

    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_comparison_with_base(top_5_significant, save_path='top_wald_statistics.svg'):
    """
    Create a comparison plot showing Wald statistics vs base model chi-square equivalent
    """

    # Extract data
    combo_names = [f"Combo {i + 1}" for i in range(len(top_5_significant))]
    combo_chi_squares = [result['wald_statistic'] for result in top_5_significant]
    base_chi_squares = [result['base_chi_square_equiv'] for result in top_5_significant]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(combo_names))
    width = 0.35
    for i, result in enumerate(top_5_significant):
        # Create short labels for the combinations
        features_str = ', '.join(result['additional_features'][:3])  # Show first 3 features

    # Create bars
    bars1 = ax.bar(x - width / 2, combo_chi_squares, width, label='Feature Combination',
                   color='steelblue', alpha=0.8, edgecolor='navy')
    bars2 = ax.bar(x + width / 2, base_chi_squares, width, label='Base Model (family_history only)',
                   color='lightcoral', alpha=0.8, edgecolor='darkred')

    ax.set_title(
        'Wald Test Statistics: Feature Combinations vs Base Model',
        fontsize=23, pad=20)
    ax.set_ylabel('Wald Test Statistic (χ²)', fontsize=20)
    # ax.set_xlabel('Feature Combinations (Ranked by Wald Statistic)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combo_names, fontsize = 17)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(combo_chi_squares) * 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=16)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(base_chi_squares) * 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()


