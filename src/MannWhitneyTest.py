import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from itertools import combinations


def mann_whitney_selected_features(df,
                                   target='is_obese',
                                   feature_to_split='family_history_with_overweight',
                                   selected_features=['FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
                                                        'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS_Walking',
                                                        'MTRANS_Public_Transportation', 'MTRANS_Motorbike',
                                                        'MTRANS_Bike', 'MTRANS_Automobile'],
                                   max_features=4,
                                   p_threshold=0.05):
    if selected_features is None:
        raise ValueError("You must provide a list of selected features.")

    results = []

    # Median split value for family_history_with_overweight
    median_val = df[feature_to_split].median()

    # Define feature types for proper handling
    binary_features = ['Gender', 'FAVC', 'SMOKE', 'SCC',
                       'MTRANS_Automobile', 'MTRANS_Bike',
                       'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']
    continuous_features = ['CAEC', 'CALC', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    # Loop over combinations of 1 to max_features from the selected list
    for r in range(1, max_features + 1):
        for feats in combinations(selected_features, r):
            # Create a subset of data with all necessary columns
            subset_cols = list(feats) + [target, feature_to_split]
            subset_df = df[subset_cols].dropna()

            if subset_df[feature_to_split].nunique() < 2:
                continue

            # Create high-risk and low-risk conditions based on the selected features
            high_risk_conditions = []
            low_risk_conditions = []

            for feat in feats:
                if feat in binary_features:
                    # For binary: high risk vs low risk
                    if feat in ['FAVC', 'SMOKE', 'MTRANS_Automobile', 'MTRANS_Motorbike']:
                        high_risk_conditions.append(subset_df[feat] == 1)
                        low_risk_conditions.append(subset_df[feat] == 0)
                    else:
                        high_risk_conditions.append(subset_df[feat] == 0)
                        low_risk_conditions.append(subset_df[feat] == 1)
                elif feat in continuous_features:
                    # For continuous: high risk vs low risk
                    feat_median = subset_df[feat].median()
                    if feat in ['NCP', 'TUE', 'CAEC', 'CALC']:
                        high_risk_conditions.append(subset_df[feat] >= feat_median)
                        low_risk_conditions.append(subset_df[feat] < feat_median)
                    else:
                        high_risk_conditions.append(subset_df[feat] < feat_median)
                        low_risk_conditions.append(subset_df[feat] >= feat_median)

            # Combine conditions (all must be true for each risk group)
            if len(high_risk_conditions) == 1:
                high_risk_condition = high_risk_conditions[0]
                low_risk_condition = low_risk_conditions[0]
            else:
                high_risk_condition = high_risk_conditions[0]
                low_risk_condition = low_risk_conditions[0]
                for i in range(1, len(high_risk_conditions)):
                    high_risk_condition = high_risk_condition & high_risk_conditions[i]
                    low_risk_condition = low_risk_condition & low_risk_conditions[i]

            # Filter to high-risk and low-risk groups
            high_risk_group = subset_df[high_risk_condition]
            low_risk_group = subset_df[low_risk_condition]

            if len(high_risk_group) < 10 or len(low_risk_group) < 10:  # Skip if too few samples
                continue

            # Group1: High-risk group with no family history, Group2: Low-risk group
            group1 = high_risk_group[high_risk_group[feature_to_split] == 0][target]
            group2 = low_risk_group[low_risk_group[feature_to_split] == 1][target]

            if len(group1) < 5 or len(group2) < 5:
                continue

            # stat, p = mannwhitneyu(group1, group2, alternative='greater')
            # print(f'comb: {feats}. p_val = {p}. stat = {stat}')
            try:
                stat, p = mannwhitneyu(group1, group2, alternative='greater')
                # print(f'comb: {feats}. p_val = {p}. stat = {stat}')
                if p < p_threshold:
                    results.append({
                        'features': feats,
                        'p_value': p,
                        'stat': stat,
                        'n_group1': len(group1),
                        'n_group2': len(group2),
                        'mean_group1': group1.mean(),
                        'mean_group2': group2.mean(),
                        'total_samples': len(high_risk_group) + len(low_risk_group),
                        'effect_size': abs(group2.mean() - group1.mean())
                    })
                    print(f'comb: {feats}. p_val = {p}. stat = {stat}')
            except ValueError:
                continue

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No significant combinations found!")
        return {
            'all_results': results_df,
            'top_5_significant': pd.DataFrame(),
            'least_5_significant': pd.DataFrame()
        }

    # Sort by p-value (ascending for most significant)
    results_df = results_df.sort_values('p_value')

    # Get top 5 most significant (lowest p-values)
    top_5_significant = results_df.head(5)

    # Get least 5 significant (highest p-values, but still below threshold)
    least_5_significant = results_df.tail(5)

    return {
        'all_results': results_df,
        'top_5_significant': top_5_significant,
        'least_5_significant': least_5_significant
    }


def mann_whitney_for_family_not_obese(df,
                                   target='is_obese',
                                   feature_to_split='family_history_with_overweight',
                                   selected_features=['FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
                                                        'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS_Walking',
                                                        'MTRANS_Public_Transportation', 'MTRANS_Motorbike',
                                                        'MTRANS_Bike', 'MTRANS_Automobile'],
                                   max_features=4,
                                   p_threshold=0.05):
    if selected_features is None:
        raise ValueError("You must provide a list of selected features.")

    results = []

    # Median split value for family_history_with_overweight
    median_val = df[feature_to_split].median()

    # Define feature types for proper handling
    binary_features = ['Gender', 'FAVC', 'SMOKE', 'SCC',
                       'MTRANS_Automobile', 'MTRANS_Bike',
                       'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']
    continuous_features = ['CAEC', 'CALC', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    # Loop over combinations of 1 to max_features from the selected list
    for r in range(1, max_features + 1):
        for feats in combinations(selected_features, r):
            # Create a subset of data with all necessary columns
            subset_cols = list(feats) + [target, feature_to_split]
            subset_df = df[subset_cols].dropna()

            if subset_df[feature_to_split].nunique() < 2:
                continue

            # Create high-risk and low-risk conditions based on the selected features
            high_risk_conditions = []
            low_risk_conditions = []

            for feat in feats:
                if feat in binary_features:
                    # For binary: high risk vs low risk
                    if feat in ['FAVC', 'SMOKE', 'MTRANS_Automobile', 'MTRANS_Motorbike']:
                        high_risk_conditions.append(subset_df[feat] == 1)
                        low_risk_conditions.append(subset_df[feat] == 0)
                    else:
                        high_risk_conditions.append(subset_df[feat] == 0)
                        low_risk_conditions.append(subset_df[feat] == 1)
                elif feat in continuous_features:
                    # For continuous: high risk vs low risk
                    feat_median = subset_df[feat].median()
                    if feat in ['NCP', 'TUE', 'CAEC', 'CALC']:
                        high_risk_conditions.append(subset_df[feat] >= feat_median)
                        low_risk_conditions.append(subset_df[feat] < feat_median)
                    else:
                        high_risk_conditions.append(subset_df[feat] < feat_median)
                        low_risk_conditions.append(subset_df[feat] >= feat_median)

            # Combine conditions (all must be true for each risk group)
            if len(high_risk_conditions) == 1:
                high_risk_condition = high_risk_conditions[0]
                low_risk_condition = low_risk_conditions[0]
            else:
                high_risk_condition = high_risk_conditions[0]
                low_risk_condition = low_risk_conditions[0]
                for i in range(1, len(high_risk_conditions)):
                    high_risk_condition = high_risk_condition & high_risk_conditions[i]
                    low_risk_condition = low_risk_condition & low_risk_conditions[i]

            # Filter to high-risk and low-risk groups
            high_risk_group = subset_df[high_risk_condition]
            low_risk_group = subset_df[low_risk_condition]

            if len(high_risk_group) < 10 or len(low_risk_group) < 10:  # Skip if too few samples
                continue

            # Group1: High-risk group with no family history, Group2: Low-risk group
            group1 = subset_df[subset_df[feature_to_split] == 0][target]
            group2 = low_risk_group[low_risk_group[feature_to_split] == 1][target]

            if len(group1) < 5 or len(group2) < 5:
                continue

            # stat, p = mannwhitneyu(group1, group2, alternative='greater')
            # print(f'comb: {feats}. p_val = {p}. stat = {stat}')
            try:
                stat, p = mannwhitneyu(group1, group2, alternative='greater')
                # print(f'comb: {feats}. p_val = {p}. stat = {stat}')
                if p < p_threshold:
                    results.append({
                        'features': feats,
                        'p_value': p,
                        'stat': stat,
                        'n_group1': len(group1),
                        'n_group2': len(group2),
                        'mean_group1': group1.mean(),
                        'mean_group2': group2.mean(),
                        'total_samples': len(high_risk_group) + len(low_risk_group),
                        'effect_size': abs(group2.mean() - group1.mean())
                    })
                    print(f'comb: {feats}. p_val = {p}. stat = {stat}')
            except ValueError:
                continue

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No significant combinations found! for if no family_history_with_overweight > low_risk_group + family_history_with_overweight")
        return {
            'all_results': results_df,
            'top_5_significant': pd.DataFrame(),
            'least_5_significant': pd.DataFrame()
        }

    # Sort by p-value (ascending for most significant)
    results_df = results_df.sort_values('p_value')

    # Get top 5 most significant (lowest p-values)
    top_5_significant = results_df.head(5)

    # Get least 5 significant (highest p-values, but still below threshold)
    least_5_significant = results_df.tail(5)

    return {
        'all_results': results_df,
        'top_5_significant': top_5_significant,
        'least_5_significant': least_5_significant
    }



def mann_whitney_for_family_obese(df,
                                   target='is_obese',
                                   feature_to_split='family_history_with_overweight',
                                   selected_features=['FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
                                                        'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS_Walking',
                                                        'MTRANS_Public_Transportation', 'MTRANS_Motorbike',
                                                        'MTRANS_Bike', 'MTRANS_Automobile'],
                                   max_features=4,
                                   p_threshold=0.05):
    if selected_features is None:
        raise ValueError("You must provide a list of selected features.")

    results = []

    # Median split value for family_history_with_overweight
    median_val = df[feature_to_split].median()

    # Define feature types for proper handling
    binary_features = ['Gender', 'FAVC', 'SMOKE', 'SCC',
                       'MTRANS_Automobile', 'MTRANS_Bike',
                       'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']
    continuous_features = ['CAEC', 'CALC', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    # Loop over combinations of 1 to max_features from the selected list
    for r in range(1, max_features + 1):
        for feats in combinations(selected_features, r):
            # Create a subset of data with all necessary columns
            subset_cols = list(feats) + [target, feature_to_split]
            subset_df = df[subset_cols].dropna()

            if subset_df[feature_to_split].nunique() < 2:
                continue

            # Create high-risk and low-risk conditions based on the selected features
            high_risk_conditions = []
            low_risk_conditions = []

            for feat in feats:
                if feat in binary_features:
                    # For binary: high risk vs low risk
                    if feat in ['FAVC', 'SMOKE', 'MTRANS_Automobile', 'MTRANS_Motorbike']:
                        high_risk_conditions.append(subset_df[feat] == 1)
                        low_risk_conditions.append(subset_df[feat] == 0)
                    else:
                        high_risk_conditions.append(subset_df[feat] == 0)
                        low_risk_conditions.append(subset_df[feat] == 1)
                elif feat in continuous_features:
                    # For continuous: high risk vs low risk
                    feat_median = subset_df[feat].median()
                    if feat in ['NCP', 'TUE', 'CAEC', 'CALC']:
                        high_risk_conditions.append(subset_df[feat] >= feat_median)
                        low_risk_conditions.append(subset_df[feat] < feat_median)
                    else:
                        high_risk_conditions.append(subset_df[feat] < feat_median)
                        low_risk_conditions.append(subset_df[feat] >= feat_median)

            # Combine conditions (all must be true for each risk group)
            if len(high_risk_conditions) == 1:
                high_risk_condition = high_risk_conditions[0]
                low_risk_condition = low_risk_conditions[0]
            else:
                high_risk_condition = high_risk_conditions[0]
                low_risk_condition = low_risk_conditions[0]
                for i in range(1, len(high_risk_conditions)):
                    high_risk_condition = high_risk_condition & high_risk_conditions[i]
                    low_risk_condition = low_risk_condition & low_risk_conditions[i]

            # Filter to high-risk and low-risk groups
            high_risk_group = subset_df[high_risk_condition]
            low_risk_group = subset_df[low_risk_condition]

            if len(high_risk_group) < 10 or len(low_risk_group) < 10:  # Skip if too few samples
                continue

            # Group1: High-risk group with no family history, Group2: Low-risk group
            group1 = high_risk_group[high_risk_group[feature_to_split] == 0][target]
            group2 = subset_df[subset_df[feature_to_split] == 1][target]

            if len(group1) < 5 or len(group2) < 5:
                continue

            # stat, p = mannwhitneyu(group1, group2, alternative='greater')
            # print(f'comb: {feats}. p_val = {p}. stat = {stat}')
            try:
                stat, p = mannwhitneyu(group1, group2, alternative='greater')
                # print(f'comb: {feats}. p_val = {p}. stat = {stat}')
                if p < p_threshold:
                    results.append({
                        'features': feats,
                        'p_value': p,
                        'stat': stat,
                        'n_group1': len(group1),
                        'n_group2': len(group2),
                        'mean_group1': group1.mean(),
                        'mean_group2': group2.mean(),
                        'total_samples': len(high_risk_group) + len(low_risk_group),
                        'effect_size': abs(group2.mean() - group1.mean())
                    })
                    print(f'comb: {feats}. p_val = {p}. stat = {stat}')
            except ValueError:
                continue

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No significant combinations found! for if high_risk_group + with no family_history_with_overweight > family_history_with_overweight")
        return {
            'all_results': results_df,
            'top_5_significant': pd.DataFrame(),
            'least_5_significant': pd.DataFrame()
        }

    # Sort by p-value (ascending for most significant)
    results_df = results_df.sort_values('p_value')

    # Get top 5 most significant (lowest p-values)
    top_5_significant = results_df.head(5)

    # Get least 5 significant (highest p-values, but still below threshold)
    least_5_significant = results_df.tail(5)

    return {
        'all_results': results_df,
        'top_5_significant': top_5_significant,
        'least_5_significant': least_5_significant
    }





def print_all_significant_groups(df, p_threshold=0.05):
    """
    Run the analysis and print ALL significant groups with p < 0.05
    """
    print("=" * 80)
    print("RUNNING ANALYSIS TO FIND ALL SIGNIFICANT GROUPS (p < 0.05)")
    print("=" * 80)

    # Run the analysis
    results = mann_whitney_selected_features(df, p_threshold=p_threshold)

    # Get all significant results
    significant_groups = results['all_results']

    if significant_groups.empty:
        print("❌ No significant combinations found with p < 0.05")
        return None

    print(f"✅ Found {len(significant_groups)} significant combinations with p < {p_threshold}")
    print("\n" + "=" * 80)
    print("ALL SIGNIFICANT GROUPS (p < 0.05):")
    print("=" * 80)

    # Print each significant group
    for idx, (_, row) in enumerate(significant_groups.iterrows(), 1):
        print(f"\n🔍 GROUP {idx}:")
        print(f"   Features: {list(row['features'])}")
        print(f"   P-value: {row['p_value']:.2e}")
        print(f"   U-statistic: {row['stat']:.2f}")
        print(f"   Effect size: {row['effect_size']:.3f}")
        print(f"   Sample sizes:")
        print(f"     - High-risk + No family history: {row['n_group1']}")
        print(f"     - Low-risk group: {row['n_group2']}")
        print(f"   Obesity rates:")
        print(f"     - High-risk + No family history: {row['mean_group1']:.3f} ({row['mean_group1'] * 100:.1f}%)")
        print(f"     - Low-risk group: {row['mean_group2']:.3f} ({row['mean_group2'] * 100:.1f}%)")
        print(f"   Total samples analyzed: {row['total_samples']}")
        print("-" * 60)

    return significant_groups


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')


def plot_mann_whitney_results(results_dict, df=None, save_plots=False, figsize=(15, 12)):
    """
    Create comprehensive visualizations for Mann-Whitney U test results

    Parameters:
    -----------
    results_dict : dict
        Dictionary returned by mann_whitney_selected_features function
    df : pd.DataFrame, optional
        Original dataframe for additional context plots
    save_plots : bool
        Whether to save plots to files
    figsize : tuple
        Figure size for the plots
    """

    # Extract results
    all_results = results_dict['all_results']

    if all_results.empty:
        print("No significant results to plot!")
        return

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)

    # 1. P-value distribution (top left)
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(all_results['p_value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.8, label='p=0.05 threshold')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Effect size vs P-value (top middle)
    ax2 = plt.subplot(2, 3, 2)
    scatter = plt.scatter(all_results['p_value'], all_results['effect_size'],
                          c=all_results['stat'], cmap='viridis', alpha=0.7, s=60)
    plt.xlabel('P-value')
    plt.ylabel('Effect Size')
    plt.title('Effect Size vs P-value')
    plt.colorbar(scatter, label='U-statistic')
    plt.grid(True, alpha=0.3)

    # 3. Sample sizes comparison (top right)
    ax3 = plt.subplot(2, 3, 3)
    x = np.arange(len(all_results))
    width = 0.35
    plt.bar(x - width / 2, all_results['n_group1'], width, label='High-risk + No family history',
            color='lightcoral', alpha=0.8)
    plt.bar(x + width / 2, all_results['n_group2'], width, label='Low-risk group',
            color='lightblue', alpha=0.8)
    plt.xlabel('Feature Combination Index')
    plt.ylabel('Sample Size')
    plt.title('Sample Sizes by Group')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Obesity rates comparison (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(len(all_results))
    plt.bar(x - width / 2, all_results['mean_group1'], width, label='High-risk + No family history',
            color='lightcoral', alpha=0.8)
    plt.bar(x + width / 2, all_results['mean_group2'], width, label='Low-risk group',
            color='lightblue', alpha=0.8)
    plt.xlabel('Feature Combination Index')
    plt.ylabel('Obesity Rate')
    plt.title('Obesity Rates by Group')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Top 10 most significant results (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    top_10 = all_results.head(10)
    feature_labels = [f"{i + 1}: {', '.join(row['features'])}" for i, (_, row) in enumerate(top_10.iterrows())]
    y_pos = np.arange(len(feature_labels))

    plt.barh(y_pos, -np.log10(top_10['p_value']), color='gold', alpha=0.8)
    plt.yticks(y_pos, [f"{i + 1}" for i in range(len(feature_labels))])
    plt.xlabel('-log10(P-value)')
    plt.ylabel('Rank')
    plt.title('Top 10 Most Significant Results')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    # 6. Feature frequency in significant combinations (bottom right)
    ax6 = plt.subplot(2, 3, 6)

    # Count feature frequencies
    feature_counts = {}
    for _, row in all_results.iterrows():
        for feature in row['features']:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Sort by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    features, counts = zip(*sorted_features)

    plt.bar(range(len(features)), counts, color='lightgreen', alpha=0.8)
    plt.xlabel('Features')
    plt.ylabel('Frequency in Significant Combinations')
    plt.title('Feature Importance')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig('mann_whitney_results_overview.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Create a detailed results table plot
    plot_results_table(all_results)

    # Create volcano plot
    plot_volcano_plot(all_results, save_plots=save_plots)

    return fig


def plot_results_table(results_df, top_n=10, save_plots=True):
    """
    Create a detailed table visualization of top results
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Select top results
    top_results = results_df.head(top_n)

    # Prepare data for table
    table_data = []
    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        table_data.append([
            f"{idx}",
            f"{', '.join(row['features'])}",
            f"{row['p_value']:.2e}",
            f"{row['stat']:.1f}",
            f"{row['mean_group1']:.3f}/{row['mean_group2']:.3f}",
        ])

    # Column headers
    headers = ['Rank', 'Features', 'P-value', 'U-stat',
               'Obesity Rates']

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.05, 0.25, 0.1, 0.08, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title(f'Top {top_n} Most Significant Feature Combinations\n'
              'High-risk + No Family History vs Low-risk Group',
              fontsize=14, fontweight='bold', pad=20)

    if save_plots:
        plt.savefig('mann_whitney_results_table_for_two_symmetric_groups.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_volcano_plot(results_df, save_plots=False):
    """
    Create a volcano plot showing effect size vs significance
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate -log10(p-value)
    neg_log_p = -np.log10(results_df['p_value'])

    # Create scatter plot
    scatter = ax.scatter(results_df['effect_size'], neg_log_p,
                         c=results_df['stat'], cmap='viridis',
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

    # Add significance threshold line
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--',
               alpha=0.8, label='p=0.05 threshold')

    # Add labels for top results
    top_5 = results_df.head(5)
    for idx, (_, row) in enumerate(top_5.iterrows()):
        ax.annotate(f"{', '.join(row['features'])}",
                    (row['effect_size'], -np.log10(row['p_value'])),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax.set_xlabel('Effect Size (|Mean Difference|)')
    ax.set_ylabel('-log10(P-value)')
    ax.set_title('Volcano Plot: Effect Size vs Statistical Significance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('U-statistic')

    if save_plots:
        plt.savefig('mann_whitney_volcano_plot.png', dpi=300, bbox_inches='tight')

    plt.show()

