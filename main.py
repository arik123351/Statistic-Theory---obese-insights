import pandas as pd
from src.CleanData import CleanData
from src.CheckNormal import check_normality_qq, check_normality_shapiro_wilks
from src.DivideToGroupsChi2 import chi2_best_combinations_excluding_weight_family, compare_combinations_detailed
from src.DivideToGroupsLogModel import evaluate_feature_subgroups, plot_top_wald_statistics, plot_comparison_with_base
from src.MannWhitneyTest import mann_whitney_selected_features, print_all_significant_groups, plot_mann_whitney_results, mann_whitney_for_family_not_obese, mann_whitney_for_family_obese

df = CleanData()

categorical_cols = [col for col in df.columns if df[col].nunique() <= 30 and col != "is_obese"]
continuous_cols = [col for col in df.columns if df[col].nunique() > 30 and col != "is_obese"]

# for col in continuous_cols:
#     check_normality_qq(df[col])
#     print(check_normality_shapiro_wilks(df[col]))
#
# results = chi2_best_combinations_excluding_weight_family(df, max_features=4)
# detailed_results = compare_combinations_detailed(df, top_combinations=3)


# results, base_model, top_5_significant = evaluate_feature_subgroups(df)
# plot_top_wald_statistics(top_5_significant)
# plot_comparison_with_base(top_5_significant)

selected = ['FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
            'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS_Walking',
            'MTRANS_Public_Transportation', 'MTRANS_Motorbike',
            'MTRANS_Bike', 'MTRANS_Automobile']

results = mann_whitney_selected_features(df, selected_features=selected)
significant_groups = print_all_significant_groups(df)
plot_mann_whitney_results(results, df)

for prec in range(50, 100, 10):
    results = mann_whitney_for_family_not_obese(df, selected_features=selected,percentile = prec)
    plot_mann_whitney_results(results, df)

results = mann_whitney_for_family_obese(df, selected_features=selected)
plot_mann_whitney_results(results, df)



