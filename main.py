import pandas as pd
from src.CleanData import CleanData
from src.CheckNormal import check_normality_qq, check_normality_shapiro_wilks
from src.DivideToGroupsChi2 import chi2_best_combinations_excluding_weight_family, compare_combinations_detailed
from src.DivideToGroupsLogModel import evaluate_feature_subgroups, plot_top_wald_statistics, plot_comparison_with_base
df = CleanData()

categorical_cols = [col for col in df.columns if df[col].nunique() <= 10 and col != "is_obese"]
continuous_cols = [col for col in df.columns if df[col].nunique() > 10 and col != "is_obese"]

# for col in continuous_cols:
#     check_normality_qq(df[col])
#     print(check_normality_shapiro_wilks(df[col]))

# results = chi2_best_combinations_excluding_weight_family(df, max_features=4)
detailed_results = compare_combinations_detailed(df, top_combinations=3)


# results, base_model, top_5_significant = evaluate_feature_subgroups(df)
# plot_top_wald_statistics(top_5_significant)
# plot_comparison_with_base(top_5_significant)
