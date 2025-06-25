import pandas as pd
from src.CleanData import CleanData
from src.CheckNormal import check_normality_qq, check_normality_shapiro_wilks
from src.DivideToGroups import chi2_best_subgroup_with_the_best_8_cols, wald_best_subgroup_with_the_best_8_cols
df = CleanData()

categorical_cols = [col for col in df.columns if df[col].nunique() <= 10 and col != "is_obese"]
continuous_cols = [col for col in df.columns if df[col].nunique() > 10 and col != "is_obese"]

# for col in continuous_cols:
#     check_normality_qq(df[col])
#     print(check_normality_shapiro_wilks(df[col]))

results_chi2 = chi2_best_subgroup_with_the_best_8_cols(df)

results_wald = wald_best_subgroup_with_the_best_8_cols(df)

print(df.head())