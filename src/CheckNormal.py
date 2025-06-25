import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def check_normality_qq(data, title='Q-Q Plot'):
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.grid(True)
    plt.show()


def check_normality_shapiro_wilks(data, alpha=0.05):
    stat, p = stats.shapiro(data)
    is_normal = p > alpha
    conclusion = "Data looks normal (fail to reject H0)" if is_normal else "Data is not normal (reject H0)"

    return {
        'Statistic': stat,
        'p-value': p,
        'Normal?': is_normal,
        'Conclusion': conclusion
    }

