from scipy.stats import shapiro, f_oneway, kruskal, pearsonr, spearmanr
import numpy as np
import math


# Check whether the data follows a normal distribution
def check_normality(scores):
    scores = list(scores)

    # If all values are equal or less than 3 values, we consider the test unsuitable
    if len(scores) < 3 or len(set(scores)) <= 1:
        return {
            "statistic": None,
            "p_value": None,
            "is_normal": None,
            "note": "Normality test not applicable (too few values or constant input)."
        }

    # Implementing the Shapiro-Wilk test
    stat, p = shapiro(scores)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > 0.05)
    }


# Comparing three sets of values
def compare_three_groups(group1, group2, group3):
    normality_1 = check_normality(group1)["is_normal"]
    normality_2 = check_normality(group2)["is_normal"]
    normality_3 = check_normality(group3)["is_normal"]

    # Use ANOVA if all groups are normal
    if normality_1 is True and normality_2 is True and normality_3 is True:
        stat, p = f_oneway(group1, group2, group3)
        return {
            "test": "ANOVA",
            "statistic": float(stat),
            "p_value": float(p)
        }
    # Otherwise use Kruskal-Wallis
    else:
        stat, p = kruskal(group1, group2, group3)
        return {
            "test": "Kruskal-Wallis",
            "statistic": float(stat),
            "p_value": float(p)
        }


# Cohen's d calculation to measure the effect size between two groups
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


# Correlation analysis between two lists of values
def correlation_analysis(x_scores, y_scores, method="pearson"):
    x_scores = list(x_scores)
    y_scores = list(y_scores)

    # If one of the inputs is constant the relationship is undefined
    if len(set(x_scores)) <= 1 or len(set(y_scores)) <= 1:
        return {
            "method": method.capitalize(),
            "correlation": None,
            "p_value": None,
            "note": "Correlation not defined because one input is constant."
        }

    if method == "pearson":
        stat, p = pearsonr(x_scores, y_scores)
        return {
            "method": "Pearson",
            "correlation": float(stat),
            "p_value": float(p)
        }
    else:
        stat, p = spearmanr(x_scores, y_scores)
        return {
            "method": "Spearman",
            "correlation": float(stat),
            "p_value": float(p)
        }