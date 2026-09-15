"""
Analysis code for: []


This script contains functions for:
1. Group comparison of gradient dispersion measures with covariate adjustment
2. Brain-behavior correlation analysis with permutation testing
3. Moderation analysis of age effects




"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration (modify paths and parameters as needed)
# =============================================================================

# File paths - UPDATE THESE FOR YOUR DATA
DATA_PATH = ''  # Path to your Excel file
SHEET_NAME = ''
OUTPUT_DIR = ''  # Path for output figures

# Plotting settings
FIGSIZE = (7.7, 5.65)
DPI = 300
PALETTE = ['#B291B5', '#3BA997']  # Young, Old
COLORS = {'Young': '#B291B5', 'Old': '#2C9F89'}

# Analysis parameters
N_PERM = 1000  # Number of permutations for correlation testing

# =============================================================================
# Part 1: Group Comparison Functions
# =============================================================================

def regress_out_covariates(y, covariate_data):
    """
    Residualize a variable with respect to covariates using OLS regression.
    
    Parameters
    ----------
    y : pd.Series
        Dependent variable to residualize
    covariate_data : pd.DataFrame
        Covariate matrix
    
    Returns
    -------
    np.ndarray
        Residuals after removing covariate effects
    """
    X = sm.add_constant(covariate_data)
    model = sm.OLS(y, X, missing='drop').fit()
    return model.resid


def compute_cohens_d(group1, group2):
    """
    Compute Cohen's d effect size for two independent groups.
    
    Parameters
    ----------
    group1, group2 : array-like
        Data for each group
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) +
         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
    )
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def group_comparison(data, feature, covariates, group_var='Age_Cat', 
                     groups=('Young', 'Old')):
    """
    Perform covariate-adjusted group comparison using t-test on residuals.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    feature : str
        Column name of feature to compare
    covariates : list
        List of covariate column names
    group_var : str
        Column name for group variable
    groups : tuple
        Group labels to compare
    
    Returns
    -------
    dict
        Dictionary containing t-statistic, p-value, and Cohen's d
    """
    subset = data.dropna(subset=[feature] + covariates + [group_var]).copy()
    
    # Compute residuals
    residuals = regress_out_covariates(subset[feature], subset[covariates])
    subset['residuals'] = residuals
    
    # Split by group
    group1_resid = subset[subset[group_var] == groups[0]]['residuals']
    group2_resid = subset[subset[group_var] == groups[1]]['residuals']
    
    # Statistics
    t_stat, p_value = ttest_ind(group1_resid, group2_resid)
    cohen_d = compute_cohens_d(group1_resid, group2_resid)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohen_d,
        'n_group1': len(group1_resid),
        'n_group2': len(group2_resid)
    }


def plot_violin_comparison(data, feature, covariates, display_label, 
                           stats_dict, output_path=None):
    """
    Create violin plot for group comparison with statistics annotation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    feature : str
        Column name of feature to plot
    covariates : list
        List of covariate column names
    display_label : str
        Label for y-axis
    stats_dict : dict
        Dictionary with t_statistic, p_value, cohens_d
    output_path : str, optional
        Path to save figure
    """
    sns.set_style("white")
    sns.set_context("talk")
    
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    subset = data.dropna(subset=[feature] + covariates + ['Age_Cat']).copy()
    
    ax = sns.violinplot(
        data=subset,
        x='Age_Cat', y=feature,
        palette=PALETTE,
        inner='box',
        linewidth=3,
        width=0.7,
    )
    
    # Statistics annotation
    y_max, y_min = subset[feature].max(), subset[feature].min()
    text_y = y_max + 0.1 * (y_max - y_min)
    
    p_str = "p < 0.001" if stats_dict['p_value'] < 0.001 else f"p = {stats_dict['p_value']:.3f}"
    
    plt.text(
        0.5, text_y,
        f"t = {stats_dict['t_statistic']:.2f}\n{p_str}\n$d$ = {stats_dict['cohens_d']:.2f}",
        ha='center', va='center', fontsize=22,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')
    )
    
    ax.set_xlabel("")
    ax.set_ylabel(display_label, fontsize=20)
    ax.set_xticklabels(['Younger Adults', 'Older Adults'], fontsize=18)
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1)
    plt.yticks(fontsize=18)
    sns.despine()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.show()


# =============================================================================
# Part 2: Brain-Behavior Correlation Functions
# =============================================================================

def residualize_for_correlation(y, covariates_df):
    """
    Residualize variable for correlation analysis.
    
    Parameters
    ----------
    y : pd.Series
        Variable to residualize
    covariates_df : pd.DataFrame
        Covariate data
    
    Returns
    -------
    tuple
        (residuals array, index of valid observations)
    """
    df = pd.concat([y, covariates_df], axis=1).dropna()
    X = pd.get_dummies(df[covariates_df.columns], drop_first=True)
    X = sm.add_constant(X)
    resid = sm.OLS(df[y.name].values, X).fit().resid
    return resid, df.index


def permutation_correlation_test(x, y, n_perm=1000, rng=None):
    """
    Permutation test for Pearson correlation with Fisher z confidence interval.
    
    Parameters
    ----------
    x, y : array-like
        Variables to correlate
    n_perm : int
        Number of permutations
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    dict
        Dictionary with r, p_perm, ci_lower, ci_upper
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    r_obs = np.corrcoef(x, y)[0, 1]
    r_perm = [np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)]
    p_perm = (1 + np.sum(np.abs(r_perm) >= abs(r_obs))) / (n_perm + 1)
    
    # Fisher z transformation for CI
    z = np.arctanh(r_obs)
    se = 1 / np.sqrt(len(x) - 3)
    ci_lower, ci_upper = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    
    return {
        'r': r_obs,
        'p_perm': p_perm,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def test_age_moderation(data, predictor, outcome, covariates, group_var='Age_Cat'):
    """
    Test moderation of brain-behavior relationship by age group.
    Returns standardized beta for interaction term.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    predictor : str
        Brain measure column name
    outcome : str
        Behavioral outcome column name
    covariates : list
        List of covariate column names
    group_var : str
        Grouping variable column name
    
    Returns
    -------
    dict
        Dictionary with beta_std, p_value, ci_lower, ci_upper
    """
    df = data[[predictor, outcome, group_var] + covariates].dropna().copy()
    
    # Standardize continuous variables
    for col in [predictor, outcome] + [c for c in covariates if df[c].dtype in ['float64', 'int64']]:
        if df[col].std() > 0:
            df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()
    
    # Build formula with standardized variables
    cov_terms = ' + '.join([f'C({c})' if df[c].dtype == 'object' else f'{c}_z' 
                            for c in covariates])
    formula = f"{outcome}_z ~ {predictor}_z * C({group_var}) + {cov_terms}"
    
    model = sm.OLS.from_formula(formula, data=df).fit()
    
    # Find interaction term
    int_term = [p for p in model.params.index if predictor + '_z' in p and group_var in p]
    if not int_term:
        return {'beta_std': np.nan, 'p_value': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    int_name = int_term[0]
    ci = model.conf_int().loc[int_name]
    
    return {
        'beta_std': model.params[int_name],
        'p_value': model.pvalues[int_name],
        'ci_lower': ci[0],
        'ci_upper': ci[1]
    }


def plot_correlation_by_group(data, predictor, outcome, covariates, 
                              predictor_label, outcome_label,
                              groups=('Young', 'Old'), n_perm=1000,
                              rng=None, output_path=None):
    """
    Generate scatter plot with regression lines by group.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    predictor : str
        Brain measure column name
    outcome : str
        Behavioral outcome column name
    covariates : list
        List of covariate column names
    predictor_label : str
        Display label for predictor
    outcome_label : str
        Display label for outcome
    groups : tuple
        Group labels
    n_perm : int
        Number of permutations
    rng : np.random.Generator, optional
        Random number generator
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    dict
        Statistics for each group and moderation test
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=350)
    results = {}
    
    for group in groups:
        sub = data[data['Age_Cat'] == group].dropna(subset=[predictor, outcome] + covariates)
        if len(sub) < 5:
            continue
        
        # Residualize
        x_res, idx_x = residualize_for_correlation(sub[predictor], sub[covariates])
        y_res, idx_y = residualize_for_correlation(sub[outcome], sub[covariates])
        idx = idx_x.intersection(idx_y)
        x_res = pd.Series(x_res, idx_x).loc[idx].values
        y_res = pd.Series(y_res, idx_y).loc[idx].values
        
        # Statistics
        corr_stats = permutation_correlation_test(x_res, y_res, n_perm, rng)
        r_ana, p_ana = pearsonr(x_res, y_res)
        
        results[group] = {
            'n': len(x_res),
            'r': r_ana,
            'p_analytic': p_ana,
            **corr_stats
        }
        
        # Plot
        ax.scatter(x_res, y_res, color=COLORS[group], edgecolor='black',
                   s=90, linewidth=1.2, alpha=0.95, label=f"{group} (n={len(x_res)})")
        
        # Regression line with CI
        X_ols = sm.add_constant(x_res)
        model = sm.OLS(y_res, X_ols).fit()
        xx = np.linspace(x_res.min(), x_res.max(), 150)
        pred = model.get_prediction(sm.add_constant(xx))
        ax.plot(xx, pred.predicted_mean, color=COLORS[group], linewidth=2.8)
        ci = pred.conf_int(alpha=0.05)
        ax.fill_between(xx, ci[:, 0], ci[:, 1], color=COLORS[group], alpha=0.2)
    
    # Moderation test
    results['moderation'] = test_age_moderation(data, predictor, outcome, covariates)
    
    # Formatting
    ax.set_xlabel(f"{predictor_label} (residuals)", fontsize=15, fontweight='bold')
    ax.set_ylabel(f"{outcome_label} (residuals)", fontsize=15, fontweight='bold')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.6)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='lower right')
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.show()
    
    return results


# =============================================================================
# Part 3: Multiple Comparison Correction
# =============================================================================

def fdr_correction(p_values, method='fdr_bh'):
    """
    Apply FDR correction to p-values.
    
    Parameters
    ----------
    p_values : array-like
        Raw p-values
    method : str
        Correction method (default: Benjamini-Hochberg)
    
    Returns
    -------
    tuple
        (reject array, corrected p-values)
    """
    reject, pvals_corrected, _, _ = multipletests(p_values, method=method)
    return reject, pvals_corrected


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    """
    Example workflow - modify paths and variable names for your data.
    """
    
    # Load data
    # data = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    # data = data[data['Age_Cat'].isin(['Young', 'Old'])]
    
    # Define features and covariates
    features = {
        "Global_Dispersion": "Global Dispersion",
        "Dispersion_N1": "Dispersion N1",
        # Add more features as needed
    }
    
    covariates_group = ['Gender', 'Mean_FD_Negative', 'Cortical_Thickness']
    

    # )
    
    print("Code loaded successfully. Modify paths and run analysis.")