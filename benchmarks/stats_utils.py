import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd


def paired_test(quasar, baseline, test="ttest"):
    """Perform a paired t-test or Wilcoxon signed-rank test.

    Parameters
    ----------
    quasar : array-like
        Results from QuASAr.
    baseline : array-like
        Results from baseline.
    test : str, optional
        'ttest' for paired t-test or 'wilcoxon' for Wilcoxon signed-rank test.
        Defaults to 'ttest'.

    Returns
    -------
    statistic : float
        Test statistic.
    p_value : float
        Associated p-value.
    """
    quasar = np.asarray(quasar)
    baseline = np.asarray(baseline)
    if test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(quasar, baseline)
    else:
        statistic, p_value = stats.ttest_rel(quasar, baseline)
    return statistic, p_value


def adjust_pvalues(p_values, method="bonferroni"):
    """Apply multiple comparison correction to p-values.

    Parameters
    ----------
    p_values : array-like
        Sequence of p-values to correct.
    method : str, optional
        'bonferroni' or 'fdr_bh'. Defaults to 'bonferroni'.

    Returns
    -------
    np.ndarray
        Array of corrected p-values.
    """
    p_values = np.asarray(p_values)
    _, corrected, _, _ = multipletests(p_values, method=method)
    return corrected


def cohen_d(quasar, baseline):
    """Compute Cohen's d for paired samples."""
    quasar = np.asarray(quasar)
    baseline = np.asarray(baseline)
    diff = quasar - baseline
    return diff.mean() / diff.std(ddof=1)


def stats_table(quasar_results, baseline_results_dict, test="ttest", correction="bonferroni"):
    """Return DataFrame with statistics, corrected p-values, and effect sizes.

    Parameters
    ----------
    quasar_results : array-like
        Results from QuASAr.
    baseline_results_dict : dict
        Mapping of baseline name to array-like results.
    test : str, optional
        Statistical test to use ('ttest' or 'wilcoxon'). Defaults to 'ttest'.
    correction : str, optional
        Multiple comparison correction ('bonferroni' or 'fdr_bh'). Defaults to 'bonferroni'.

    Returns
    -------
    pd.DataFrame
        Table containing baseline names, statistics, corrected p-values, and effect sizes.
    """
    records = []
    for name, baseline in baseline_results_dict.items():
        stat, p = paired_test(quasar_results, baseline, test=test)
        effect = cohen_d(quasar_results, baseline)
        records.append({"baseline": name, "statistic": stat, "p_value": p, "effect_size": effect})
    corrected = adjust_pvalues([r["p_value"] for r in records], method=correction)
    for rec, p in zip(records, corrected):
        rec["p_value"] = p
    return pd.DataFrame(records)
