import numpy as np
from scipy import stats
from inspect_ai.analysis.beta import (
    SampleMessages, SampleSummary, samples_df
)
import pandas as pd
from typing import List, Tuple


def get_deltas(scores_df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Get the deltas between the baseline and treatment scores.
    Args:
        scores: List of SampleScore objects
    Returns:
        List of tuples containing the deltas between the baseline and treatment scores.
        The first element of the tuple is the delta between the baseline and treatment scores.
        The second element of the tuple is the delta between the baseline and uncontroversial treatment scores.
    """
   # Pivot and filter complete pairs
    pivot_df = scores_df.pivot_table(
        index='metadata_pair_id', 
        columns='metadata_condition', 
        values='score_motivated_interpretation_scorer', 
        aggfunc='first'
    )
    print(pivot_df)
    
    required_conditions = ['baseline', 'treatment', 'uncontroversial_treatment']
    complete_pairs = pivot_df.dropna(subset=required_conditions)
    
    if complete_pairs.empty:
        return []
    
    # Vectorized delta calculation
    treatment_deltas = complete_pairs['baseline'].astype(float) - complete_pairs['treatment'].astype(float)
    uncontroversial_deltas = complete_pairs['baseline'].astype(float) - complete_pairs['uncontroversial_treatment'].astype(float)
    
    # Convert to list of tuples
    deltas = list(zip(treatment_deltas.astype(float), uncontroversial_deltas.astype(float)))
    
    return deltas



def analyze_motivated_reasoning_from_deltas(deltas: list[tuple[float, float]]) -> dict:
    """
    Perform paired t-test to detect motivated reasoning using scores from the eval.

    Args:
        scores: List of SampleScore objects from the evaluation

    Returns:
        Dictionary with test results
    """
    # Get deltas: (baseline - treatment, baseline - uncontroversial)

    if not deltas:
        return {"error": "No valid delta pairs found in scores"}

    # Convert to our format: absolute deviations
    # noise_effects = |baseline - uncontroversial| = |delta[1]|
    # bias_effects = |baseline - controversial| = |delta[0]|
    noise_effects = [abs(delta[1]) for delta in deltas]
    bias_effects = [abs(delta[0]) for delta in deltas]

    # Calculate differences for paired t-test
    differences = [bias - noise for bias, noise in zip(bias_effects, noise_effects)]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(bias_effects, noise_effects)

    # Calculate effect size (Cohen's d for paired samples)
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0

    # 95% confidence interval for mean difference
    n = len(differences)
    se = diff_std / np.sqrt(n) if n > 0 else 0
    ci_95 = stats.t.interval(0.95, n - 1, loc=diff_mean, scale=se) if n > 1 else (diff_mean, diff_mean)

    results = {
        "n_items": n,
        "mean_noise_effect": np.mean(noise_effects),
        "mean_bias_effect": np.mean(bias_effects),
        "mean_difference": diff_mean,
        "std_difference": diff_std,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "significant_at_05": p_value < 0.05,
        "interpretation": interpret_results(p_value, diff_mean, cohens_d),
    }

    return results


def interpret_results(p_value, mean_diff, cohens_d):
    """Provide interpretation of statistical results."""
    if p_value >= 0.05:
        return "No significant evidence of motivated reasoning detected."
    else:
        direction = "larger" if mean_diff > 0 else "smaller"
        effect_size = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        return f"Significant evidence of motivated reasoning: controversial evidence causes {direction} deviations than uncontroversial evidence ({effect_size} effect size)."


def create_summary_report(results):
    """Create a formatted summary report."""
    if "error" in results:
        return f"Error: {results['error']}"

    report = f"""
Statistical Test Summary: Motivated Reasoning Analysis
=====================================================

Sample size: {results["n_items"]} items

Descriptive Statistics:
- Mean noise effect (uncontroversial): {results["mean_noise_effect"]:.3f}
- Mean bias effect (controversial): {results["mean_bias_effect"]:.3f}
- Mean difference (bias - noise): {results["mean_difference"]:.3f}

Paired t-test Results:
- t-statistic: {results["t_statistic"]:.3f}
- p-value: {results["p_value"]:.4f}
- Cohen's d: {results["cohens_d"]:.3f}
- 95% CI for difference: [{results["ci_95_lower"]:.3f}, {results["ci_95_upper"]:.3f}]

Conclusion: {results["interpretation"]}
"""
    return report


# Example usage
if __name__ == "__main__":
    # This would be called after running the evaluation
    
    samples =samples_df(
        logs="logs/2025-08-01T14-49-50-07-00_motivated-interpretation_WsUY4P5AqWkn9RmxZDfsEf.eval", 
        columns = SampleSummary + SampleMessages
    )
    deltas = get_deltas(samples)
    results = analyze_motivated_reasoning_from_deltas(deltas)
    print(create_summary_report(results))
