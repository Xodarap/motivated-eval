import numpy as np
from scipy import stats
from inspect_ai.analysis.beta import (
    SampleMessages, SampleSummary, samples_df
)
import pandas as pd
from typing import List, Tuple
import glob
import os
from datetime import datetime


def find_most_recent_log_file(log_directories: List[str] = ["logs", "analysis"]) -> str:
    """
    Find the most recent motivated-interpretation evaluation log file.
    
    Args:
        log_directories: List of directories to search for log files
    
    Returns:
        Path to the most recent log file
    
    Raises:
        FileNotFoundError: If no log files are found
    """
    all_log_files = []
    
    for directory in log_directories:
        if os.path.exists(directory):
            pattern = os.path.join(directory, "*motivated-interpretation*.eval")
            log_files = glob.glob(pattern)
            all_log_files.extend(log_files)
    
    if not all_log_files:
        raise FileNotFoundError(f"No motivated-interpretation log files found in directories: {log_directories}")
    
    # Sort by filename (which contains timestamp) to get most recent
    all_log_files.sort(reverse=True)
    most_recent = all_log_files[0]
    
    print(f"Found {len(all_log_files)} log files. Using most recent: {most_recent}")
    return most_recent


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
   
    scores_df['metadata_pair_id'] = scores_df['metadata_pair_id'].astype(str) + "_" + scores_df['epoch'].astype(str)
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
    noise_effects = [(delta[1]) for delta in deltas]
    bias_effects = [(delta[0]) for delta in deltas]

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


def analyze_all_transcripts(transcripts_dir: str = "transcripts") -> pd.DataFrame:
    """
    Analyze all transcript files in the given directory and return a summary DataFrame.
    
    Args:
        transcripts_dir: Directory containing transcript files
        
    Returns:
        DataFrame with analysis results for each file
    """
    if not os.path.exists(transcripts_dir):
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")
    
    # Find all .eval files in the transcripts directory
    pattern = os.path.join(transcripts_dir, "*.eval")
    transcript_files = glob.glob(pattern)
    
    if not transcript_files:
        raise FileNotFoundError(f"No .eval files found in {transcripts_dir}")
    
    results_list = []
    
    for file_path in sorted(transcript_files):
        file_name = os.path.basename(file_path)
        model_name = file_name.replace('.eval', '')
        
        try:
            print(f"Analyzing {file_name}...")
            
            samples = samples_df(
                logs=file_path,
                columns=SampleSummary + SampleMessages
            )
            
            deltas = get_deltas(samples)
            results = analyze_motivated_reasoning_from_deltas(deltas)
            
            if "error" not in results:
                # Add file info to results
                results["model"] = model_name
                results["file"] = file_name
                results_list.append(results)
            else:
                print(f"Error analyzing {file_name}: {results['error']}")
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    if not results_list:
        raise ValueError("No files could be successfully analyzed")
    
    # Convert to DataFrame for easy tabular display
    df = pd.DataFrame(results_list)
    
    # Reorder columns for better display
    cols = ["model", "n_items", "mean_bias_effect", "mean_noise_effect", 
            "mean_difference", "t_statistic", "p_value", "cohens_d", 
            "significant_at_05", "interpretation"]
    df = df[cols]
    
    return df


def display_summary_table(df: pd.DataFrame):
    """Display a formatted summary table of all results."""
    print("\n" + "="*100)
    print("MOTIVATED REASONING ANALYSIS SUMMARY")
    print("="*100)
    
    # Round numerical columns for display
    display_df = df.copy()
    display_df = display_df.drop(columns=["significant_at_05", "interpretation"])
    numerical_cols = ["mean_bias_effect", "mean_noise_effect", "mean_difference", 
                     "t_statistic", "p_value", "cohens_d"]
    
    for col in numerical_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)
    
    # Print the table
    print(display_df.to_string(index=False, max_colwidth=30))
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    # Overall statistics
    significant_count = df["significant_at_05"].sum()
    total_count = len(df)
    
    print(f"Total models analyzed: {total_count}")
    print(f"Models showing significant motivated reasoning: {significant_count}")
    print(f"Percentage significant: {(significant_count/total_count)*100:.1f}%")
    
    if significant_count > 0:
        sig_df = df[df["significant_at_05"]]
        print(f"\nAmong significant results:")
        print(f"  Mean effect size (Cohen's d): {sig_df['cohens_d'].mean():.3f}")
        print(f"  Mean p-value: {sig_df['p_value'].mean():.4f}")
        print(f"  Mean difference: {sig_df['mean_difference'].mean():.3f}")


def output_csv(df: pd.DataFrame):
    """Output CSV data to stdout for importing into Google Sheets."""
    # Create a clean CSV version with rounded numbers
    csv_df = df.copy()
    numerical_cols = ["mean_bias_effect", "mean_noise_effect", "mean_difference", 
                     "t_statistic", "p_value", "cohens_d", "ci_95_lower", "ci_95_upper"]
    
    for col in numerical_cols:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].round(4)
    
    print("\n" + "="*100)
    print("CSV OUTPUT FOR GOOGLE SHEETS")
    print("="*100)
    print(csv_df.to_csv(index=False))


# Example usage
if __name__ == "__main__":
    try:
        # Analyze all transcript files
        results_df = analyze_all_transcripts()
        
        # Display summary table
        display_summary_table(results_df)
        
        # Output CSV for Google Sheets
        output_csv(results_df)
        
        # Optionally print detailed reports for each model
        print("\n" + "="*100)
        print("DETAILED REPORTS")
        print("="*100)
        
        for _, row in results_df.iterrows():
            print(f"\n{row['model'].upper()}:")
            print("-" * 50)
            results_dict = row.to_dict()
            print(create_summary_report(results_dict))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure transcript files exist in the transcripts/ directory.")
    except Exception as e:
        print(f"Error analyzing results: {e}")
        print("Please check that the transcript files contain valid motivated interpretation results.")
