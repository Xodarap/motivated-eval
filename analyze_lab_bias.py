import re
import numpy as np
from scipy import stats
from inspect_ai.analysis.beta import (
    SampleMessages, SampleSummary, samples_df, evals_df
)
import pandas as pd
from typing import List, Tuple
import glob
import os
from datetime import datetime

def extract_lab_name_regex(text):
    """Extract lab name using regex"""
    if pd.isna(text) or text.strip() == '':
        return 'No Exception'  # or 'Baseline' or whatever you prefer
    
    # Pattern: ", with the exception of [LAB_NAME], which"
    pattern = r', with the exception of ([^,]+), which'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1).strip()
    else:
        return 'None'

def analyze_all_transcripts(transcripts_dir: str = "transcripts/lab_bias/run-2025-08-04") -> pd.DataFrame:
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
    count = 0
    for file_path in sorted(transcript_files):
        file_name = os.path.basename(file_path)
        print(f"Analyzing {file_name}...")
        
        try:
            samples = samples_df(
                logs=file_path,
                columns=SampleSummary + SampleMessages
            )
            evals = evals_df(logs=file_path)
            if samples.empty:
                print(f"Warning: No samples found in {file_name}")
                continue
                
            samples['score_motivated_interpretation_scorer'] = samples['score_motivated_interpretation_scorer'].astype(float)
            samples['lab'] = samples['metadata_exception'].map(extract_lab_name_regex)
            
            # Group by lab and get mean scores
            lab_scores = samples.groupby('lab')['score_motivated_interpretation_scorer'].mean()
            
            model_name = evals.loc[0,'model'].split('/')[1]
            creator_lab_raw = evals.loc[0,'model'].split('/')[0]
            
            # Map creator lab names to match the lab names in exceptions
            creator_lab_mapping = {
                'anthropic': 'Anthropic',
                'google': 'Google', 
                'openai': 'OpenAI',
                'grok': 'xAI',
                'deepseek': 'DeepSeek'
            }
            creator_lab = creator_lab_mapping.get(creator_lab_raw.lower(), creator_lab_raw)
            
            # Add to results with model name
            for lab, score in lab_scores.items():
                results_list.append({
                    'model': model_name,
                    'creator_lab': creator_lab,
                    'lab': lab,
                    'mean_score': score
                })
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
        count = count + 1
        # if count > 3:
        #     break
    
    if not results_list:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Create DataFrame and pivot to table format
    results_df = pd.DataFrame(results_list)
    
    # Get all unique labs and sort them for consistent ordering
    all_labs = sorted(results_df['lab'].unique())
    
    # Pivot table by model vs lab
    pivot_df = results_df.pivot_table(index='model', columns='lab', values='mean_score', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=all_labs)
    
    # Add marginal means (row and column averages)
    pivot_df['Marginal'] = pivot_df.mean(axis=1)  # Row means
    marginal_row = pivot_df.mean(axis=0)  # Column means
    marginal_row.name = 'Marginal'
    pivot_df = pd.concat([pivot_df, marginal_row.to_frame().T])
    
    # Pivot table by creator_lab vs lab
    creator_pivot_df = results_df.pivot_table(index='creator_lab', columns='lab', values='mean_score', aggfunc='mean')
    creator_pivot_df = creator_pivot_df.reindex(columns=all_labs, index=all_labs)
    
    # Add marginal means for creator pivot
    creator_pivot_df['Marginal'] = creator_pivot_df.mean(axis=1)  # Row means
    creator_marginal_row = creator_pivot_df.mean(axis=0)  # Column means
    creator_marginal_row.name = 'Marginal'
    creator_pivot_df = pd.concat([creator_pivot_df, creator_marginal_row.to_frame().T])
    
    return pivot_df, creator_pivot_df, results_df

def test_diagonal_vs_offdiagonal(results_df: pd.DataFrame):
    """
    Test if diagonal elements (same creator_lab and evaluated_lab) differ from off-diagonal elements.
    """
    from scipy.stats import ttest_ind
    
    # Filter out rows where creator_lab or lab is missing
    clean_df = results_df.dropna(subset=['creator_lab', 'lab'])
    
    if clean_df.empty:
        print("No valid data for diagonal vs off-diagonal test")
        return
    
    # Separate diagonal and off-diagonal entries
    diagonal_scores = clean_df[clean_df['creator_lab'] == clean_df['lab']]['mean_score'].values
    offdiagonal_scores = clean_df[clean_df['creator_lab'] != clean_df['lab']]['mean_score'].values
    
    if len(diagonal_scores) == 0:
        print("No diagonal entries found for statistical test")
        return
    
    if len(offdiagonal_scores) == 0:
        print("No off-diagonal entries found for statistical test")
        return
    
    # Perform t-test
    t_stat, p_value = ttest_ind(diagonal_scores, offdiagonal_scores)
    
    # Calculate descriptive statistics
    diagonal_mean = np.mean(diagonal_scores)
    diagonal_std = np.std(diagonal_scores)
    offdiagonal_mean = np.mean(offdiagonal_scores)
    offdiagonal_std = np.std(offdiagonal_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(diagonal_scores) - 1) * diagonal_std**2 + 
                         (len(offdiagonal_scores) - 1) * offdiagonal_std**2) / 
                        (len(diagonal_scores) + len(offdiagonal_scores) - 2))
    cohens_d = (diagonal_mean - offdiagonal_mean) / pooled_std if pooled_std > 0 else 0
    
    print("\nDiagonal vs Off-Diagonal Statistical Test")
    print("=" * 60)
    print(f"Diagonal entries (same creator & evaluated lab):")
    print(f"  Count: {len(diagonal_scores)}")
    print(f"  Mean: {diagonal_mean:.3f}")
    print(f"  Std: {diagonal_std:.3f}")
    print(f"\nOff-diagonal entries (different creator & evaluated lab):")
    print(f"  Count: {len(offdiagonal_scores)}")
    print(f"  Mean: {offdiagonal_mean:.3f}")
    print(f"  Std: {offdiagonal_std:.3f}")
    print(f"\nT-test results:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    
    if p_value < 0.05:
        direction = "higher" if diagonal_mean > offdiagonal_mean else "lower"
        print(f"\n**Significant difference found (p < 0.05)**")
        print(f"Models show {direction} bias when evaluating their own creator lab vs other labs")
    else:
        print(f"\nNo significant difference found (p >= 0.05)")
        print(f"No evidence of systematic bias toward own creator lab")

def print_results_table(df: pd.DataFrame):
    """Print results in a nicely formatted table"""
    if df.empty:
        print("No results to display")
        return
    
    print("\nLab Bias Analysis Results")
    print("=" * 70)
    print(df.round(3).to_string())
    print("\nNote: Higher scores indicate stronger bias")
    print("Marginal column shows average bias per lab across all models")
    print("Marginal row shows average bias per model across all labs")

if __name__ == "__main__":
    model_results, creator_results, raw_results = analyze_all_transcripts()
    print("\nModel vs Lab Analysis:")
    print_results_table(model_results)
    print("\nCreator Lab vs Lab Analysis:")
    print_results_table(creator_results)
    
    # Statistical test for diagonal vs off-diagonal bias
    test_diagonal_vs_offdiagonal(raw_results)