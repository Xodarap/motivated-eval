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
            creator_lab = evals.loc[0,'model'].split('/')[0]
            
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
        return pd.DataFrame()
    
    # Create DataFrame and pivot to table format
    results_df = pd.DataFrame(results_list)
    pivot_df = results_df.pivot_table(index='model', columns='lab', values='mean_score', aggfunc='mean')
    
    # Add marginal means (row and column averages)
    pivot_df['Marginal'] = pivot_df.mean(axis=1)  # Row means
    marginal_row = pivot_df.mean(axis=0)  # Column means
    marginal_row.name = 'Marginal'
    pivot_df = pd.concat([pivot_df, marginal_row.to_frame().T])
    
    return pivot_df

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
    results = analyze_all_transcripts()
    print_results_table(results)