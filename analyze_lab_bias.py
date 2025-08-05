import re
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
        print(f"Analyzing {file_name}...")
        
        samples = samples_df(
            logs="transcripts/lab_bias/gpt-4.1-mini.eval",
            columns=SampleSummary + SampleMessages
        )
        samples['score_motivated_interpretation_scorer'] = samples['score_motivated_interpretation_scorer'].astype(float)
        samples['lab'] = samples['metadata_exception'].map(extract_lab_name_regex)
        
        return samples.groupby('lab')['score_motivated_interpretation_scorer'].mean()

if __name__ == "__main__":
    print(analyze_all_transcripts())