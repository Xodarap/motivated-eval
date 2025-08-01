"""
Tests for the bias_score metric.
"""

import pytest
from motivated_interpretation import bias_score
from inspect_ai.scorer import SampleScore, Score, CORRECT, INCORRECT


def create_sample_score(score_value, pair_id, sample_metadata=None):
    """Helper function to create SampleScore objects for testing."""
    if sample_metadata is None:
        sample_metadata = {"pair_id": pair_id}
    
    # Create a mock sample object with metadata
    class MockSample:
        def __init__(self, metadata):
            self.metadata = metadata
    
    sample = MockSample(sample_metadata)
    score = Score(value=score_value)
    
    return SampleScore(
        sample_id="test_id",
        sample=sample,
        sample_metadata=sample_metadata,
        score=score
    )


class TestBiasScore:
    """Test class for bias_score metric."""
    
    def test_bias_score_basic_calculation(self):
        """Test basic bias score calculation with integer values."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(1, "pair1"),  # First score in pair1
            create_sample_score(0, "pair1"),  # Second score in pair1 -> delta = 1 - 0 = 1
            create_sample_score(1, "pair2"),  # First score in pair2
            create_sample_score(1, "pair2"),  # Second score in pair2 -> delta = 1 - 1 = 0
        ]
        
        result = metric_func(scores)
        # Implementation divides by complete pairs (2), so (1 + 0) / 2 = 0.5
        expected = (1 + 0) / 2
        assert result == expected
    
    def test_bias_score_with_integer_conversion(self):
        """Test bias score with values that convert to integers."""
        metric_func = bias_score()
        
        # The implementation uses int() conversion, so float values get truncated
        scores = [
            create_sample_score(1, "pair1"),
            create_sample_score(0, "pair1"),  # delta = 1 - 0 = 1
            create_sample_score(2, "pair2"),
            create_sample_score(1, "pair2"),  # delta = 2 - 1 = 1
        ]
        
        result = metric_func(scores)
        expected = (1 + 1) / 2  # Divided by complete pairs
        assert result == expected
    
    def test_bias_score_negative_bias(self):
        """Test bias score with negative bias (second score higher than first)."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(0, "pair1"),
            create_sample_score(1, "pair1"),  # delta = 0 - 1 = -1
            create_sample_score(1, "pair2"),
            create_sample_score(2, "pair2"),  # delta = 1 - 2 = -1
        ]
        
        result = metric_func(scores)
        expected = (-1 + -1) / 2  # Divided by complete pairs
        assert result == expected
    
    def test_bias_score_mixed_bias(self):
        """Test bias score with mixed positive and negative bias."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(2, "pair1"),
            create_sample_score(1, "pair1"),  # delta = 2 - 1 = 1
            create_sample_score(1, "pair2"),
            create_sample_score(3, "pair2"),  # delta = 1 - 3 = -2
        ]
        
        result = metric_func(scores)
        expected = (1 + -2) / 2  # Divided by complete pairs
        assert result == expected
    
    def test_bias_score_single_pair(self):
        """Test bias score with only one pair."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(3, "pair1"),
            create_sample_score(1, "pair1"),  # delta = 3 - 1 = 2
        ]
        
        result = metric_func(scores)
        # Divided by complete pairs (1)
        assert result == 2.0
    
    def test_bias_score_incomplete_pairs_ignored(self):
        """Test that incomplete pairs (not exactly 2 scores) are ignored."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(4, "pair1"),
            create_sample_score(1, "pair1"),  # Complete pair: delta = 4 - 1 = 3
            create_sample_score(2, "pair2"),  # Incomplete pair (only 1 score)
            create_sample_score(5, "pair3"),
            create_sample_score(2, "pair3"),
            create_sample_score(1, "pair3"),  # Incomplete pair (3 scores)
        ]
        
        result = metric_func(scores)
        # Only pair1 contributes (delta=3), divided by complete pairs (1)
        expected = 3.0  # Now correctly divides by number of complete pairs
        assert result == expected
    
    def test_bias_score_missing_pair_id_raises_error(self):
        """Test that missing pair_id raises KeyError."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(1, "pair1"),
            create_sample_score(0, None),  # Missing pair_id
        ]
        
        with pytest.raises(KeyError, match="SampleScore missing 'pair_id' in sample_metadata"):
            metric_func(scores)
    
    def test_bias_score_none_sample_metadata_skipped(self):
        """Test that scores with None sample_metadata are skipped."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(3, "pair1"),
            create_sample_score(1, "pair1"),  # Complete pair: delta = 3 - 1 = 2
            create_sample_score(2, "pair2", sample_metadata=None),  # Skipped
        ]
        
        result = metric_func(scores)
        # Only pair1 contributes (delta=2), divided by complete pairs (1)
        expected = 2.0
        assert result == expected
    
    def test_bias_score_empty_scores_list(self):
        """Test bias score with empty scores list."""
        metric_func = bias_score()
        
        scores = []
        
        # This should return 0/0 which results in ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            metric_func(scores)
    
    def test_bias_score_no_complete_pairs(self):
        """Test bias score when no complete pairs exist."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(1, "pair1"),  # Only one score for pair1
            create_sample_score(2, "pair2"),  # Only one score for pair2
        ]
        
        # This should return 0/0 which results in ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            metric_func(scores)
    
    def test_bias_score_large_dataset(self):
        """Test bias score with a larger dataset of integers."""
        metric_func = bias_score()
        
        scores = []
        expected_deltas = []
        
        # Create 10 pairs with known deltas (using integers)
        for i in range(1, 11):  # Start from 1 to avoid 0/0 issues
            score1 = i * 2
            score2 = i
            delta = score1 - score2  # Will be i each time
            expected_deltas.append(delta)
            
            scores.extend([
                create_sample_score(score1, f"pair{i}"),
                create_sample_score(score2, f"pair{i}"),
            ])
        
        result = metric_func(scores)
        expected = sum(expected_deltas) / len(expected_deltas)
        assert abs(result - expected) < 1e-10  # Account for floating point precision
    
    def test_bias_score_order_dependence(self):
        """Test that the order of scores within pairs matters - first score minus second score."""
        metric_func = bias_score()
        
        # First arrangement: 0.8 first, 0.3 second for pair1
        scores1 = [
            create_sample_score(0.8, "pair1"),
            create_sample_score(0.3, "pair1"),  # delta = 0.8 - 0.3 = 0.5
            create_sample_score(0.9, "pair2"),
            create_sample_score(0.6, "pair2"),  # delta = 0.9 - 0.6 = 0.3
        ]
        
        # Second arrangement: 0.3 first, 0.8 second for pair1
        scores2 = [
            create_sample_score(0.3, "pair1"),
            create_sample_score(0.8, "pair1"),  # delta = 0.3 - 0.8 = -0.5
            create_sample_score(0.6, "pair2"),
            create_sample_score(0.9, "pair2"),  # delta = 0.6 - 0.9 = -0.3
        ]
        
        result1 = metric_func(scores1)  # (0.5 + 0.3) / 2 = 0.4
        result2 = metric_func(scores2)  # (-0.5 + -0.3) / 2 = -0.4
        
        # Results should be negatives of each other
        assert result1 == -result2
        assert result1 == 0.4
        assert result2 == -0.4
    
    def test_bias_score_with_float_values(self):
        """Test bias score with float values (now supported)."""
        metric_func = bias_score()
        
        scores = [
            create_sample_score(0.8, "pair1"),
            create_sample_score(0.3, "pair1"),  # delta = 0.8 - 0.3 = 0.5
            create_sample_score(0.9, "pair2"),
            create_sample_score(0.7, "pair2"),  # delta = 0.9 - 0.7 = 0.2
        ]
        
        result = metric_func(scores)
        expected = (0.5 + 0.2) / 2
        assert abs(result - expected) < 1e-10