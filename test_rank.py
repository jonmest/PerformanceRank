from rank import rank_performances, get_pct_change, active_per_row, momentum_rank
import pandas as pd

def test_pct_change():
    df = pd.DataFrame({'AAPL': [1, 2, 4, 8, 16], 'GME': [16, 8, 4, 2, 1]})
    expected_pct = pd.DataFrame({'AAPL': [np.nan, 1, 1, 1, 1], 'GME': [np.nan, -0.5, -0.5, -0.5, -0.5]}).values
    actual_pct = get_pct_change(df.values)
    assert np.allclose(actual_pct, expected_pct, equal_nan=True)

def test_rank_performances():
    performances = pd.DataFrame({'AAPL': [np.nan, 1, 1, 1, 1], 
                                 'GME': [np.nan, -0.5, -0.5, -0.5, -0.5],
                                 'GOOGL': [np.nan, 1.2, -0.1, 1.2, -0.6]}).values
    expected_performance_ranks = pd.DataFrame({'AAPL': [np.nan, 1, 2, 1, 2], 
                                               'GME': [np.nan, 0, 0, 0, 1],
                                               'GOOGL': [np.nan, 2, 1, 2, 0]}).values
    actual_performance_ranks = rank_performances(performances)
    assert np.array_equal(actual_performance_ranks, expected_performance_ranks, equal_nan=True)
