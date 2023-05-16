import numpy as np
from numba import njit
import pandas as pd
import vectorbtpro as vbt


"""
Rank the performances of each row in the input array.
Best performance is ranked n-1, worst performance is ranked 0.
"""
@njit
def rank_performances(performances):
    ranks = np.zeros(performances.shape)
    for i_row in range(performances.shape[0]):
        temp = performances[i_row].argsort()
        row_ranks = np.empty_like(temp)
        row_ranks[temp] = np.arange(len(performances[i_row]))
        ranks[i_row] = row_ranks
    
    for i in range(performances.shape[0]):
        for j in range(performances.shape[1]):
            if np.isnan(performances[i][j]):
                ranks[i][j] = np.nan
    return ranks


"""
Get the percentage change between each row and the previous row.
Returns the same shape as the input array.
periods: number of rows to look back
"""
@njit
def get_pct_change(close, periods=1):
    out_row = np.full_like(close, np.nan, dtype=np.double)
    out_row[periods:] = close[periods:] / close[:-periods] - 1
    return out_row


"""
Get the number of non-nan values in each row.
"""
@njit
def active_per_row(arr):
    rows, cols = arr.shape
    result = np.zeros(arr.shape)
    for row in range(rows):
        n = np.count_nonzero(~np.isnan(arr[row]))
        for col in range(cols):
            result[row][col] = n
    return result


"""
Indicator function
"""
@njit
def momentum_rank(close, period=1):
    pct = get_pct_change(close, period)
    return rank_performances(pct)


MomRank = vbt.IF(
    class_name='MomRank',
    short_name='mr',
    input_names=['close'],
    param_names=['period'],
    output_names=['rank']
).with_apply_func(
    momentum_rank, 
    takes_1d=False,  
    period=1
)


class _MomRank(MomRank):
    def is_top_percent(self, percentage: float):
        active = active_per_row(self.rank.values)
        return self.rank >= (active * percentage)
    
    def is_bottom_percent(self, percentage: float):
        active = active_per_row(self.rank.values)
        return self.rank < (active * percentage)


setattr(MomRank, "is_top_percent", _MomRank.is_top_percent)
setattr(MomRank, "is_bottom_percent", _MomRank.is_bottom_percent)