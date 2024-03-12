import contextlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


os.chdir(os.path.dirname(__file__))

df = (
    pd.read_csv('ibl_results.csv')
    .query('is_single_shank')
    .query('is_v1_probe')
)


fig = plt.figure()
data = []
for _, _df in df.groupby('subject_id'):
    for _, group_df in _df.groupby('device'):
        if len(df) == 1:
            continue
        group_df.sort_values('session_start_time', ascending=True, inplace=True)
        num_good_units_by_day = group_df['num_good_units'].array
        a = np.full(10, np.nan)
        a[:len(group_df)] = (num_good_units_by_day - num_good_units_by_day[0])
        data.append(a)
data = np.array(data)
plt.plot(data.T, color='r', alpha=0.2)
plt.plot(np.nanmedian(data, axis=0), color='r', linestyle='--', lw=2)


plt.xlabel('session day')
plt.ylabel('change in number of good units')
plt.title('all insertions')

plt.savefig(f'good_units_all_insertions.png') 

