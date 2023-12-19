import os
import pickle

recsys_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
performance_path = os.path.join(recsys_directory, 'performance')
with open(os.path.join(performance_path, 'hitrates_avg_CF.pkl'), 'rb') as f:
    best_hr = pickle.load(f)

print(best_hr)