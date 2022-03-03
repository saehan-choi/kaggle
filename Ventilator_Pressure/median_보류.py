import pandas as pd
import numpy as np

files = ['../input/gb-vpp-pulp-fiction/median_submission.csv',
         '../input/basic-ensemble-of-public-notebooks/submission_median.csv',
         '../input/gaps-features-tf-lstm-resnet-like-ff/sub.csv']

sub = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
sub['pressure'] = np.median(np.concatenate([pd.read_csv(f)['pressure'].values.reshape(-1, 1) for f in files], axis=1), axis=1)
sub.to_csv('submission.csv', index=False)
sub.head(5)
