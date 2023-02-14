import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import scipy

dataset = datasets.load_iris()
ext_target = dataset.target[:, None]
pd.DataFrame(
    np.concatenate((dataset.data, ext_target, dataset.target_names[ext_target]), axis=1),
    columns=dataset.feature_names + ['target label', 'target name'],
)

features = dataset.data
target = dataset.target

print(features.shape, target.shape)

loc0, scale0 = scipy.stats.laplace.fit(features[:, 0])
print(loc0, scale0)

