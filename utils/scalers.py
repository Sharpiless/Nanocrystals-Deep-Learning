import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer
)

class LogScaler:
    def transform(self, x):
        return np.log(x + 1)

    def fit_transform(self, x):
        return self.transform(x)

    def fit(self, x):
        return 


class MixedStandardScaler:

    def __init__(self):
        self.scaler = StandardScaler()

    def transform(self, x):
        return self.scaler.transform(np.log10(x + 1))

    def fit_transform(self, x):
        return self.scaler.fit_transform(np.log10(x + 1))

    def fit(self, x):
        self.scaler.fit(np.log10(x + 1))
        return 


class MixedRobustScaler:

    def __init__(self):
        self.scaler = RobustScaler()

    def transform(self, x):
        return self.scaler.transform(np.log10(x + 1))

    def fit_transform(self, x):
        return self.scaler.fit_transform(np.log10(x + 1))

    def fit(self, x):
        self.scaler.fit(np.log10(x + 1))
        return 


class MixedMinMaxScaler:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def transform(self, x):
        return self.scaler.transform(np.log10(x + 1))

    def fit_transform(self, x):
        return self.scaler.fit_transform(np.log10(x + 1))

    def fit(self, x):
        self.scaler.fit(np.log10(x + 1))
        return 

sklearn_scalers_dict = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "log": LogScaler,
    "mixed_standard": MixedStandardScaler,
    "mixed_robust": MixedRobustScaler,
    "mixed_minmax": MixedMinMaxScaler,
    "power": PowerTransformer
}