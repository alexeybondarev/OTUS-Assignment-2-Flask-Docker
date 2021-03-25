import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

data = pd.read_csv('./heart.csv')

feature_cols = list(data.columns)
feature_cols.remove('target')
data_train, data_val, target_train, target_val = train_test_split(data[feature_cols], data['target'], 
                                                    train_size=0.8,
                                                    random_state=42)
scaler = StandardScaler()
model_lr = Pipeline(
    steps=[
        ("normalisation", scaler),
        (
            "classifier",
            LogisticRegressionCV(
                solver='saga', Cs=[2.2], penalty='l1', cv = 5, scoring="roc_auc", refit=True, random_state=0, max_iter=2000
            ),
        ),
    ]
)
model_lr.fit(data_train, target_train)
model_lr.predict(data_val)
print('model AUC: ', model_lr.score(data_val, target_val))

test_1 = list(data_val.iloc[[0]].values)
predict_prob = model_lr.predict_proba(test_1)
print('Predicted probability', predict_prob)
