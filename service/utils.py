import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns


def make_submit(test_df, model):
    today = datetime.today().strftime('%Y-%m-%d')
    base_filename = f'./data/submission_{today}.csv'
    filename = base_filename
    counter = 1

    while os.path.exists(filename):
        filename = f'./data/submission_{today}_{counter}.csv'
        counter += 1

    submit_df = pd.read_csv('./data/sample_submission.csv')
    submit_df['채무 불이행 확률'] = model.predict_proba(test_df)[:, 1]
    submit_df.to_csv(filename, index=False)

def reset_seeds(func, seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 파이썬 환경변수 시드 고정
    np.random.seed(seed)

    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_func

def plot_features(train_df, columns):
    plt.figure(figsize=(15, 10))

    for i, column in enumerate(columns, 1):
        plt.subplot(6, 5, i)
        sns.histplot(train_df[column], kde=True)
        plt.title(column)

    plt.suptitle('Histograms of Specified Columns')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def feature_importance(model, X):
    # 피처 중요도 계산
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 중요도에 따라 정렬

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()