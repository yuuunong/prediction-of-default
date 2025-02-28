from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from service.utils import reset_seeds

@reset_seeds
def base_model(X, y):
    # 데이터 분할 (학습 데이터와 테스트 데이터)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 불균형 데이터 처리
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # hpo
    best_params = hpo(X_train, y_train)

    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # 모델 평가
    model_report(model, X_test, y_test)

    return model

def roc_auc_curve_plt(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def model_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print('Classification Report:')
    print(report)

    roc_auc_curve_plt(y_test, y_pred_proba)

@reset_seeds
def hpo(X_train, y_train):
    
    params = {
        'n_estimators': randint(10, 300),
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
    }

    rf = RandomForestClassifier(random_state=42)
    random_search_cv = RandomizedSearchCV(rf, param_distributions=params, cv=3, n_jobs=-1, random_state=42, n_iter=100, scoring='roc_auc')
    random_search_cv.fit(X_train, y_train)

    print('최적 하이퍼 파라미터: ', random_search_cv.best_params_)
    print('최고 예측 정확도: {:.4f}'.format(random_search_cv.best_score_))

    return random_search_cv.best_params_
