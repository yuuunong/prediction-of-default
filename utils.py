import pandas as pd
from datetime.datetime import today
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def make_submit(test_df, features, model):
    today = today().strftime('%Y-%m-%d')
    submit_df = pd.read_csv('./data/sample_submission.csv')

    submit_df['채무 불이행 확률'] = model.predict(test_df[features])
    submit_df.to_csv(f'./data/submission_{today}.csv', index=False)



def base_model(train_df, features, target):
    # 데이터 분할 (입력 변수와 목표 변수)
    X = train_df[features]
    y = train_df[target]

    # 데이터 분할 (학습 데이터와 테스트 데이터)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

    return model

