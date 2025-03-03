import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def preprocessing(train_df, test_df):
    # 범주형 데이터 인코딩
    train_df = pd.get_dummies(train_df, columns=['주거 형태'])
    test_df = pd.get_dummies(test_df, columns=['주거 형태'])
    train_df.drop(['주거 형태_주택 담보 대출 (비거주 중)'], axis=1, inplace=True)
    test_df.drop(['주거 형태_주택 담보 대출 (비거주 중)'], axis=1, inplace=True)

    ordinal_mapping = {
        '1년 미만': 0,
        '1년': 1,
        '2년': 2,
        '3년': 3,
        '4년': 4,
        '5년': 5,
        '6년': 6,
        '7년': 7,
        '8년': 8,
        '9년': 9,
        '10년 이상': 10
    }
    train_df['현재 직장 근속 연수'] = train_df['현재 직장 근속 연수'].map(ordinal_mapping)
    test_df['현재 직장 근속 연수'] = test_df['현재 직장 근속 연수'].map(ordinal_mapping)

    train_df['대출 목적'] = train_df['대출 목적'].apply(lambda x: 1 if x == '부채 통합' else 0)
    test_df['대출 목적'] = test_df['대출 목적'].apply(lambda x: 1 if x == '부채 통합' else 0)

    train_df['대출 상환 기간'] = train_df['대출 상환 기간'].apply(lambda x: 0 if x == '단기 상환' else 1)
    test_df['대출 상환 기간'] = test_df['대출 상환 기간'].apply(lambda x: 0 if x == '단기 상환' else 1)

    train_df['체납 세금 압류 횟수'] = train_df['체납 세금 압류 횟수'].apply(lambda x: 0 if x == 0 else 1)
    test_df['체납 세금 압류 횟수'] = test_df['체납 세금 압류 횟수'].apply(lambda x: 0 if x == 0 else 1)

    train_df['신용 문제 발생 횟수'] = train_df['신용 문제 발생 횟수'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))
    test_df['신용 문제 발생 횟수'] = test_df['신용 문제 발생 횟수'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))  

    train_df['개인 파산 횟수'] = train_df['개인 파산 횟수'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))
    test_df['개인 파산 횟수'] = test_df['개인 파산 횟수'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))
    
    # 수치형 데이터
    columns = [
        '연간 소득', '개설된 신용계좌 수', '신용 거래 연수', '최대 신용한도',
        '마지막 연체 이후 경과 개월 수', '현재 대출 잔액', '현재 미상환 신용액',
        '월 상환 부채액', '신용 점수'
    ]

    outlier_columns = [
        '연간 소득', '개설된 신용계좌 수', '신용 거래 연수',
        '마지막 연체 이후 경과 개월 수', '현재 대출 잔액', '현재 미상환 신용액',
        '월 상환 부채액', '신용 점수'
    ]
    
    # 수치형 데이터 이상치 처리
    #train_df = remove_outliers(train_df, outlier_columns)
    
    # '최대 신용한도' 이상치 처리
    #Q1 = train_df['최대 신용한도'].quantile(0.25)
    #Q3 = train_df['최대 신용한도'].quantile(0.75)
    #IQR = Q3 - Q1
    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 + 1.5 * IQR
    #train_df['최대 신용한도_outlier'] = train_df['최대 신용한도'].apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0)
    #median_value = train_df['최대 신용한도'].median()
    #train_df.loc[train_df['최대 신용한도_outlier'] == 1, '최대 신용한도'] = median_value
    #test_df['최대 신용한도_outlier'] = test_df['최대 신용한도'].apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0)
    #test_df.loc[test_df['최대 신용한도_outlier'] == 1, '최대 신용한도'] = median_value

    # 수치형 데이터 transform
    pt = PowerTransformer(method='yeo-johnson')
    train_df[columns] = pt.fit_transform(train_df[columns])
    test_df[columns] = pt.transform(test_df[columns])
    
    # 스케일링
    scaler = StandardScaler()
    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])

    train_df_target = train_df['채무 불이행 여부']
    train_df_features = train_df.drop(['채무 불이행 여부', 'UID'], axis=1)
    test_df = test_df.drop(['UID'], axis=1)

    return train_df_features, train_df_target, test_df