import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize


# 1. 데이터 불러오기
data = pd.read_csv('azdigar nav sartirovka+0 delete.csv')


# 분석할 열들
columns = [
    'w08chronic_a', 'w08chronic_b', 'w08chronic_c',
    'w08chronic_d', 'w08chronic_e', 'w08chronic_f',
    'w08chronic_g', 'w08chronic_h', 'w08chronic_i',
    'w08chronic_k', 'w08chronic_l', 'w08chronic_m'
]

# 2. 데이터 클리닝 - 'w08chronic_m'에서 값이 3인 행 제거
cleaned_data = data[data['w08chronic_m'] != 3]


# 결과 저장을 위한 빈 데이터프레임 생성
result_table = pd.DataFrame(columns=['Target', 'Model', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Micro F1', 'Macro F1', 'Weighted F1', 'AUC'])


# 3. 각 타겟 열을 사용한 학습 및 평가
for target_column in columns:
    print(f'\n[{target_column}] 열을 예측:')
    print('==================================')
    
    # 4. 피처 데이터 정의 (타겟 열을 제외한 모든 chronic 컬럼 제외)
    X = cleaned_data.drop(columns=columns)
    y = cleaned_data[target_column].values


    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # 5. SelectKBest를 이용하여 피처 선택 (f_classif 사용)
    # 최적의 K 값을 결정하기 위해서는 그리드 서치나 반복 테스트가 필요할 수 있음 (여기선 10으로 설정)
    k = 10
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 타겟 클래스가 두 개 이상이면 AUC 계산을 위해 타겟을 이진화 (One-vs-Rest 방식)
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    if y_test_bin.shape[1] == 1:  # 클래스가 하나인 경우 (로지스틱 회귀처럼 처리 안 되게)
        y_test_bin = np.concatenate([1 - y_test_bin, y_test_bin], axis=1)
        

          # 6. 모델 정의
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'MLP': MLPClassifier(max_iter=2000, random_state=42)
    }
    
    for model_name, model in models.items():
        print(f'\n[{model_name}] 모델 평가:')
        print('--------------------------')

        # 모델 학습
        model.fit(X_train, y_train)

        # 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 정확도 계산
        accuracy = accuracy_score(y_test, y_pred)

        # precision, recall, f1-score 계산 (weighted average)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Micro, Macro, Weighted F1 계산
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        # AUC 계산 (다중 클래스의 경우 One-vs-Rest 방식 사용)
        try:
            auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr", average='weighted')
        except ValueError:
            auc = np.nan  # AUC 계산 불가 시 처리

        # 결과 추가
        new_row = pd.DataFrame({
            'Model': [model_name],
            'Target': [target_column],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1],
            'Accuracy': [accuracy],
            'Micro F1': [micro_f1],
            'Macro F1': [macro_f1],
            'Weighted F1': [weighted_f1],
            'AUC': [auc]
        })
        result_table = pd.concat([result_table, new_row], ignore_index=True)

# 8. 결과 테이블 출력
print("\n전체 모델 평가 결과:")
print(result_table)