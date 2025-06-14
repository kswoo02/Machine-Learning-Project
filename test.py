# 이 코드는 기존 학습된 시계열 예측 모델을 활용하여 후보의 긍정률 데이터를 기반으로 마지막 144개 시간 구간에 대한 지지율을 예측하는 프로그램

# 주요 단계:
# 1. 데이터 로드 및 정렬: 긍정률 CSV 파일 로드 후 시간대 정렬
# 2. 스케일러 로드: 학습 당시 사용한 MinMaxScaler 로드
# 3. 입력 데이터 전처리: 마지막 144구간에 대한 시퀀스 구성
# 4. 모델 로드 및 예측: 학습된 모델 로드 후 예측 수행
# 5. 역정규화 및 결과 추출: 스케일러로 예측값 역변환
# 6. 시각화 및 저장: 그래프 및 CSV 저장

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import platform
import joblib

#폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  

# 파라미터
SEQ_LEN = 24  # 입력 시퀀스 길이 (24시간)
PRED_LEN = 144  # 예측 시퀀스 길이 (144시간)
FEATURES = ['긍정률_이동평균', '긍정률']  # 사용할 입력 피처 리스트

#모델 정의
class HybridForecastNet(nn.Module):
    def __init__(self, input_dim=2, conv_dim=64, attn_heads=4, hidden_dim=128, output_len=144):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1)  # conv1d로 국소 패턴 추출
        self.attn = nn.MultiheadAttention(embed_dim=conv_dim, num_heads=attn_heads, batch_first=True)  # self-attention
        self.norm = nn.LayerNorm(conv_dim)  # layernorm으로 안정화
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim * SEQ_LEN, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T) 형태로 변환
        x = self.conv(x).transpose(1, 2)  # conv1d → 다시 (B, T, C)
        attn_out, _ = self.attn(x, x, x)  # self-attention 적용
        x = self.norm(attn_out + x)  # residual + layernorm
        return self.mlp(x)  # mlp 예측값 출력

#CSV 입력
df = pd.read_csv('merged_김문수_긍정률_5월13일~6월1일.csv')  # csv 파일 로드
df['시간대'] = pd.to_datetime(df['시간대'])  # 시간대 컬럼 datetime 변환
df = df.sort_values('시간대').reset_index(drop=True)  

#스케일러 불러오기
scaler: MinMaxScaler = joblib.load('scaler_LJS.pkl')  # 학습 당시 scaler 불러오기

#예측할 타깃 시간 범위 (마지막 144구간)
target_times = df['시간대'].iloc[-PRED_LEN:].reset_index(drop=True)  # 예측 시간대 추출

# X 시퀀스 구성
X_list = []
temp = df[FEATURES].copy()  # 입력 피처 복사
temp['지지율'] = 0  # 스케일러 입력을 위한 더미지지율 추가
scaled = scaler.transform(temp)    # 전체 데이터 스케일링

for i in range(len(scaled) - PRED_LEN, len(scaled)):
    X_seq = scaled[i - SEQ_LEN:i, :-1]  # 24시간 시퀀스 추출
    X_list.append(X_seq)

X_input = torch.tensor(np.array(X_list), dtype=torch.float32)  # 텐서 변환

#모델 불러오기 및 예측
model = HybridForecastNet(input_dim=2)
model.load_state_dict(torch.load("hybrid_forecast_model_LJS.pt", map_location='cpu'))  # 학습된 모델 로드
model.eval()

with torch.no_grad():
    preds_scaled = model(X_input).cpu().numpy()  # 예측 수행

#역정규화 후 예측값 추출
preds_full = []
for i in range(PRED_LEN):
    dummy = np.zeros((PRED_LEN, len(FEATURES)))
    combined = np.concatenate([dummy, preds_scaled[i].reshape(-1, 1)], axis=1)  # 스케일러 입력 형식 맞춤
    restored = scaler.inverse_transform(combined)[:, -1]  # 역정규화 지지율만 추출
    preds_full.append(restored[i])

# 시각화  저장
plt.figure(figsize=(14, 5))
plt.plot(target_times, preds_full, label="예측 지지율", linestyle='--', color='tab:blue')
plt.title("마지막 144개 시점에 대한 지지율 예측")
plt.xlabel("시간")
plt.ylabel("지지율 (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("마지막144개_지지율예측결과.png")
plt.show()

#CSV 저장
result_df = pd.DataFrame({
    '시간대': target_times,
    '예측지지율': preds_full
})
result_df.to_csv("마지막144개_지지율예측결과.csv", index=False, encoding='utf-8-sig')
print("CSV 저장 완료: 마지막144개_지지율예측결과.csv")
