# 이 코드는 김문수 후보의 지지율을 예측하기 위해 시계열 데이터를 기반으로 한 딥러닝 모델을 학습하고 평가하는 전체 과정을 담고 있음

# 1. 데이터 로딩 및 전처리: 긍정률 관련 피처와 지지율 데이터를 시간 기준으로 병합하고 정규화
# 2. 입력 시퀀스 생성: 모델이 학습할 수 있도록 SEQ_LEN(24시간) 입력과 PRED_LEN(144시간) 예측 대상 시퀀스를 생성
# 3. 모델 정의: 1D CNN + Self-Attention + MLP 구조의 HybridForecastNet을 구현
# 4. 모델 학습: 300 에폭 동안 MSELoss를 기준으로 모델을 학습
# 5. 모델 저장: 학습된 모델을 파일로 저장
# 6. 예측 및 평가: 마지막 입력 데이터를 기반으로 예측을 수행하고, 실제 지지율과 비교하여 R2, MSE, RMSE를 평가
# 7. 시각화: 예측 결과와 실제 결과를 그래프로 비교하여 확인

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import platform

#폰트 설정

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러옴
# 긍정률과 지지율 데이터를 불러와 시간 기준으로 병합
sentiment_df = pd.read_csv('편향완화_시간대별_긍정률_결과_김문수.csv')  # 긍정률
support_df = pd.read_csv('김문수_최종지지율.csv')   # 지지율
# 시간대 형식 통일 및 정렬
sentiment_df['시간대'] = pd.to_datetime(sentiment_df['시간대'])
support_df['시간'] = pd.to_datetime(support_df['시간'])

# 공통 시간대 기준으로 병합 가능한 구간만 추출
start = max(sentiment_df['시간대'].min(), support_df['시간'].min())
end = min(sentiment_df['시간대'].max(), support_df['시간'].max())
sentiment_df = sentiment_df[(sentiment_df['시간대'] >= start) & (sentiment_df['시간대'] <= end)]
support_df = support_df[(support_df['시간'] >= start) & (support_df['시간'] <= end)]
support_df = support_df.groupby('시간', as_index=False).mean()  # 중복 시간 평균 처리
merged = pd.merge(sentiment_df, support_df, left_on='시간대', right_on='시간').drop(columns='시간')

# 피처 선택 및 정규화
# 아래 단계에서는 학습에 사용할 피처만 선택하고, MinMaxScaler를 통해 0~1 사이로 정규화
#나머지 컬럼은 지지율과의 결정계수가 0~음수가 나와서 제외.
features = ['긍정률_이동평균', '긍정률']
target = '지지율'
data = merged[features + [target]]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Scaler 저장
import joblib
joblib.dump(scaler, 'scaler_LJS.pkl')

#시퀀스 구성
#과거 24시간 데이터를 입력으로 사용하고, 이후 144시간의 지지율을 예측하는 입력-출력 시퀀스를 구성
SEQ_LEN = 24
PRED_LEN = 144

X, y = [], []
for i in range(len(scaled) - SEQ_LEN - PRED_LEN + 1):
    X.append(scaled[i:i+SEQ_LEN, :-1])            # 입력 피처: 긍정률 2개 (24시간)
    y.append(scaled[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN, -1])  # 타깃: 지지율 (144시간)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# 모델 정의
#1D Conv, Self-Attention, MLP를 이용한 하이브리드 예측 모델을 정의
class HybridForecastNet(nn.Module):
    def __init__(self, input_dim=2, conv_dim=64, attn_heads=4, hidden_dim=128, output_len=144):
        super().__init__()

        # 1D Convolution Layer: 입력 시계열 데이터를 필터링하여 로컬 패턴 추출
        # input_dim: 입력 피처 수 (예: 2개 - 긍정률, 이동평균)
        # conv_dim: 컨볼루션을 통해 출력될 채널 수 (feature map 수)
        # kernel_size=3, padding=1: 시퀀스 길이를 유지하면서 3개 시간 구간 기준 패턴을 학습
        self.conv = nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1)

        # Multi-Head Attention Layer: 시간 흐름 간의 장기 의존성 파악
        # embed_dim: attention 연산에 사용될 차원 (conv_dim과 동일해야 함)
        # num_heads: 동시에 병렬적으로 다양한 관점에서 시계열을 바라봄
        self.attn = nn.MultiheadAttention(embed_dim=conv_dim, num_heads=attn_heads, batch_first=True)

        # Layer Normalization: 안정적인 학습을 위해 attention output 정규화
        self.norm = nn.LayerNorm(conv_dim)

        # MLP(다층 퍼셉트론)로 시계열 전체를 Flatten한 후, 144개 시간 구간을 예측
        # - 입력: conv_dim * SEQ_LEN (시계열 전체 feature map을 펼친 것)
        # - hidden_dim: 은닉층 크기
        # - output_len: 예측할 지지율 시점 수 (예: 144시간)
        self.mlp = nn.Sequential(
            nn.Flatten(),  # (B, T, C) → (B, T*C)
            nn.Linear(conv_dim * SEQ_LEN, hidden_dim),  # FC 레이어 1
            nn.ReLU(),  # 비선형 활성화
            nn.Linear(hidden_dim, output_len)  # FC 레이어 2 → 출력값 (지지율 144개)
        )

    def forward(self, x):
        # 입력 텐서 x: (B, T, C) → (B, C, T) 로 변환 (Conv1d는 channel-first 형태 요구)
        x = x.transpose(1, 2)

        # 1D Convolution 적용: 로컬 시계열 특징 추출
        x = self.conv(x).transpose(1, 2)  # 다시 (B, T, C) 형태로 복원

        # Self-Attention 적용: 시점 간 상호작용 모델링 (장기 의존성 학습)
        attn_out, _ = self.attn(x, x, x)

        # Residual Connection + LayerNorm: 학습 안정성과 gradient 흐름 개선
        x = self.norm(attn_out + x)

        # MLP를 통해 예측값 출력 (B, output_len)
        return self.mlp(x)


# 모델 학습
# 아래에서는 전체 데이터를 반복 학습하여 예측 성능을 향상
model = HybridForecastNet()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(300):
    model.train()
    total_loss = 0
    for i in range(len(X)):
        xb = X[i:i+1]# 배치 구성
        yb = y[i:i+1]
        pred = model(xb)# 예측값 계산
        loss = loss_fn(pred, yb)# MSE 손실 계산
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(X):.4f}")

# 모델 저장
# 학습이 끝난 모델을 파일로 저장합니다
torch.save(model.state_dict(), "hybrid_forecast_model_LJS.pt")
print("모델 저장 완료: hybrid_forecast_model_LJS.pt")

#예측 및 평가
# 마지막 입력 시퀀스를 기반으로 예측을 수행하고, 실제 지지율과 비교하여 정확도를 평가
model.eval()
with torch.no_grad():
    pred_scaled = model(X[-1:]).cpu().numpy().flatten()
    true_scaled = y[-1].cpu().numpy()

    # 역정규화를 위해 피처 2개 + 지지율(3열) 구성
    pred_full = scaler.inverse_transform(np.concatenate([
        np.zeros((PRED_LEN, len(features))),
        pred_scaled.reshape(-1, 1)
    ], axis=1))[:, -1]  # 예측 지지율만 추출

    true_full = scaler.inverse_transform(np.concatenate([
        np.zeros((PRED_LEN, len(features))),
        true_scaled.reshape(-1, 1)
    ], axis=1))[:, -1]  # 실제 지지율만 추출

#성능 평가 지표 출력
r2 = r2_score(true_full, pred_full)
mse = mean_squared_error(true_full, pred_full)
rmse = np.sqrt(mse)
print(f"R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# 결과 시각화
# 실제 vs 예측 지지율 그래프를 비교하여 확인
plt.figure(figsize=(14, 5))
plt.plot(true_full, label="실제 지지율", linewidth=2)
plt.plot(pred_full, label="예측 지지율", linestyle='--')
plt.title("하이브리드 모델 기반 지지율 예측 결과")
plt.xlabel("시간(Hour)")
plt.ylabel("지지율(%)")
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hybrid_forecast_result.png")
plt.show()
