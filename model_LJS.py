# 이 코드는 지지율을 예측하기 위한 하이브리드 딥러닝 모델 학습 코드임
# 김문수 후보와 동일한 HybridForecastNet 구조를 사용하지만, 이준석 후보는 긍정률에 비해 실제 지지율이 훨씬 낮기 때문에
# 별도의 모델로 분리 학습하며, 피처-타깃 간의 스케일 격차 문제를 방지하기 위해 학습 스케일러도 별도로 저장




# 전체 흐름:
# 1. 데이터 로딩 및 정렬: 긍정률과 지지율 데이터를 시간 기준으로 정렬 및 병합
# 2. 시계열 입력 구성: 24시간 입력 ➜ 144시간 지지율 예측
# 3. 모델 정의: 1D CNN + Self-Attention + MLP로 구성된 하이브리드 예측 모델 정의
# 4. 학습 루프: 전체 데이터를 대상으로 300에폭 학습 수행
# 5. 저장: 모델과 스케일러를 각각 파일로 저장
# 6. 예측 및 평가: 마지막 시점 입력 기준으로 예측하여 실제값과 비교(R2, MSE, RMSE)
# 7. 시각화: 예측값과 실제 지지율 비교 시각화 그래프 생성 및 저장

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import platform
import joblib

#폰트 설정
# 그래프 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

#데이터 불러오기
# 감정 분석 기반 긍정률 데이터와 지지율 데이터를 불러온 후, 공통 시간 구간 기준으로 병합
sentiment_df = pd.read_csv('편향완화_시간대별_긍정률_결과_이준석.csv')
support_df = pd.read_csv('이준석_최종지지율.csv')

sentiment_df['시간대'] = pd.to_datetime(sentiment_df['시간대'])
support_df['시간'] = pd.to_datetime(support_df['시간'])

start = max(sentiment_df['시간대'].min(), support_df['시간'].min())
end = min(sentiment_df['시간대'].max(), support_df['시간'].max())

sentiment_df = sentiment_df[(sentiment_df['시간대'] >= start) & (sentiment_df['시간대'] <= end)]
support_df = support_df[(support_df['시간'] >= start) & (support_df['시간'] <= end)]
support_df = support_df.groupby('시간', as_index=False).mean()
merged = pd.merge(sentiment_df, support_df, left_on='시간대', right_on='시간').drop(columns='시간')

#피처 선택 및 정규화
# 긍정률(이동평균 포함)과 지지율을 합쳐서 학습 데이터로 구성하고 정규화 진행
features = ['긍정률_이동평균', '긍정률']
target = '지지율'

data = merged[features + [target]]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(data)

#시계열 입력 구성
SEQ_LEN = 24 # 과거 24시간 입력
PRED_LEN = 144 # 미래 144시간 예측

X, y = [], []
for i in range(len(scaled) - SEQ_LEN - PRED_LEN + 1):
    X.append(scaled[i:i+SEQ_LEN, :-1])
    y.append(scaled[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN, -1])
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

#모델 정의
class HybridForecastNet(nn.Module):
    def __init__(self, input_dim=2, conv_dim=64, attn_heads=4, hidden_dim=128, output_len=144):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1)# 1D CNN 적용
        self.attn = nn.MultiheadAttention(embed_dim=conv_dim, num_heads=attn_heads, batch_first=True)  # Self-Attention
        self.norm = nn.LayerNorm(conv_dim)# Layer Normalization
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim * SEQ_LEN, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len)
        )

    def forward(self, x):
        x = x.transpose(1, 2)# (B, C, T)로 변환하여 Conv1D 입력 형태로 변환
        x = self.conv(x).transpose(1, 2)# Conv1D 수행 후 다시 (B, T, C) 형태로 변환
        attn_out, _ = self.attn(x, x, x)# Self-Attention 적용
        x = self.norm(attn_out + x) # Residual + LayerNorm 적용
        return self.mlp(x)  # MLP를 통해 예측값 출력

#모델 학습
model = HybridForecastNet()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


for epoch in range(300):
    model.train()
    total_loss = 0
    for i in range(len(X)):
        xb = X[i:i+1]
        yb = y[i:i+1]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(X):.4f}")

#모델 저장 및 스케일러 저장
torch.save(model.state_dict(), "hybrid_forecast_model_LEEJ.pt")
joblib.dump(scaler, "scaler_LEEJ.pkl")
print("\u2705 모델 저장 완료: hybrid_forecast_model_LEEJ.pt")
print("\u2705 스케일러 저장 완료: scaler_LEEJ.pkl")

#예측 및 평가
model.eval()
with torch.no_grad():
    pred_scaled = model(X[-1:]).cpu().numpy().flatten()
    true_scaled = y[-1].cpu().numpy()

    dummy = np.zeros((PRED_LEN, len(features)))
    pred_full = scaler.inverse_transform(np.concatenate([dummy, pred_scaled.reshape(-1, 1)], axis=1))[:, -1]
    true_full = scaler.inverse_transform(np.concatenate([dummy, true_scaled.reshape(-1, 1)], axis=1))[:, -1]

# 평가 지표 출력
r2 = r2_score(true_full, pred_full)
mse = mean_squared_error(true_full, pred_full)
rmse = np.sqrt(mse)
print(f"R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

#시각화
plt.figure(figsize=(14, 5))
plt.plot(true_full, label="실제 지지율", linewidth=2)
plt.plot(pred_full, label="예측 지지율", linestyle='--')
plt.title("이준석 후보 지지율 예측 결과")
plt.xlabel("시간(Hour)")
plt.ylabel("지지율(%)")
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("이준석_지지율예측결과.png")
plt.show()
