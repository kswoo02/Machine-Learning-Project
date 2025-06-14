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

#파라미터
SEQ_LEN = 24
PRED_LEN = 144
FEATURES = ['긍정률_이동평균', '긍정률']

# ──────── 모델 정의 ────────
class HybridForecastNet(nn.Module):
    def __init__(self, input_dim=2, conv_dim=64, attn_heads=4, hidden_dim=128, output_len=144):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=conv_dim, num_heads=attn_heads, batch_first=True)
        self.norm = nn.LayerNorm(conv_dim)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim * SEQ_LEN, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        return self.mlp(x)

#데이터 불러오기
df = pd.read_csv('merged_이준석_긍정률_5월13일~6월1일.csv')
df['시간대'] = pd.to_datetime(df['시간대'])
df = df.sort_values('시간대').reset_index(drop=True)

#Scaler 불러오기
scaler: MinMaxScaler = joblib.load('scaler_LEEJ.pkl')

# 마지막 144시간 타겟 시점 추출
target_times = df['시간대'].iloc[-PRED_LEN:].reset_index(drop=True)

#규화 및 입력 시퀀스 구성
temp = df[FEATURES].copy()
temp['지지율'] = 0  # 스케일러 구조 맞추기용 더미
scaled = scaler.transform(temp)

X_list = []
for i in range(len(scaled) - PRED_LEN, len(scaled)):
    X_seq = scaled[i - SEQ_LEN:i, :-1]  # (24, 2)
    X_list.append(X_seq)

X_input = torch.tensor(np.array(X_list), dtype=torch.float32)  # (144, 24, 2)

#모델 불러오기 및 예측
model = HybridForecastNet(input_dim=2)
model.load_state_dict(torch.load("hybrid_forecast_model_LEEJ.pt", map_location='cpu'))
model.eval()

with torch.no_grad():
    preds_scaled = model(X_input).cpu().numpy()  # (144, 144)

#예측값 복원 (지지율만 역정규화)
preds_full = []
for i in range(PRED_LEN):
    dummy = np.zeros((PRED_LEN, len(FEATURES)))  # (144, 2)
    combined = np.concatenate([dummy, preds_scaled[i].reshape(-1, 1)], axis=1)
    restored = scaler.inverse_transform(combined)[:, -1]
    preds_full.append(restored[i])  # 중앙 예측값만 추출

# 시각화 및 저장
plt.figure(figsize=(14, 5))
plt.plot(target_times, preds_full, label="예측 지지율", linestyle='--', color='tab:blue')
plt.title("이준석 후보: 마지막 144시간 지지율 예측")
plt.xlabel("시간")
plt.ylabel("지지율 (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("이준석_마지막144_지지율예측.png")
plt.show()

#CSV 저장
result_df = pd.DataFrame({
    '시간대': target_times,
    '예측지지율': preds_full
})
result_df.to_csv("이준석_마지막144_지지율예측.csv", index=False, encoding='utf-8-sig')
print("CSV 저장 완료: 이준석_마지막144_지지율예측.csv")
