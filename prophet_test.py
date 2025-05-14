import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib import rc
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

#  파일 불러오기 (너 파일 경로로 바꿔!)
df = pd.read_csv('C:\\Users\\wjsgy\\PycharmProjects\\PythonProject1\\candidate_sentiment_realistic_final_v2.csv')

#  후보별 예측 + 하나의 그래프에 출력 + 정확도 평가
plt.figure(figsize=(12, 6))
candidates = df['candidate'].unique()
colors = {'이재명': 'blue', '김문수': 'red'}

for candidate in candidates:
    print(f"\n✅ {candidate} 후보 예측 + 평가 시작")
    df_cand = df[df['candidate'] == candidate].copy()
    df_cand.rename(columns={'comment_day': 'ds', 'hits': 'y'}, inplace=True)
    df_cand['ds'] = pd.to_datetime(df_cand['ds'])
    df_cand['cap'] = 100
    df_cand['floor'] = 0

    #  train/test split (최근 7일 test)
    split_date = df_cand['ds'].max() - pd.Timedelta(days=7)
    train = df_cand[df_cand['ds'] <= split_date]
    test = df_cand[df_cand['ds'] > split_date]

    model = Prophet(
        growth='logistic',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(train)

    #  test 기간까지 예측
    future = model.make_future_dataframe(periods=7)  # test set 크기만큼 예측
    #  최대값 100, 최소값 0
    future['cap'] = 100
    future['floor'] = 0
    forecast = model.predict(future)

    #  정확도 평가
    pred = forecast[['ds', 'yhat']].set_index('ds').loc[test['ds']]
    true = test.set_index('ds')['y']

    mae = mean_absolute_error(true, pred['yhat'])
    rmse = np.sqrt(mean_squared_error(true, pred['yhat']))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    #  전체 예측 (향후 30일 예측도 추가)
    future_full = model.make_future_dataframe(periods=30)
    future_full['cap'] = 100
    future_full['floor'] = 0
    forecast_full = model.predict(future_full)

    #  시각화: 예측 라인 + 실제 점
    plt.plot(forecast_full['ds'], forecast_full['yhat'],
             label=f"{candidate} 예측", color=colors[candidate])
    plt.scatter(df_cand['ds'], df_cand['y'],
                color=colors[candidate], s=10, alpha=0.6,
                label=f"{candidate} 실제")

plt.title("응기잇응깃: 후보별 긍정률 예측 + 실제 데이터 + 정확도 평가")
plt.xlabel("날짜")
plt.ylabel("긍정률 (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
