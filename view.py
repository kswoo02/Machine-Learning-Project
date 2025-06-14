import pandas as pd
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm

#폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

#파일 불러오기
df_approval = pd.read_csv('이재명_최종지지율.csv').rename(columns={'시간': '날짜'})
df_sentiment = pd.read_csv('편향완화_시간대별_긍정률_결과_이재명.csv').rename(columns={'시간대': '날짜'})
df_predicted = pd.read_csv('이재명 결과.csv').rename(columns={'시간대': '날짜', '지지율': '예측지지율'})

df_approval['날짜'] = pd.to_datetime(df_approval['날짜'])
df_sentiment['날짜'] = pd.to_datetime(df_sentiment['날짜'])
df_predicted['날짜'] = pd.to_datetime(df_predicted['날짜'])

#병합
merged_df = pd.merge(df_approval, df_sentiment, on='날짜', how='inner')
final_df = pd.merge(merged_df, df_predicted, on='날짜', how='outer').sort_values(by='날짜')

#최종 수치 표시 함수
def annotate_last_value(x, y, label, color):
    if len(x) > 0 and len(y) > 0:
        plt.text(x.iloc[-1], y.iloc[-1] + 2, f'{label} {y.iloc[-1]:.1f}%', color=color, fontsize=10, ha='left')

# 시각화 1: 1시간 단위
plt.figure(figsize=(14, 6))
plt.plot(final_df['날짜'], final_df['지지율'], label='지지율', linewidth=2)
plt.plot(final_df['날짜'], final_df['예측지지율'], label='예측 지지율', linewidth=2, linestyle='--')
plt.plot(final_df['날짜'], final_df['긍정률'], label='긍정률', linewidth=2, linestyle='-.')

annotate_last_value(final_df['날짜'], final_df['지지율'], '지지율', 'blue')
annotate_last_value(final_df['날짜'], final_df['예측지지율'], '예측', 'orange')
annotate_last_value(final_df['날짜'], final_df['긍정률'], '긍정률', 'green')

plt.ylim(0, 100)
plt.xlabel('날짜')
plt.ylabel('비율 (%)')
plt.title('이재명 긍정률, 지지율, 예측 지지율 변화 (1시간 단위)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('이재명_1시간단위_시각화.png', dpi=300)
plt.close()

#하루 단위 데이터 생성
daily_df = final_df.copy()
daily_df['일자'] = daily_df['날짜'].dt.date
daily_grouped = daily_df.groupby('일자').mean(numeric_only=True).reset_index()

# 시각화 2: 하루 단위
plt.figure(figsize=(14, 6))
plt.plot(daily_grouped['일자'], daily_grouped['지지율'], label='지지율 (일평균)', linewidth=2, marker='o')
plt.plot(daily_grouped['일자'], daily_grouped['예측지지율'], label='예측 지지율 (일평균)', linewidth=2, marker='s', linestyle='--')
plt.plot(daily_grouped['일자'], daily_grouped['긍정률'], label='긍정률 (일평균)', linewidth=2, marker='^', linestyle='-.')

annotate_last_value(pd.Series(daily_grouped['일자']), daily_grouped['지지율'], '지지율', 'blue')
annotate_last_value(pd.Series(daily_grouped['일자']), daily_grouped['예측지지율'], '예측', 'orange')
annotate_last_value(pd.Series(daily_grouped['일자']), daily_grouped['긍정률'], '긍정률', 'green')

plt.ylim(0, 100)
plt.xlabel('일자')
plt.ylabel('비율 (%)')
plt.title('이재명 긍정률, 지지율, 예측 지지율 변화 (하루 단위)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('이재명_하루단위_시각화.png', dpi=300)
plt.close()

print("그래프 2종 저장 완료: 이재명_1시간단위_시각화.png, 이재명_하루단위_시각화.png")
