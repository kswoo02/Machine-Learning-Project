
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("news_comment_nli_result중앙_이재명샘플.csv")

# 'contradiction' 열 제거
df = df.drop(columns=['contradiction'])
\
    
# 'NLI 판단'이 contradiction이 아닌 데이터만 유지
filtered_df = df[df['NLI 판단'] != 'contradiction']

# 후보별로 분리 후 저장
candidates = filtered_df['라벨'].unique()

for candidate in candidates:
    candidate_df = filtered_df[filtered_df['라벨'] == candidate]
    file_name = f"{candidate}_entailment_neutral.csv"
    candidate_df.to_csv(file_name, index=False, encoding="utf-8-sig")
    print(f"Saved: {file_name}")
