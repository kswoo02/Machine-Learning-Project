import pandas as pd
import glob
import os

# ✅ 1. 분석 대상 CSV 파일 경로 설정
base_path = r"C:\Users\user\PycharmProjects\Machine-Learning\이재명"
csv_files = glob.glob(os.path.join(base_path, "*.csv"))

merged_list = []

for file in csv_files:
    df = pd.read_csv(file)

    # ✅ 2. 후보자 이름 추출 (파일명 기준)
    candidate = os.path.basename(file).split("_")[0]
    df["후보"] = candidate

    # ✅ 3. '댓글 작성일' 컬럼이 없으면 건너뜀
    if "댓글 작성일" not in df.columns:
        print(f"⚠️ 파일 제외: {file} - '댓글 작성일' 없음")
        continue

    # ✅ 4. 날짜 형식 변환 (오전/오후 포함)
    df["댓글 작성일"] = df["댓글 작성일"].astype(str).str.replace("오전", "AM").str.replace("오후", "PM")
    df["댓글 작성일"] = pd.to_datetime(df["댓글 작성일"], errors="coerce")

    merged_list.append(df)

# ✅ 5. 통합
merged_df = pd.concat(merged_list, ignore_index=True)

# ✅ 6. 댓글 작성일 기준 정렬
merged_df = merged_df.dropna(subset=["댓글 작성일"])
merged_df = merged_df.sort_values(by="댓글 작성일")

# ✅ 7. 결과 저장
output_path = os.path.join(base_path, "통합_후보별_댓글_정렬.csv")
merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ✅ 8. 확인 메시지
print("✅ CSV 통합 및 날짜 정렬 완료")
print(f"총 댓글 수: {len(merged_df)}")
print(f"저장된 파일: {output_path}")
