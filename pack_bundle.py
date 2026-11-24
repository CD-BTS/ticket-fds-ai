# [이전 MVP 버전] 이 파일은 현재 실행 경로에서 쓰지 않는다.
# 현재 서빙 모델(6개 Feature)은 train_fds_model.py 가 만들고, Docker 이미지도 그 스크립트를 쓴다.
# 기록 보존을 위해 남겨 둔 파일이다. 원본 더미로 학습한 모델을 전처리 없이 묶는다. 만든 번들은 현재 서버가 받지 않는다 (스키마 버전 1).
import os
import joblib
import pandas as pd
from inference import build_bundle, save_bundle, raw_anomaly_score
from preprocessing import ScoreNormalizer

# 1. 2번 이슈에서 학습한 최적 모델과 학습 데이터 로드
model_path = "ml_experiment/models/best_isolation_forest.joblib"
data_path = "ml_experiment/data/fairticket_synthetic_features.csv"

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# 최적 Feature 6개
feature_columns = list(model.feature_names_in_)
X_train = df[feature_columns]

# 2. 민채 님의 ScoreNormalizer(percentile 방식) 피팅
train_raw_scores = raw_anomaly_score(model, X_train)
normalizer = ScoreNormalizer(method="percentile")
normalizer.fit(train_raw_scores)

# 3. 민채 님의 번들 빌더 호출
bundle = build_bundle(
    model=model,
    normalizer=normalizer,
    feature_columns=feature_columns,
    preprocess={},
    score_method="-score_samples",
    anomaly_threshold=0.5,
    model_version="v2.0.0-isolation-forest",
    min_booking_count=1,  # 최소 1건 이상이면 점수 산출
)

# 4. FastAPI 기본 모델 경로에 번들 저장
target_path = "models/behavior_model.joblib"
save_bundle(bundle, target_path)

print(f"🎉 모델 번들 패키징 및 저장 완료: {target_path}")
print(f"- 사용 Feature 목록: {bundle['feature_columns']}")