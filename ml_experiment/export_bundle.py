import os
import joblib
from datetime import datetime, timezone

# 1. 5번 이슈에서 학습한 모델 로드
model_path = "models/best_isolation_forest.joblib"
if not os.path.exists(model_path):
    model_path = "ml_experiment/models/best_isolation_forest.joblib"

best_model = joblib.load(model_path)

# 2. 이번에 확정된 최적 Feature 6개
feature_columns = list(best_model.feature_names_in_)

# 3. FastAPI bundle 규격 생성
bundle = {
    "model": best_model,
    "model_version": "v2.0.0-isolation-forest",
    "feature_columns": feature_columns,
    "anomaly_threshold": 0.5, # decision_function 기반 정규화 점수 기준 (필요시 조정)
    "trained_at": datetime.now(timezone.utc).isoformat(),
}

# 4. FastAPI가 바라보는 기본 위치(프로젝트 루트/models/behavior_model.joblib)에 저장
target_dir = "../models" if os.path.exists("../models") or os.path.exists("../app") else "models"
os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, "behavior_model.joblib")

joblib.dump(bundle, target_path)
print(f"✅ FastAPI 서빙용 bundle 저장 완료: {target_path}")
print(f"- 피처 목록: {bundle['feature_columns']}")
print(f"- 모델 버전: {bundle['model_version']}")