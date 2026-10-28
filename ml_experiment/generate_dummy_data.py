import numpy as np
import pandas as pd
import os

def generate_fairticket_dataset(n_users=1000, seed=42):
    np.random.seed(seed)
    data = []
    
    # 7대 시나리오 확률 가중치 (정상 계열 80%, 이상 계열 20%)
    scenarios = [
        "normal", "fast_normal", "shared_ip", "frequent_cancel",
        "macro", "hold_cancel", "multi_account"
    ]
    probabilities = [0.50, 0.15, 0.10, 0.05, 0.08, 0.06, 0.06]
    
    for i in range(n_users):
        user_id = f"user_{i:04d}"
        scenario = np.random.choice(scenarios, p=probabilities)
        is_anomaly = 1 if scenario in ["macro", "hold_cancel", "multi_account"] else 0
        
        # 1. 일반 정상 사용자 (로그정규분포 평균 60초 기준)
        if scenario == "normal":
            avg_interval = np.random.lognormal(mean=4.1, sigma=0.5)
            interval_var = np.random.uniform(5.0, 50.0)
            seat_latency = np.random.exponential(scale=30.0) + 1.0
            cancel_rate = np.random.beta(a=1, b=9)
            card_count = 1
            ip_count = np.random.randint(1, 3)
            booking_count = np.random.poisson(lam=3) + 1
            
        # 2. 빠른 정상 사용자 (손이 빠른 일반인)
        elif scenario == "fast_normal":
            avg_interval = np.random.uniform(15.0, 40.0)
            interval_var = np.random.uniform(3.0, 20.0)
            seat_latency = np.random.uniform(1.2, 3.0) # 1초 이상 정상 속도
            cancel_rate = np.random.beta(a=1, b=8)
            card_count = 1
            ip_count = 1
            booking_count = np.random.poisson(lam=5) + 1
            
        # 3. 공용 IP 정상 사용자 (PC방/기숙사 망)
        elif scenario == "shared_ip":
            avg_interval = np.random.lognormal(mean=3.8, sigma=0.5)
            interval_var = np.random.uniform(4.0, 30.0)
            seat_latency = np.random.exponential(scale=25.0) + 1.0
            cancel_rate = np.random.beta(a=1, b=9)
            card_count = 1
            ip_count = np.random.randint(4, 7) # IP 반복수 높음
            booking_count = np.random.poisson(lam=4) + 1
            
        # 4. 반복 취소 사용자 (단순 사유 취소 빈번)
        elif scenario == "frequent_cancel":
            avg_interval = np.random.lognormal(mean=4.0, sigma=0.5)
            interval_var = np.random.uniform(5.0, 40.0)
            seat_latency = np.random.exponential(scale=20.0) + 1.0
            cancel_rate = np.random.uniform(0.5, 0.8) # 취소율만 높음
            card_count = 1
            ip_count = 1
            booking_count = np.random.poisson(lam=4) + 1
            
        # 5. 매크로 의심 사용자 (정밀 타이머 반복 호출)
        elif scenario == "macro":
            avg_interval = np.random.exponential(scale=0.2) + 0.1 # 5초 이하 초단기
            interval_var = np.random.uniform(0.01, 0.5) # 분산 1.0 미만
            seat_latency = np.random.uniform(0.1, 0.8) # 1초 미만 비인간적 속도
            cancel_rate = np.random.uniform(0.1, 0.4)
            card_count = np.random.randint(1, 3)
            ip_count = np.random.randint(1, 4)
            booking_count = np.random.randint(15, 50)
            
        # 6. 반복 선점 의심 사용자 (미입금 취소 반복)
        elif scenario == "hold_cancel":
            avg_interval = np.random.uniform(2.0, 8.0)
            interval_var = np.random.uniform(0.5, 2.0)
            seat_latency = np.random.uniform(0.3, 1.2)
            cancel_rate = np.random.uniform(0.85, 1.0) # 취소율 극단적 높음
            card_count = 1
            ip_count = np.random.randint(1, 3)
            booking_count = np.random.randint(10, 30)
            
        # 7. 다계정 의심 사용자 (동일 카드/IP 공유 집중 결제)
        elif scenario == "multi_account":
            avg_interval = np.random.uniform(3.0, 10.0)
            interval_var = np.random.uniform(1.0, 5.0)
            seat_latency = np.random.uniform(0.5, 2.0)
            cancel_rate = np.random.uniform(0.3, 0.7)
            card_count = np.random.randint(4, 8) # 동일 카드 4건 이상 공유
            ip_count = np.random.randint(5, 12) # 동일 IP 5건 이상 집중
            booking_count = np.random.randint(10, 25)

        data.append({
            "user_id": user_id,
            "scenario": scenario,
            "is_anomaly": is_anomaly,
            "avg_booking_interval": round(max(0.1, avg_interval), 2),
            "booking_interval_variance": round(max(0.01, interval_var), 4),
            "seat_selection_latency": round(max(0.05, seat_latency), 2),
            "cancel_rate": round(min(1.0, max(0.0, cancel_rate)), 4),
            "card_reuse_count": int(card_count),
            "ip_booking_density": int(ip_count),
            "booking_count": int(booking_count)
        })

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    output_path = "data/fairticket_synthetic_features.csv"
    df.to_csv(output_path, index=False)
    
    print("=== 합성 데이터셋 생성 완료 ===")
    print(f"저장 경로: {output_path}")
    print(f"총 데이터 수: {len(df)}건")
    print(f"- 정상 사용자(0): {(df['is_anomaly'] == 0).sum()}건 ({(df['is_anomaly'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"- 이상 사용자(1): {(df['is_anomaly'] == 1).sum()}건 ({(df['is_anomaly'] == 1).sum() / len(df) * 100:.1f}%)")
    print("\n시나리오별 분포:")
    print(df["scenario"].value_counts())

if __name__ == "__main__":
    generate_fairticket_dataset()