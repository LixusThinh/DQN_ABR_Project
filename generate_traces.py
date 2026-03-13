# generate_traces.py — version 2, thêm burst drop
import numpy as np
import os

def generate_trace(duration=320, mean_bw=2.0, std_bw=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    timestamps = []
    bandwidths = []
    t = 0.0
    bw = mean_bw  # bandwidth hiện tại

    while t < duration:
        # 10% xác suất xảy ra "burst drop" — mạng tụt đột ngột
        if np.random.random() < 0.10:
            bw = bw * np.random.uniform(0.1, 0.3)
            # Tụt còn 10-30% so với hiện tại — simulate mạng yếu đột ngột
        else:
            # Bình thường: dao động log-normal quanh mean
            bw = np.random.lognormal(
                mean=np.log(max(mean_bw, 0.1)),
                sigma=std_bw * 0.3
            )

        bw = max(bw, 0.1)
        bw = min(bw, 20.0)

        timestamps.append(round(t, 1))
        bandwidths.append(round(bw, 4))
        t += 0.5

    return timestamps, bandwidths


def save_trace(filepath, timestamps, bandwidths):
    with open(filepath, 'w') as f:
        for t, bw in zip(timestamps, bandwidths):
            f.write(f"{t}\t{bw}\n")


if __name__ == "__main__":
    train_dir = "traces/train"
    test_dir  = "traces/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    configs = [
        {"mean_bw": 0.8, "std_bw": 0.8,  "label": "weak"},
        # mean thấp, std cao → mạng yếu và chập chờn
        {"mean_bw": 2.5, "std_bw": 1.2,  "label": "medium"},
        # mean trung bình, std vừa
        {"mean_bw": 5.0, "std_bw": 1.5,  "label": "strong"},
        # mean cao nhưng vẫn có burst drop
    ]

    count = 0
    for cfg in configs:
        for i in range(30):
            ts, bw = generate_trace(
                duration=320,
                mean_bw=cfg["mean_bw"],
                std_bw=cfg["std_bw"],
                seed=count
            )
            save_trace(f"{train_dir}/trace_{cfg['label']}_{i}.txt", ts, bw)
            count += 1

    print(f"Generated {count} training traces")

    np.random.seed(999)
    for i in range(20):
        mean = np.random.uniform(0.5, 6.0)
        std  = np.random.uniform(0.5, 2.0)
        ts, bw = generate_trace(duration=320, mean_bw=mean, std_bw=std, seed=1000+i)
        save_trace(f"{test_dir}/trace_test_{i}.txt", ts, bw)

    print("Generated 20 testing traces")
    print("Done!")