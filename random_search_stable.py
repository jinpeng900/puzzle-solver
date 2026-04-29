#!/usr/bin/env python3
"""
快速随机采样搜索：在原始参数附近小范围扰动，找最稳定的参数。
目标: max(avg - 0.5 * std)
"""
import subprocess, random, os, sys, math, statistics, multiprocessing, time

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
WORKERS = max(1, multiprocessing.cpu_count() // 2)
EXE = "./eval_seed"
CWD = os.path.dirname(os.path.abspath(__file__))

PARAMS = [
    ("dfs_limit", int, 300000, 1800000, 50000),
    ("keep1", int, 60, 350, 10),
    ("keep2", int, 20, 180, 5),
    ("keep3", int, 3, 35, 1),
    ("cache_limit", int, 20000, 100000, 5000),
    ("mxc_w", float, 8.0, 35.0, 0.5),
    ("dq_w", float, 0.3, 1.5, 0.05),
    ("bh_w", float, 0.0, 0.08, 0.005),
    ("s3_dq_w", float, 0.3, 1.2, 0.05),
    ("s3_bh_w", float, 0.0, 0.08, 0.005),
    ("surv_mul", float, 0.2, 2.5, 0.1),
    ("beam_w", int, 0, 10, 1),
    ("beam_d", int, 0, 6, 1),
    ("short_pen", int, 0, 120, 5),
    ("surv_check", int, 3, 10, 1),
    ("beam_bonus", float, 0.0, 0.8, 0.05),
    ("len_bonus", float, 0.0, 0.5, 0.05),
    ("tfast_ratio", float, 0.70, 0.95, 0.01),
    ("tdead_ratio", float, 0.85, 0.99, 0.01),
    ("max_time_ms", int, 5000, 15000, 500),
    ("fallback_len_thresh", int, 3, 9, 1),
]

DEFAULTS_RAW = {
    1: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,0.5,8,4,55,6,0.3,0.0,0.90,0.97,10000,6],
    2: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,1.0,8,4,25,6,0.3,0.0,0.90,0.97,10000,6],
    3: [ 700000,120, 60,12,60000,18.0,0.9,0.02,0.60,0.025,1.0,0,0,55,6,0.3,0.0,0.90,0.97,10000,6],
    4: [ 600000,120, 60,12,80000,15.0,0.8,0.02,0.55,0.025,1.0,5,3,55,6,0.3,0.0,0.90,0.97,10000,6],
    5: [ 800000, 60, 30, 6,50000,15.0,0.9,0.02,0.60,0.025,1.0,0,0,55,6,0.3,0.0,0.90,0.97,10000,6],
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def round_step(v, step, typ):
    if typ == int:
        return int((int(round(v / step)) * step))
    else:
        prec = max(0, len(str(step).split('.')[-1])) if '.' in str(step) else 0
        return round(round(v / step) * step, prec)

def make_chrom(noise=0.10):
    chrom = []
    for lv in [1,2,3,4,5]:
        base = DEFAULTS_RAW[lv]
        for i, (name, typ, lo, hi, step) in enumerate(PARAMS):
            b = base[i]
            span = (hi - lo) * noise
            if typ == int:
                v = int(round(b + random.uniform(-span, span)))
            else:
                v = b + random.uniform(-span, span)
            v = clamp(v, lo, hi)
            v = round_step(v, step, typ)
            chrom.append(v)
    return chrom

def chrom_to_file(chrom, fpath):
    with open(fpath, 'w') as f:
        idx = 0
        for lv in [1,2,3,4,5]:
            for name, typ, lo, hi, step in PARAMS:
                f.write(f"L{lv}_{name} {chrom[idx]}\n")
                idx += 1

def evaluate_seed(seed, fpath):
    try:
        res = subprocess.run([EXE, str(seed), fpath], capture_output=True, text=True, timeout=120, cwd=CWD)
        return int(res.stdout.strip())
    except Exception as e:
        return 0

def evaluate_one(args):
    chrom, idx = args
    fpath = f"/tmp/rs_p{idx}.txt"
    chrom_to_file(chrom, fpath)
    seeds = [random.randint(1, 100000) for _ in range(N_SEEDS)]
    scores = []
    for seed in seeds:
        s = evaluate_seed(seed, fpath)
        if s == 0:
            return -1e9, chrom, []
        scores.append(s)
    avg = sum(scores) / len(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0
    fitness = avg - 0.5 * std
    return fitness, chrom, scores

def main():
    print(f"[Random Search] samples={N_SAMPLES} seeds={N_SEEDS} workers={WORKERS}")
    pool = multiprocessing.Pool(WORKERS)
    
    pop = [make_chrom(noise=0.10) for _ in range(N_SAMPLES)]
    # Also include original params as one sample
    orig = []
    for lv in [1,2,3,4,5]:
        orig.extend(DEFAULTS_RAW[lv])
    pop[0] = orig
    
    args = [(pop[i], i) for i in range(len(pop))]
    results = pool.map(evaluate_one, args)
    pool.close()
    pool.join()
    
    # Sort by fitness
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n=== Top 5 Results ===")
    for i, (fit, chrom, scores) in enumerate(results[:5]):
        avg = sum(scores)/len(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0
        print(f"#{i+1} fitness={fit:.1f} avg={avg:.1f} std={std:.1f} scores={scores}")
    
    best_fit, best_chrom, best_scores = results[0]
    chrom_to_file(best_chrom, "best_stable.txt")
    print(f"\nBest saved to best_stable.txt")
    
    # Verify on seed 42
    chrom_to_file(best_chrom, "/tmp/rs_verify.txt")
    s42 = evaluate_seed(42, "/tmp/rs_verify.txt")
    print(f"Seed 42 score: {s42}")

if __name__ == "__main__":
    main()
