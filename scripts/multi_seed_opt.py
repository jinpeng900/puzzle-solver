#!/usr/bin/env python3
"""
multi_seed_opt.py
多种子参数优化器（爬山 + 随机采样）
针对 2.cpp 求解器，跨种子找到广泛适应的参数。

用法: py scripts/multi_seed_opt.py [--seeds 10] [--iter 100] [--start-seed 1]

依赖: test_2.exe 已在本地编译好
"""
import subprocess
import random
import sys
import os
import time

CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_EXE = os.path.join(CWD, "test_2.exe")

# ============================================================
# 可调参数定义
# ============================================================
PARAMS = [
    ("dfs_limit",  int,   300000, 1800000, 50000),
    ("keep1",      int,   60,     350,     10),
    ("keep2",      int,   20,     180,     5),
    ("keep3",      int,   3,      35,      1),
    ("cache_limit",int,   20000,  100000,  5000),
    ("mxc_w",      float, 8.0,    35.0,    0.5),
    ("dq_w",       float, 0.3,    1.5,     0.05),
    ("bh_w",       float, 0.0,    0.08,    0.005),
    ("s3_dq_w",    float, 0.3,    1.2,     0.05),
    ("s3_bh_w",    float, 0.0,    0.08,    0.005),
    ("surv_mul",   float, 0.2,    2.5,     0.1),
    ("beam_w",     int,   0,      10,      1),
    ("beam_d",     int,   0,      6,       1),
    ("short_pen",  int,   0,      120,     5),
    ("surv_check", int,   3,      10,      1),
    ("beam_bonus", float, 0.0,    0.8,     0.05),
    ("len_bonus",  float, 0.0,    0.5,     0.05),
    ("tfast",      float, 0.70,   0.95,    0.01),
    ("tdead",      float, 0.85,   0.99,    0.01),
    ("max_time",   int,   5000,   15000,   500),
    ("fallback_len",int,  3,      9,       1),
]

# Baseline defaults (from 1.cpp hardcoded params, adapted for shared_ptr Board)
BASELINE = {
    1: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,0.5,8,4,55,6,0.3,0.0,0.90,0.97,10000,6],
    2: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,1.0,8,4,25,6,0.3,0.0,0.90,0.97,10000,6],
    3: [ 700000,120, 60,12,60000,18.0,0.9,0.02,0.60,0.025,1.0,2,2,55,6,0.3,0.0,0.90,0.97,10000,6],  # beam=2
    4: [ 600000,120, 60,12,80000,15.0,0.8,0.02,0.55,0.025,1.0,5,3,55,6,0.3,0.0,0.90,0.97,10000,6],
    5: [1200000, 60, 30, 6,50000,15.0,0.9,0.02,0.60,0.025,1.0,2,2,55,6,0.3,0.0,0.90,0.97,10000,6],  # dfs 1.2M, beam=2
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def round_step(v, step, typ):
    if typ == int:
        return int(round(v / step) * step)
    else:
        s = str(step)
        prec = max(0, len(s.split('.')[-1])) if '.' in s else 0
        return round(round(v / step) * step, prec)

def make_chromosome(base=None, noise=0.15):
    """Generate a random chromosome around baseline."""
    chrom = [[], [], [], [], []]
    for lv in range(5):
        for i, (name, typ, lo, hi, step) in enumerate(PARAMS):
            if base is not None:
                b = base[lv][i]
                span = (hi - lo) * noise
                if typ == int:
                    v = int(round(b + random.uniform(-span, span)))
                else:
                    v = b + random.uniform(-span, span)
            else:
                if typ == int:
                    v = random.randrange(lo // step, hi // step + 1) * step
                else:
                    steps = int(round((hi - lo) / step))
                    v = lo + random.randint(0, steps) * step
            v = clamp(v, lo, hi)
            v = round_step(v, step, typ)
            chrom[lv].append(v)
    return chrom

def chrom_to_file(chrom, fpath):
    """Write chromosome to parameter file (best_all.txt format)."""
    names = [p[0] for p in PARAMS]
    with open(fpath, 'w') as f:
        for lv in range(5):
            for n, v in zip(names, chrom[lv]):
                f.write(f"L{lv+1}_{n} {v}\n")

def evaluate_chrom(chrom, seeds, start_seed):
    """Evaluate a chromosome by running test_2.exe across multiple seeds.
    Returns average total score."""
    # Currently test_2.exe doesn't read param files; it uses hardcoded params.
    # For optimization to work, we need to either:
    # 1. Modify test_2.exe to accept params from file
    # 2. Or use a simpler optimization approach
    
    # For now, we use the baseline params hardcoded in test_2.exe
    # The optimization focuses on the baseline already tested
    
    # Placeholder: returns baseline score for now
    # TODO: integrate param file reading into test_2.exe
    return 0  # Placeholder

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--iter", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=8)
    args = ap.parse_args()

    print(f"[Multi-Seed Opt] seeds={args.seeds} iter={args.iter} start_seed={args.start_seed}")
    
    # Phase 1: Test baseline
    print("\n=== Phase 1: Testing Baseline ===")
    baseline_chrom = [BASELINE[lv][:] for lv in range(1, 6)]
    
    # Phase 2: Random sampling around baseline
    print("\n=== Phase 2: Random Sampling ===")
    best_chrom = baseline_chrom
    best_score = 0  # Will be filled
    
    # Phase 3: Hill climbing
    print("\n=== Phase 3: Hill Climbing ===")
    
    # For now, we run the baseline test and report
    total_seeds = args.seeds
    start = args.start_seed
    
    print(f"\nRun baseline with test_2.exe on {total_seeds} seeds (start={start}):")
    t0 = time.time()
    result = subprocess.run(
        [TEST_EXE, str(total_seeds), str(start)],
        capture_output=True, text=True, timeout=600, cwd=CWD
    )
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    print(f"Time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
