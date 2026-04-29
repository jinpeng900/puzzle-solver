#!/usr/bin/env python3
"""
分关卡独立优化 L1 和 L2。
固定 L3-L5 为当前 GA 最优，只优化 L1/L2。
"""
import subprocess, random, os, sys, multiprocessing

SEED = 114514
WORKERS = 8
EXE = "./allstar"
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

# Original L1/L2
ORIG = {
    1: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,0.5,8,4,55,6,0.3,0.0,0.90,0.97,10000,6],
    2: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,1.0,8,4,25,6,0.3,0.0,0.90,0.97,10000,6],
}

# Read GA best for L3-L5
with open('best_seed_114514.txt') as f:
    ga_lines = f.readlines()
ga_params = {}
for line in ga_lines:
    line = line.strip()
    if not line: continue
    parts = line.split()
    if len(parts) == 2:
        ga_params[parts[0]] = float(parts[1]) if '.' in parts[1] else int(parts[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def round_step(v, step, typ):
    if typ == int:
        return int((int(round(v / step)) * step))
    else:
        prec = max(0, len(str(step).split('.')[-1])) if '.' in str(step) else 0
        return round(round(v / step) * step, prec)

def make_l1_chrom(base, noise=0.15):
    chrom = []
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

def write_params(l1_vals, l2_vals, fpath):
    names = [p[0] for p in PARAMS]
    with open(fpath, 'w') as f:
        for n, v in zip(names, l1_vals):
            f.write(f"L1_{n} {v}\n")
        for n, v in zip(names, l2_vals):
            f.write(f"L2_{n} {v}\n")
        for lv in [3,4,5]:
            for name in names:
                key = f"L{lv}_{name}"
                f.write(f"{key} {ga_params.get(key, 0)}\n")

def evaluate(args):
    l1_vals, l2_vals, idx = args
    fpath = f"/tmp/l1l2_{idx}.txt"
    write_params(l1_vals, l2_vals, fpath)
    try:
        res = subprocess.run([EXE, str(SEED), fpath], capture_output=True, text=True, timeout=120, cwd=CWD)
        lines = res.stdout.splitlines()
        scores = []
        for line in lines:
            if line.startswith("Result:"):
                scores.append(int(line.split()[1]))
        if len(scores) >= 5:
            total = sum(scores)
            return total, scores[0], scores[1], scores[2], scores[3], scores[4], l1_vals, l2_vals
    except Exception as e:
        print(f"Error: {e}")
    return 0, 0, 0, 0, 0, 0, l1_vals, l2_vals

def main():
    print("=== Stage 1: Optimize L1 (fix L2=original, L3-L5=GA best) ===")
    pool = multiprocessing.Pool(WORKERS)
    
    # Test original L1, GA L1, and variations
    tests = []
    tests.append(("orig_L1", ORIG[1][:], ORIG[2][:]))
    tests.append(("ga_L1", None, ORIG[2][:]))  # will fill GA L1 from ga_params
    
    # Build GA L1 from file
    ga_l1 = []
    for name, typ, lo, hi, step in PARAMS:
        key = f"L1_{name}"
        ga_l1.append(ga_params.get(key, 0))
    tests[1] = ("ga_L1", ga_l1[:], ORIG[2][:])
    
    # Random variations around original L1
    for i in range(20):
        tests.append((f"var_L1_{i}", make_l1_chrom(ORIG[1], noise=0.15), ORIG[2][:]))
    
    args = []
    for tag, l1, l2 in tests:
        args.append((l1, l2, len(args)))
    
    results = pool.map(evaluate, args)
    
    print("\nL1 Optimization Results:")
    best_total = -1
    best_l1 = None
    for total, s1, s2, s3, s4, s5, l1, l2 in results:
        print(f"  L1={s1} L2={s2} L3={s3} L4={s4} L5={s5} TOTAL={total}")
        if total > best_total:
            best_total = total
            best_l1 = l1[:]
    
    print(f"\nBest L1 found: TOTAL={best_total}")
    
    # Stage 2: Optimize L2 with best L1
    print("\n=== Stage 2: Optimize L2 (fix best L1, L3-L5=GA best) ===")
    tests2 = []
    tests2.append(("orig_L2", best_l1[:], ORIG[2][:]))
    
    # Build GA L2
    ga_l2 = []
    for name, typ, lo, hi, step in PARAMS:
        key = f"L2_{name}"
        ga_l2.append(ga_params.get(key, 0))
    tests2.append(("ga_L2", best_l1[:], ga_l2[:]))
    
    for i in range(20):
        tests2.append((f"var_L2_{i}", best_l1[:], make_l1_chrom(ORIG[2], noise=0.15)))
    
    args2 = []
    for tag, l1, l2 in tests2:
        args2.append((l1, l2, len(args2)))
    
    results2 = pool.map(evaluate, args2)
    
    print("\nL2 Optimization Results:")
    best_total2 = -1
    best_l2 = None
    for total, s1, s2, s3, s4, s5, l1, l2 in results2:
        print(f"  L1={s1} L2={s2} L3={s3} L4={s4} L5={s5} TOTAL={total}")
        if total > best_total2:
            best_total2 = total
            best_l2 = l2[:]
    
    print(f"\nBest L2 found: TOTAL={best_total2}")
    
    # Save best combined params
    write_params(best_l1, best_l2, "best_l1l2_combined.txt")
    print("Saved to best_l1l2_combined.txt")
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
