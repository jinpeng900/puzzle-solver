#!/usr/bin/env python3
"""
全局 Hill Climbing 微调：以当前 best_all.txt 为起点，随机扰动少量参数，追求总分最高。
"""
import subprocess, random, os, sys, re, time, multiprocessing

EXE = "./o_local"
CWD = os.path.dirname(os.path.abspath(__file__))
WORKERS = max(1, multiprocessing.cpu_count() // 2)

# 参数定义: (name, type, low, high, step)
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

def load_chrom(path):
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 2:
                vals[parts[0]] = float(parts[1]) if '.' in parts[1] else int(parts[1])
    chrom = []
    for lv in range(1,6):
        for name, typ, lo, hi, step in PARAMS:
            key = f"L{lv}_{name}"
            chrom.append(vals.get(key, 0))
    return chrom

def write_chrom(chrom, path):
    with open(path, 'w') as f:
        idx = 0
        for lv in range(1,6):
            for name, typ, lo, hi, step in PARAMS:
                f.write(f"L{lv}_{name} {chrom[idx]}\n")
                idx += 1

def evaluate(path):
    try:
        res = subprocess.run([EXE, path], capture_output=True, text=True, timeout=180, cwd=CWD)
        for line in res.stdout.splitlines():
            if "TOTAL SCORE:" in line:
                return int(line.split(":")[-1].strip())
    except Exception as e:
        print("eval error:", e)
    return 0

def mutate(chrom, n_changes=3, noise=0.15):
    newc = chrom[:]
    indices = random.sample(range(len(chrom)), n_changes)
    for i in indices:
        lv = i // len(PARAMS)
        p_idx = i % len(PARAMS)
        name, typ, lo, hi, step = PARAMS[p_idx]
        span = (hi - lo) * noise
        if typ == int:
            delta = int(round(random.uniform(-span, span)))
            newc[i] = max(lo, min(hi, newc[i] + delta))
            newc[i] = (newc[i] // step) * step
        else:
            delta = random.uniform(-span, span)
            newc[i] = max(lo, min(hi, newc[i] + delta))
            newc[i] = round(round(newc[i] / step) * step, max(0, len(str(step).split('.')[-1])))
    return newc

def main():
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    start_file = "best_all.txt"
    best_chrom = load_chrom(start_file)
    best_score = evaluate(start_file)
    print(f"Start score: {best_score}, max_iter={max_iter}")

    write_chrom(best_chrom, "hill_best.txt")
    it = 0
    no_improve = 0

    while it < max_iter:
        it += 1
        n_changes = random.choice([1, 2, 3])
        noise = random.choice([0.08, 0.12, 0.18])
        cand = mutate(best_chrom, n_changes, noise)
        fpath = f"/tmp/hc_{it}.txt"
        write_chrom(cand, fpath)
        score = evaluate(fpath)

        if score > best_score:
            best_score = score
            best_chrom = cand[:]
            write_chrom(best_chrom, "hill_best.txt")
            no_improve = 0
            print(f"Iter {it}: NEW BEST = {best_score} (+{score-best_score})")
            if best_score >= 31000:
                print("TARGET 31000 REACHED!")
                break
        else:
            no_improve += 1
            if it % 20 == 0:
                print(f"Iter {it}: current={score} best={best_score} no_improve={no_improve}")

        # 如果长时间没有改进，增大变异幅度
        if no_improve >= 50:
            print("Stuck, doing large mutation...")
            best_chrom = mutate(best_chrom, 5, 0.30)
            no_improve = 0

if __name__ == "__main__":
    main()
