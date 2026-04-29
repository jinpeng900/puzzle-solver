#!/usr/bin/env python3
"""
逐关 GA 调参，专为 seed=114514
用法: python3 ga_per_level.py <level> [pop] [gen]
示例: python3 ga_per_level.py 1 12 30    # 优化 L1
      python3 ga_per_level.py 5 16 50    # 优化 L5 用更大规模
"""
import subprocess, random, os, sys, math, multiprocessing, time, re

TARGET_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
POP_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 12
GENERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SEED = 114514
ELITE = 2
CX_PB = 0.75
MUT_PB = 0.30
TOURNAMENT_SIZE = 5
WORKERS = max(1, multiprocessing.cpu_count() // 2)
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

def load_baseline_chrom(path="best_seed_114514.txt"):
    """Load baseline params from file, return full 105-length chrom."""
    vals = {}
    with open(os.path.join(CWD, path)) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 2:
                v = float(parts[1])
                vals[parts[0]] = int(v) if v == int(v) and '.' not in parts[1] else v
    chrom = []
    for lv in range(1, 6):
        for name, typ, lo, hi, step in PARAMS:
            key = f"L{lv}_{name}"
            chrom.append(vals.get(key, 0))
    return chrom

def chrom_to_file(chrom, fpath):
    with open(fpath, 'w') as f:
        idx = 0
        for lv in range(1, 6):
            for name, typ, lo, hi, step in PARAMS:
                f.write(f"L{lv}_{name} {chrom[idx]}\n")
                idx += 1

def make_chrom(base_chrom, noise=0.15):
    """Create variant by mutating only target level params."""
    chrom = base_chrom[:]
    start = (TARGET_LEVEL - 1) * len(PARAMS)
    end = start + len(PARAMS)
    for i in range(start, end):
        name, typ, lo, hi, step = PARAMS[i - start]
        span = (hi - lo) * noise
        if typ == int:
            v = int(round(chrom[i] + random.uniform(-span, span)))
        else:
            v = chrom[i] + random.uniform(-span, span)
        v = max(lo, min(hi, v))
        if typ == int:
            v = int((int(round(v / step)) * step))
        else:
            prec = max(0, len(str(step).split('.')[-1])) if '.' in str(step) else 0
            v = round(round(v / step) * step, prec)
        chrom[i] = v
    return chrom

def evaluate_one(args):
    chrom, idx = args
    fpath = f"/tmp/ga_pl{TARGET_LEVEL}_p{idx}.txt"
    chrom_to_file(chrom, fpath)
    try:
        res = subprocess.run([EXE, str(SEED), fpath], capture_output=True, text=True, timeout=120, cwd=CWD)
        scores = []
        for line in res.stdout.splitlines():
            m = re.search(r'Result:\s*(-?\d+)\s*pts', line)
            if m:
                scores.append(int(m.group(1)))
        if len(scores) >= 5 and TARGET_LEVEL <= len(scores):
            return scores[TARGET_LEVEL - 1], chrom
        return 0, chrom
    except Exception as e:
        return 0, chrom

def tournament_select(pop, fitness, k=3):
    best, best_f = None, -1e9
    for _ in range(k):
        idx = random.randrange(len(pop))
        if fitness[idx] > best_f:
            best_f, best = fitness[idx], pop[idx]
    return best

def crossover(p1, p2):
    c1, c2 = p1[:], p2[:]
    start, end = (TARGET_LEVEL - 1) * len(PARAMS), TARGET_LEVEL * len(PARAMS)
    for i in range(start, end):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutate(chrom):
    start, end = (TARGET_LEVEL - 1) * len(PARAMS), TARGET_LEVEL * len(PARAMS)
    for i in range(start, end):
        if random.random() < MUT_PB:
            name, typ, lo, hi, step = PARAMS[i - start]
            if typ == int:
                delta = random.choice([-1, 1]) * step * random.randint(1, 5)
                chrom[i] = max(lo, min(hi, chrom[i] + delta))
                chrom[i] = int((int(round(chrom[i] / step)) * step))
            else:
                delta = random.choice([-1.0, 1.0]) * step * random.randint(1, 5)
                chrom[i] = max(lo, min(hi, chrom[i] + delta))
                prec = max(0, len(str(step).split('.')[-1])) if '.' in str(step) else 0
                chrom[i] = round(round(chrom[i] / step) * step, prec)
    return chrom

def main():
    print(f"[GA Per-Level L{TARGET_LEVEL}] pop={POP_SIZE} gen={GENERATIONS} seed={SEED} workers={WORKERS}")
    pool = multiprocessing.Pool(WORKERS)

    # Load baseline
    base_chrom = load_baseline_chrom()
    print(f"Loaded baseline ({len(base_chrom)} params)")

    # Create initial population
    pop = []
    for _ in range(POP_SIZE // 2):
        pop.append(make_chrom(base_chrom, noise=0.12))
    for _ in range(POP_SIZE - len(pop)):
        pop.append(make_chrom(base_chrom, noise=0.30))

    best_overall, best_chrom = -1, None
    out_best = f"best_l{TARGET_LEVEL}_114514.txt"
    out_log = f"ga_l{TARGET_LEVEL}_114514.log"

    no_improve_count = 0
    with open(out_log, 'w') as logf:
        for gen in range(GENERATIONS):
            t0 = time.time()
            args = [(pop[i], i) for i in range(len(pop))]
            results = pool.map(evaluate_one, args)
            fitness = [r[0] for r in results]

            sorted_idx = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
            pop = [pop[i][:] for i in sorted_idx]
            fitness = [fitness[i] for i in sorted_idx]

            gen_best = fitness[0]
            improved = False
            if gen_best > best_overall:
                best_overall = gen_best
                best_chrom = pop[0][:]
                chrom_to_file(best_chrom, out_best)
                improved = True

            msg = f"Gen{gen:02d} best={gen_best} overall={best_overall} time={time.time()-t0:.1f}s"
            if improved: msg += " **IMPROVED**"
            print(msg)
            logf.write(msg + "\n"); logf.flush()

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= 5 and gen < GENERATIONS - 5:
                print(f"  Random restart (stuck {no_improve_count} gens)")
                no_improve_count = 0
                new_pop = [pop[i][:] for i in range(ELITE)]
                while len(new_pop) < POP_SIZE:
                    new_pop.append(make_chrom(base_chrom, noise=0.30))
                pop = new_pop
                continue

            new_pop = [pop[i][:] for i in range(ELITE)]
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop, fitness, TOURNAMENT_SIZE)
                p2 = tournament_select(pop, fitness, TOURNAMENT_SIZE)
                if random.random() < CX_PB:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1 = mutate(c1); c2 = mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE: new_pop.append(c2)
            pop = new_pop

    pool.close(); pool.join()
    print(f"\n[GA L{TARGET_LEVEL}] FINAL BEST = {best_overall}")
    if best_chrom:
        chrom_to_file(best_chrom, out_best)
        print(f"Saved to {out_best}")

if __name__ == "__main__":
    main()
