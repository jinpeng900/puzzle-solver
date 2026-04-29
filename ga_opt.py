#!/usr/bin/env python3
"""
分关卡遗传算法参数优化器。
用法: python3 ga_opt.py <level> [pop_size] [generations]
"""
import subprocess, random, os, sys, math, json, multiprocessing, time

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
POP_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 16
GENERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ELITE = 2
CX_PB = 0.80
MUT_PB = 0.15
TOURNAMENT_SIZE = 3
WORKERS = max(1, multiprocessing.cpu_count() // 2)
EXE = "./o_local"
CWD = os.path.dirname(os.path.abspath(__file__))

# (name, type, low, high, step)
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

def load_defaults_from_file(path="best_all.txt"):
    if not os.path.exists(path):
        return DEFAULTS_RAW
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 2:
                vals[parts[0]] = float(parts[1]) if '.' in parts[1] else int(parts[1])
    defaults = {}
    for lv in range(1,6):
        defaults[lv] = []
        for i, (name, typ, lo, hi, step) in enumerate(PARAMS):
            key = f"L{lv}_{name}"
            defaults[lv].append(vals.get(key, DEFAULTS_RAW[lv][i]))
    return defaults

DEFAULTS = load_defaults_from_file()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def round_step(v, step, typ):
    if typ == int:
        return int((int(round(v / step)) * step))
    else:
        # float: round to step precision
        prec = max(0, len(str(step).split('.')[-1])) if '.' in str(step) else 0
        return round(round(v / step) * step, prec)

def make_chromosome(defaults=None, noise=0.20):
    chrom = []
    base = defaults[LEVEL] if defaults else [0]*len(PARAMS)
    for i, (name, typ, lo, hi, step) in enumerate(PARAMS):
        if defaults is not None:
            b = base[i]
            span = (hi - lo) * noise
            if typ == int:
                v = int(round(b + random.uniform(-span, span)))
            else:
                v = b + random.uniform(-span, span)
            v = clamp(v, lo, hi)
            v = round_step(v, step, typ)
        else:
            if typ == int:
                v = random.randrange(int(lo//step), int(hi//step)+1) * step
            else:
                steps = int(round((hi-lo)/step))
                v = lo + random.randint(0, steps) * step
        chrom.append(v)
    return chrom

def chrom_to_file(chrom, fpath):
    with open(fpath, 'w') as f:
        for lv in [1,2,3,4,5]:
            for j, (name, typ, lo, hi, step) in enumerate(PARAMS):
                if lv == LEVEL:
                    v = chrom[j]
                else:
                    v = DEFAULTS[lv][j]
                f.write(f"L{lv}_{name} {v}\n")

def evaluate_one(args):
    chrom, idx = args
    fpath = f"/tmp/ga_l{LEVEL}_p{idx}.txt"
    chrom_to_file(chrom, fpath)
    try:
        res = subprocess.run([EXE, fpath, str(LEVEL)], capture_output=True, text=True, timeout=120, cwd=CWD)
        out = res.stdout
        for line in out.splitlines():
            if f"LEVEL {LEVEL} SCORE:" in line:
                score = int(line.split(":")[-1].strip())
                return score, chrom
    except Exception as e:
        print(f"eval error L{LEVEL} idx{idx}: {e}")
    return 0, chrom

def tournament_select(pop, fitness, k=3):
    best = None
    best_f = -1e9
    for _ in range(k):
        idx = random.randrange(len(pop))
        if fitness[idx] > best_f:
            best_f = fitness[idx]
            best = pop[idx]
    return best

def crossover(p1, p2):
    c1, c2 = p1[:], p2[:]
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < MUT_PB:
            name, typ, lo, hi, step = PARAMS[i]
            if typ == int:
                delta = random.choice([-1, 1]) * step * random.randint(1, 5)
                chrom[i] = clamp(chrom[i] + delta, lo, hi)
                chrom[i] = round_step(chrom[i], step, typ)
            else:
                delta = random.choice([-1.0, 1.0]) * step * random.randint(1, 5)
                chrom[i] = clamp(chrom[i] + delta, lo, hi)
                chrom[i] = round_step(chrom[i], step, typ)
    return chrom

def main():
    print(f"[GA Level {LEVEL}] pop={POP_SIZE} gen={GENERATIONS} workers={WORKERS}")
    pool = multiprocessing.Pool(WORKERS)

    pop = []
    for _ in range(POP_SIZE // 2):
        pop.append(make_chromosome(DEFAULTS, noise=0.20))
    for _ in range(POP_SIZE - len(pop)):
        pop.append(make_chromosome())

    best_overall = -1
    best_chrom = None
    out_best = f"best_l{LEVEL}.txt"
    out_log = f"ga_l{LEVEL}.log"

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

            msg = f"L{LEVEL} Gen{gen:02d} best={gen_best} overall={best_overall} time={time.time()-t0:.1f}s"
            if improved:
                msg += " **IMPROVED**"
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

            # 停止条件
            if LEVEL <= 2 and best_overall >= 4500:
                print(f"Target reached for L{LEVEL}!")
                break
            if LEVEL == 3 and best_overall >= 5800:
                print(f"Target reached for L{LEVEL}!")
                break
            if LEVEL == 4 and best_overall >= 6200:
                print(f"Target reached for L{LEVEL}!")
                break
            if LEVEL == 5 and best_overall >= 10500:
                print(f"Target reached for L{LEVEL}!")
                break

            # 生成下一代
            new_pop = [pop[i][:] for i in range(ELITE)]
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop, fitness, TOURNAMENT_SIZE)
                p2 = tournament_select(pop, fitness, TOURNAMENT_SIZE)
                if random.random() < CX_PB:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2)
            pop = new_pop

    pool.close()
    pool.join()
    print(f"[GA Level {LEVEL}] FINAL BEST = {best_overall}")
    if best_chrom:
        chrom_to_file(best_chrom, out_best)

if __name__ == "__main__":
    main()
