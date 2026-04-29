#!/usr/bin/env python3
"""
针对 seed 114514 的专项遗传算法优化。
目标: 30000+ 分（当前基线 26390/25908）
评估: 只跑 seed 114514
"""
import subprocess, random, os, sys, math, multiprocessing, time

POP_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 12
GENERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
SEED = 114514
ELITE = 2
CX_PB = 0.75
MUT_PB = 0.28
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

# Use ORIGINAL defaults as base (better on seed 114514)
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

def make_chrom(noise=0.15):
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

def evaluate_one(args):
    chrom, idx = args
    fpath = f"/tmp/ga_114514_p{idx}.txt"
    chrom_to_file(chrom, fpath)
    try:
        res = subprocess.run([EXE, str(SEED), fpath], capture_output=True, text=True, timeout=120, cwd=CWD)
        # Parse total score
        total = 0
        for line in res.stdout.splitlines():
            if 'TOTAL SCORE:' in line:
                total = int(line.split(':')[-1].strip())
                break
        return total, chrom
    except Exception as e:
        print(f"eval error: {e}")
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
            name, typ, lo, hi, step = PARAMS[i % len(PARAMS)]
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
    print(f"[GA Seed 114514] pop={POP_SIZE} gen={GENERATIONS} target={TARGET} workers={WORKERS}")
    pool = multiprocessing.Pool(WORKERS)

    pop = []
    for _ in range(POP_SIZE // 2):
        pop.append(make_chrom(noise=0.15))
    for _ in range(POP_SIZE - len(pop)):
        pop.append(make_chrom(noise=0.35))

    best_overall = -1
    best_chrom = None
    out_best = "best_seed_114514.txt"
    out_log = "ga_114514.log"

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
            if improved:
                msg += " **IMPROVED**"
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

            if best_overall >= TARGET:
                print(f"TARGET {TARGET} REACHED!")
                break

            # Random restart if stuck for 5 generations
            no_improve_count = getattr(main, '_no_improve', 0)
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
            main._no_improve = no_improve_count

            if no_improve_count >= 5:
                print(f"Random restart at gen {gen}")
                main._no_improve = 0
                new_pop = [pop[i][:] for i in range(ELITE)]  # keep elites
                while len(new_pop) < POP_SIZE:
                    new_pop.append(make_chrom(noise=0.30))
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
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2)
            pop = new_pop

    pool.close()
    pool.join()
    print(f"[GA Seed 114514] FINAL BEST = {best_overall}")
    if best_chrom:
        chrom_to_file(best_chrom, out_best)
        print(f"Best params saved to {out_best}")

if __name__ == "__main__":
    main()
