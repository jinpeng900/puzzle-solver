#!/usr/bin/env python3
"""
多种子稳定遗传算法：追求 avg - k*std 最大化
用法: python3 ga_stable.py [n_seeds] [pop] [gen]
"""
import subprocess, random, os, sys, statistics, multiprocessing, time, re

N_SEEDS   = int(sys.argv[1]) if len(sys.argv) > 1 else 5
POP_SIZE  = int(sys.argv[2]) if len(sys.argv) > 2 else 14
GENS      = int(sys.argv[3]) if len(sys.argv) > 3 else 30
ELITE     = 3
CX_PB     = 0.75
MUT_PB    = 0.25
TOURNAMENT_SIZE = 5
WORKERS   = max(1, multiprocessing.cpu_count() * 2 // 3)
EXE       = "./eval_seed"
CWD       = os.path.dirname(os.path.abspath(__file__))

PARAMS = [
    ("dfs_limit", int, 300000, 1800000, 50000),
    ("keep1", int, 30, 400, 5),
    ("keep2", int, 10, 220, 5),
    ("keep3", int, 2, 40, 1),
    ("cache_limit", int, 15000, 120000, 5000),
    ("mxc_w", float, 6.0, 35.0, 0.5),
    ("dq_w", float, 0.0, 2.0, 0.05),
    ("bh_w", float, 0.0, 0.10, 0.005),
    ("s3_dq_w", float, 0.0, 1.5, 0.05),
    ("s3_bh_w", float, 0.0, 0.08, 0.005),
    ("surv_mul", float, 0.1, 3.0, 0.1),
    ("beam_w", int, 0, 12, 1),
    ("beam_d", int, 0, 8, 1),
    ("short_pen", int, 0, 150, 5),
    ("surv_check", int, 2, 12, 1),
    ("beam_bonus", float, 0.0, 1.0, 0.05),
    ("len_bonus", float, 0.0, 0.6, 0.05),
    ("tfast_ratio", float, 0.65, 0.97, 0.01),
    ("tdead_ratio", float, 0.80, 0.99, 0.01),
    ("max_time_ms", int, 4000, 16000, 500),
    ("fallback_len_thresh", int, 2, 10, 1),
]

def load_params_from_file(path):
    vals = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) == 2:
                    v = float(parts[1])
                    vals[parts[0]] = int(v) if v == int(v) and '.' not in parts[1] else v
    return vals

def vals_to_chrom(vals):
    chrom = []
    for lv in range(1, 6):
        for name, typ, lo, hi, step in PARAMS:
            key = f"L{lv}_{name}"
            default = {
                1: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,0.5,8,4,55,6,0.3,0.0,0.90,0.97,10000,6],
                2: [1200000,200,100,20,80000,25.0,0.8,0.02,0.55,0.025,1.0,8,4,25,6,0.3,0.0,0.90,0.97,10000,6],
                3: [ 700000,120, 60,12,60000,18.0,0.9,0.02,0.60,0.025,1.0,0,0,55,6,0.3,0.0,0.90,0.97,10000,6],
                4: [ 600000,120, 60,12,80000,15.0,0.8,0.02,0.55,0.025,1.0,5,3,55,6,0.3,0.0,0.90,0.97,10000,6],
                5: [ 800000, 60, 30, 6,50000,15.0,0.9,0.02,0.60,0.025,1.0,0,0,55,6,0.3,0.0,0.90,0.97,10000,6],
            }
            i = [p[0] for p in PARAMS].index(name)
            chrom.append(vals.get(key, default[lv][i]))
    return chrom

def chrom_to_file(chrom, fpath):
    with open(fpath, 'w') as f:
        idx = 0
        for lv in range(1, 6):
            for name, typ, lo, hi, step in PARAMS:
                f.write(f"L{lv}_{name} {chrom[idx]}\n")
                idx += 1

def clamp(v, lo, hi): return max(lo, min(hi, v))

def round_step(v, step, typ):
    if typ == int:
        return int((int(round(v / step)) * step))
    else:
        prec = max(0, len(str(step).split('.')[-1])) if '.' in str(step) else 0
        return round(round(v / step) * step, prec)

def make_chrom(base_chrom, noise=0.15):
    chrom = base_chrom[:]
    for i in range(len(chrom)):
        lv_idx = i // len(PARAMS)
        p_idx = i % len(PARAMS)
        name, typ, lo, hi, step = PARAMS[p_idx]
        span = (hi - lo) * noise
        if typ == int:
            v = int(round(chrom[i] + random.uniform(-span, span)))
        else:
            v = chrom[i] + random.uniform(-span, span)
        chrom[i] = round_step(clamp(v, lo, hi), step, typ)
    return chrom

def evaluate_one(args):
    chrom, idx, eval_seeds = args
    fpath = f"/tmp/ga_stable_{idx}.txt"
    chrom_to_file(chrom, fpath)
    scores = []
    for s in eval_seeds:
        try:
            res = subprocess.run([EXE, str(s), fpath], capture_output=True, text=True, timeout=300, cwd=CWD)
            sc = int(res.stdout.strip())
            scores.append(sc)
        except Exception as e:
            return -1e9, chrom, idx
    avg = sum(scores) / len(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0
    fitness = avg - 0.35 * std - 0.15 * (avg - min(scores))
    return fitness, chrom, idx

def tournament_select(pop, fitness, k=TOURNAMENT_SIZE):
    best, best_f = None, -1e9
    for _ in range(k):
        idx = random.randrange(len(pop))
        if fitness[idx] > best_f: best_f, best = fitness[idx], pop[idx]
    return best

def crossover(p1, p2):
    c1, c2 = p1[:], p2[:]
    for i in range(len(p1)):
        if random.random() < 0.5: c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < MUT_PB:
            name, typ, lo, hi, step = PARAMS[i % len(PARAMS)]
            delta = random.choice([-1, 1]) * step * random.randint(1, 4)
            if typ == int:
                chrom[i] = round_step(clamp(chrom[i] + delta, lo, hi), step, typ)
            else:
                chrom[i] = round_step(clamp(chrom[i] + delta, lo, hi), step, typ)
    return chrom

def main():
    print(f"[GA Stable] seeds={N_SEEDS} pop={POP_SIZE} gen={GENS} workers={WORKERS}")
    pool = multiprocessing.Pool(WORKERS)

    # 加载当前最优参数作为起点
    base_vals = load_params_from_file(os.path.join(CWD, "best_all_v2.txt"))
    base_chrom = vals_to_chrom(base_vals)
    
    # 固定评估种子（用于每代一致性比较）
    eval_seeds = [random.randint(1, 1000000) for _ in range(N_SEEDS)]
    print(f"Eval seeds: {eval_seeds}")

    # 初始种群
    pop = []
    for _ in range(POP_SIZE // 2):
        pop.append(make_chrom(base_chrom, noise=0.10))
    for _ in range(POP_SIZE - len(pop)):
        pop.append(make_chrom(base_chrom, noise=0.30))

    best_fitness = -1e9
    best_chrom = None
    best_avg = 0
    out_best = os.path.join(CWD, "best_stable_v2.txt")
    out_log  = os.path.join(CWD, "ga_stable_v2.log")

    # 评估基线
    base_fitness, _, _ = evaluate_one((base_chrom, -1, eval_seeds))
    print(f"Baseline fitness: {base_fitness:.1f}")

    with open(out_log, 'w') as logf:
        no_improve = 0
        for gen in range(GENS):
            t0 = time.time()
            args = [(pop[i], i, eval_seeds) for i in range(len(pop))]
            results = pool.map(evaluate_one, args)
            fitness = [r[0] for r in results]

            sorted_idx = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
            pop = [pop[i][:] for i in sorted_idx]
            fitness = [fitness[i] for i in sorted_idx]

            gen_best = fitness[0]
            improved = False
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_chrom = pop[0][:]
                chrom_to_file(best_chrom, out_best)
                # 计算最佳染色体的实际 avg
                _, _, _ = evaluate_one((best_chrom, -2, eval_seeds))
                improved = True

            msg = f"Gen{gen:02d} bestFit={gen_best:.1f} overallFit={best_fitness:.1f} time={time.time()-t0:.1f}s"
            if improved: msg += " **IMPROVED**"
            print(msg)
            logf.write(msg + "\n"); logf.flush()

            if not improved: no_improve += 1
            else: no_improve = 0

            # 停滞超过 7 代 → 重启
            if no_improve >= 7 and gen < GENS - 5:
                print(f"  [restart] stuck {no_improve} gens")
                no_improve = 0
                new_pop = [pop[i][:] for i in range(ELITE)]
                while len(new_pop) < POP_SIZE:
                    new_pop.append(make_chrom(best_chrom, noise=0.25))
                pop = new_pop
                continue

            # 生成下一代
            new_pop = [pop[i][:] for i in range(ELITE)]
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop, fitness)
                p2 = tournament_select(pop, fitness)
                if random.random() < CX_PB:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1, c2 = mutate(c1), mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE: new_pop.append(c2)
            pop = new_pop

    pool.close()
    pool.join()

    # 最终验证：在多组新种子上评估最优参数
    print(f"\n=== Final verification ===")
    verify_seeds = [random.randint(1, 1000000) for _ in range(8)]
    fpath = "/tmp/ga_verify_final.txt"
    chrom_to_file(best_chrom, fpath)
    scores = []
    for s in verify_seeds:
        res = subprocess.run([EXE, str(s), fpath], capture_output=True, text=True, timeout=300, cwd=CWD)
        sc = int(res.stdout.strip())
        scores.append(sc)
        print(f"  seed {s}: {sc}")
    print(f"Final: avg={sum(scores)/len(scores):.1f} std={statistics.stdev(scores):.1f} min={min(scores)} max={max(scores)}")
    print(f"Best params saved to {out_best}")

if __name__ == "__main__":
    main()
