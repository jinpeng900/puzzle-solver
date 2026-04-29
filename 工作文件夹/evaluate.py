#!/usr/bin/env python3
"""
多种子评估脚本：编译 solver_test 并测试多个种子
用法: python3 evaluate.py [--seeds 5] [--compile]
"""
import subprocess, os, sys, argparse, statistics, time

CWD = os.path.dirname(os.path.abspath(__file__))

FILES = ["solver_core.h", "solver_core.cpp", "solver_test.cpp"]
EXE = "solver_test.exe" if sys.platform == "win32" else "solver_test"
SUBMIT_EXE = "solver.exe" if sys.platform == "win32" else "solver"
SUBMIT_FILES = ["solver_core.h", "solver_core.cpp", "solver.cpp"]

def compile_test():
    """编译本地测试版"""
    cmd = ["g++", "-std=c++17", "-O3", "-o", EXE,
           os.path.join(CWD, "solver_core.cpp"),
           os.path.join(CWD, "solver_test.cpp")]
    print(f"[compile] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE ERROR:\n{result.stderr}")
        return False
    print("[compile] OK")
    return True

def compile_submit():
    """编译线上提交版"""
    cmd = ["g++", "-std=c++17", "-O3", "-o", SUBMIT_EXE,
           os.path.join(CWD, "solver_core.cpp"),
           os.path.join(CWD, "solver.cpp")]
    print(f"[compile] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE ERROR:\n{result.stderr}")
        return False
    print("[compile submit] OK")
    return True

def run_seed(seed, exe_path):
    """运行一个种子并返回分数"""
    try:
        result = subprocess.run(
            [os.path.join(CWD, exe_path), str(seed)],
            cwd=CWD, capture_output=True, text=True, timeout=600)
        # 解析输出 - 最后一行是总分或avg
        lines = result.stdout.strip().split('\n')
        # 找到 "Seed total:" 或 "Average" 行
        scores = []
        for line in lines:
            if "Seed total:" in line:
                try:
                    scores.append(int(line.split(":")[-1].strip()))
                except: pass
        return scores[-1] if scores else 0
    except Exception as e:
        print(f"  ERROR seed {seed}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="测试种子数")
    parser.add_argument("--compile", action="store_true", help="重新编译")
    parser.add_argument("--submit", action="store_true", help="编译线上提交版")
    args = parser.parse_args()

    if args.submit:
        if not compile_submit():
            sys.exit(1)
        # 提交版不需要本地测试（它从stdin交互）
        print(f"Submit binary: {os.path.join(CWD, SUBMIT_EXE)}")
        return

    if args.compile or not os.path.exists(os.path.join(CWD, EXE)):
        if not compile_test():
            sys.exit(1)

    # 生成测试种子
    import random
    random.seed(42)
    seeds = [random.randint(1000, 999999) for _ in range(args.seeds)]
    print(f"Testing {len(seeds)} seeds: {seeds}")

    total_scores = []
    for seed in seeds:
        print(f"\nSeed {seed}:", end=" ", flush=True)
        t0 = time.time()
        score = run_seed(seed, EXE)
        dt = time.time() - t0
        total_scores.append(score)
        print(f"{score} ({dt:.1f}s)")

    avg = sum(total_scores) / len(total_scores) if total_scores else 0
    stdev = statistics.stdev(total_scores) if len(total_scores) > 1 else 0
    print(f"\n{'='*50}")
    print(f"Results: avg={avg:.1f} std={stdev:.1f}")
    print(f"Scores: {total_scores}")
    print(f"Min: {min(total_scores)}  Max: {max(total_scores)}")

if __name__ == "__main__":
    main()
