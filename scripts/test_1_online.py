"""test_1_online.py — 通过在线协议测试 1_msvc.exe"""
import subprocess, random, sys

def gen_block(rng, level):
    if level <= 2: return (rng.randint(0,4)) + 1
    if level == 3: return 0 if (rng.randint(0,99) < 15) else (rng.randint(0,4) + 1)
    if level == 4:
        c = (rng.randint(0,4) + 1)
        return -c if (rng.randint(0,99) < 10) else c
    if (rng.randint(0,99) < 15): return 0
    base = (rng.randint(0,4) + 1)
    return -base if (rng.randint(0,99) < 10) else base

def path_score(k):
    import math
    t = math.sqrt(k) - 1.0
    return 10*k + 18*int(t*t)

def path_score_b(board, path):
    k = len(path)
    s = path_score(k)
    in_path = set(path)
    N = len(board)
    exploded = set()
    for r,c in path:
        if board[r][c] >= 0: continue
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                nr,nc = r+dr, c+dc
                if 0<=nr<N and 0<=nc<N and (nr,nc) not in in_path and (nr,nc) not in exploded:
                    exploded.add((nr,nc))
                    s += 10
    return s

def gen_initial(level, seed, N):
    rng = random.Random(seed)
    drop_q = [[gen_block(rng, level) for _ in range(1000)] for _ in range(N)]
    rng_b = random.Random(seed ^ 0x9E3779B9)
    board = [[gen_block(rng_b, level) for _ in range(N)] for _ in range(N)]
    return board, drop_q

def is_deadlocked(board):
    N = len(board)
    for r in range(N):
        for c in range(N):
            ac = abs(board[r][c])
            if c+1<N:
                c2 = abs(board[r][c+1])
                if ac==c2 or ac==0 or c2==0: return False
            if r+1<N:
                c2 = abs(board[r+1][c])
                if ac==c2 or ac==0 or c2==0: return False
    return True

def apply_move(board, drop_q, q_ptrs, path, level):
    N = len(board)
    in_path = set(path)
    to_remove = set(path)
    if level >= 4:
        for r,c in path:
            if board[r][c] >= 0: continue
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    nr,nc = r+dr, c+dc
                    if 0<=nr<N and 0<=nc<N and (nr,nc) not in in_path:
                        to_remove.add((nr,nc))
    for c in range(N):
        remaining = [board[r][c] for r in range(N) if (r,c) not in to_remove]
        empty = N - len(remaining)
        for i in range(empty):
            board[i][c] = drop_q[c][q_ptrs[c]]
            q_ptrs[c] += 1
        for i, val in enumerate(remaining):
            board[empty+i][c] = val
    return board

def test_level(seed, level):
    N = 12 if level == 5 else 10
    board, drop_q = gen_initial(level, seed, N)
    q_ptrs = [0]*N

    # 启动求解器
    proc = subprocess.Popen(
        ["./1_msvc.exe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )

    # 发送 LEVEL 消息
    level_line = f"LEVEL {level} SEED {seed} SIZE {N} STEPS 50\n"
    proc.stdin.write(level_line)
    # 发送初始棋盘
    for row in board:
        proc.stdin.write(" ".join(str(v) for v in row) + "\n")
    proc.stdin.flush()

    step = 0
    total_score = 0

    while step < 50 and not is_deadlocked(board):
        # 读取求解器的响应
        try:
            response = proc.stdout.readline().strip()
        except:
            break
        if not response:
            break

        parts = response.split()
        if len(parts) < 3:
            break
        path_len = int(parts[0])
        path = [(int(parts[i]), int(parts[i+1])) for i in range(1, len(parts), 2)]

        if len(path) < 2:
            break

        # 验证路径
        valid = True
        seen = set()
        anchor = 0
        for i,(r,c) in enumerate(path):
            if not (0<=r<N and 0<=c<N): valid=False; break
            if (r,c) in seen: valid=False; break
            seen.add((r,c))
            co = abs(board[r][c])
            if co != 0:
                if anchor == 0: anchor = co
                elif anchor != co: valid=False; break
            if i > 0:
                pr,pc = path[i-1]
                if abs(pr-r) + abs(pc-c) != 1: valid=False; break

        if not valid:
            # 发送无效响应
            step += 1
            board = apply_move(board, drop_q, q_ptrs, [], level)
            step_line = f"STEP {step} SCORE {total_score} INVALID\n"
            proc.stdin.write(step_line)
            for row in board:
                proc.stdin.write(" ".join(str(v) for v in row) + "\n")
            proc.stdin.flush()
            continue

        gained = path_score_b(board, path)
        total_score += gained
        step += 1

        board = apply_move(board, drop_q, q_ptrs, path, level)

        step_line = f"STEP {step} SCORE {total_score} VALID\n"
        proc.stdin.write(step_line)
        for row in board:
            proc.stdin.write(" ".join(str(v) for v in row) + "\n")
        proc.stdin.flush()

    proc.stdin.close()
    proc.wait()
    return total_score, step

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 114514
print(f"=== 1.cpp test seed={seed} ===")
total = 0
for level in range(1, 6):
    score, steps = test_level(seed, level)
    print(f"  L{level}: {score} pts in {steps} steps")
    total += score
print(f"  TOTAL: {total}")
