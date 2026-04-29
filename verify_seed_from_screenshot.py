#!/usr/bin/env python3
"""
验证截图中的棋盘是否匹配指定的 seed
用法: python3 verify_seed_from_screenshot.py [seed]
"""
import sys
from PIL import Image

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 114514
    screenshot = "截图 2026-04-28 12-07-12.png"

    img = Image.open(screenshot).convert("RGBA")
    w, h = img.size

    # 从之前分析得到的网格边界
    h_lines = [31, 68, 105, 142, 179, 216, 253, 290, 327, 364, 401]
    v_lines = [7, 46, 83, 120, 157, 194, 231, 268, 305, 342, 379]

    # 5 种颜色的参考 RGB（从截图中提取的中心点）
    color_map = {
        (231, 76, 60): 1,   # 红色 = 1
        (52, 152, 219): 2,  # 蓝色 = 2
        (46, 204, 113): 3,  # 绿色 = 3
        (243, 156, 18): 4,  # 橙色 = 4
        (155, 89, 182): 5,  # 紫色 = 5
    }

    def nearest_color(rgb):
        best = None
        best_dist = float('inf')
        for ref, val in color_map.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, ref))
            if dist < best_dist:
                best_dist = dist
                best = val
        return best

    # 提取截图的 10x10 数字网格
    screenshot_grid = []
    for r in range(10):
        row = []
        for c in range(10):
            cx = (h_lines[c] + h_lines[c + 1]) // 2
            cy = (v_lines[r] + v_lines[r + 1]) // 2
            rgb = img.getpixel((cx, cy))[:3]
            row.append(nearest_color(rgb))
        screenshot_grid.append(row)

    print("=" * 60)
    print(f"截图提取的 10x10 数字网格:")
    print("=" * 60)
    for row in screenshot_grid:
        print(" ".join(str(x) for x in row))

    # seed 114514 的 Level 1 初始棋盘（来自 0.txt 评测日志）
    expected_grid = [
        [1, 5, 4, 5, 3, 4, 5, 4, 5, 4],
        [5, 1, 3, 3, 3, 1, 2, 3, 4, 5],
        [5, 2, 1, 3, 4, 5, 1, 1, 5, 4],
        [4, 5, 4, 3, 1, 1, 2, 3, 1, 1],
        [1, 3, 1, 1, 5, 5, 5, 3, 3, 1],
        [1, 1, 3, 1, 3, 5, 3, 3, 3, 5],
        [2, 3, 5, 4, 1, 1, 2, 1, 4, 1],
        [2, 4, 1, 3, 1, 1, 3, 5, 2, 2],
        [2, 3, 3, 3, 1, 1, 5, 4, 3, 5],
        [4, 3, 4, 5, 5, 2, 4, 3, 1, 4],
    ]

    print()
    print("=" * 60)
    print(f"seed={seed} 的 Level 1 初始棋盘（评测日志）:")
    print("=" * 60)
    for row in expected_grid:
        print(" ".join(str(x) for x in row))

    # 对比
    print()
    print("=" * 60)
    print("对比结果:")
    print("=" * 60)
    match_count = 0
    mismatch = []
    for r in range(10):
        for c in range(10):
            if screenshot_grid[r][c] == expected_grid[r][c]:
                match_count += 1
            else:
                mismatch.append((r, c, screenshot_grid[r][c], expected_grid[r][c]))

    print(f"匹配格子: {match_count} / 100")
    print(f"匹配率: {match_count}%")

    if mismatch:
        print(f"\n不匹配的位置 ({len(mismatch)} 个):")
        for r, c, got, exp in mismatch:
            print(f"  row={r}, col={c}: 截图={got}, seed={exp}")
    else:
        print("\n✅ 截图与 seed 的初始棋盘 100% 匹配!")

    return 0 if len(mismatch) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
