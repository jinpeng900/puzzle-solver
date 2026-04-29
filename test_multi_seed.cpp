#include <iostream>
#include <vector>
#include <random>
#include <sstream>
// Forward declarations to link with o.cpp objects
struct Cell;
struct Board;
struct LevelParams;
extern LevelParams g_params[6];
extern int g_cur_level;
extern void init_default_params();

// We need to include the solver. Instead, let's just modify o.cpp temporarily
// Actually, easier approach: compile a separate binary with modified seed
