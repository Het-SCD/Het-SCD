/*
 * Het-SCD: High Quality Community Detection on Heterogenous Platforms 
 * Copyright (C) 2015, S. Heldens
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// MAP macro which performs an operation on each element of a list
// thanks to: https://github.com/swansontec/map-macro
#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0 (EVAL0 (EVAL0 (__VA_ARGS__)))
#define EVAL2(...) EVAL1 (EVAL1 (EVAL1 (__VA_ARGS__)))
#define EVAL3(...) EVAL2 (EVAL2 (EVAL2 (__VA_ARGS__)))
#define EVAL4(...) EVAL3 (EVAL3 (EVAL3 (__VA_ARGS__)))
#define EVAL(...) EVAL4 (EVAL4 (EVAL4 (__VA_ARGS__)))

#define MAP_END(...)
#define MAP_OUT
#define MAP_GET_END() 0, MAP_END
#define MAP_NEXT0(test, next, ...) next MAP_OUT
#define MAP_NEXT1(test, next) MAP_NEXT0 (test, next, 0)
#define MAP_NEXT(test, next) MAP_NEXT1 (MAP_GET_END test, next)

#define MAP0(f, x, peek, ...) ,f(x) MAP_NEXT (peek, MAP1) (f, peek, __VA_ARGS__)
#define MAP1(f, x, peek, ...) ,f(x) MAP_NEXT (peek, MAP0) (f, peek, __VA_ARGS__)
#define MAP(f, x, ...) f(x) EVAL (MAP1 (f, __VA_ARGS__, (), 0))


template <typename T>
inline static std::vector<T> singleton_vector(T var) {
    std::vector<T> l;
    l.push_back(var);
    return l;
}

inline static int ceil_div(int a, int b) {
    return a / b + (a % b != 0);
}

template <typename T>
inline static std::string to_string (T var) {
    std::ostringstream ss;
    ss << var;
    return ss.str();
}

double timer();
void timer_start(const char *name);
void timer_end(void);

#endif
