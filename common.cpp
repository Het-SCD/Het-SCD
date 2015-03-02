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
#include <iostream>
#include <stack>
#include <string>
#include <sys/time.h>

#include "common.hpp"

using namespace std;

double timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static stack<pair<string, double> > timers;

void timer_start(const char *name) {
    timers.push(make_pair(string(name), timer()));
}

void timer_end(void) {
    if (timers.empty()) {
        return;
    }

    pair<string, double> t = timers.top();
    timers.pop();

    string name = t.first;
    double before = t.second;
    double after = timer();

    cout << t.first << ": " << (after - before) << " ms" << endl;
}
