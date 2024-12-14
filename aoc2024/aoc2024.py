import math
import random
from collections import defaultdict

def daytemp():
    f = open('input.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    grid = []
    for i in range(len(f)):
        row = f[i]
        nums = [int(x) for x in row.split()]
        
    return ans

def day1():
    f = open('aoc1.txt', 'r').read().strip().split("\n")
    ans = 0
    list1 = []
    list2 = []
    for i in range(len(f)):
        row = f[i]
        rowsplit = row.split()
        list1.append(int(rowsplit[0]))
        list2.append(int(rowsplit[1]))
    
    x = sorted(list1)
    y = sorted(list2)

    # part 1
    # diffs = [abs(x[i]-y[i]) for i in range(len(x))]
    # return sum(diffs)

    # part 2
    for xnum in x:
        ans += xnum*(y.count(xnum))
    return ans


def day2():
    f = open('aoc2.txt', 'r').read().strip().split("\n")
    ans = 0

    def is_valid(nums):
        return all((1 <= nums[i]-nums[i-1] <= 3) for i in range(1, len(nums))) or all((1 <= nums[i-1]-nums[i] <= 3) for i in range(1, len(nums)))

    for row in f:
        nums = [int(i) for i in row.split()]

        # part 1
        if is_valid(nums):
            ans += 1
        # include for part 2
        else:
            for j in range(len(nums)):
                newnums = nums[:j] + nums[j+1:]
                if is_valid(newnums):
                    ans += 1
                    break
        
    return ans

def day3():
    f = open('aoc3.txt', 'r').read().strip()
    ans = 0
    
    def is_valid_num(num):
        return 1 <= len(num) <= 3 and all(x in "0123456789" for x in num)
    
    def is_valid(chunk):
        chunksplit = chunk.split(",")
        return is_valid_num(chunksplit[0]) and is_valid_num(chunksplit[1])

    chunks = f.split("mul(")
    enabled = True
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        chunkbeginning = chunk.split(")")[0]
        if enabled and is_valid(chunkbeginning):
            s = chunkbeginning.split(",")
            ans += int(s[0])*int(s[1])
        
        # for part 2
        for i in range(len(chunk) - 4):
            if chunk[i:i+4] == "do()":
                enabled = True
            elif chunk[i:i+7] == "don't()":
                enabled = False
        
    return ans

def day4():
    grid = open('aoc4.txt', 'r').read().strip().split("\n")
    ans = 0
    
    def directions(x, y):
        dirs = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        options = []
        for dir in dirs:
            option = [(x+dir[0]*i, y+dir[1]*i) for i in range(4)]
            options.append(option)
        return options
    
    def in_range(option):
        return all(0 <= x[0] < len(grid) for x in option) and all (0 <= x[1] < len(grid[0]) for x in option)
    
    # part 1
    for xpos in range(len(grid)):
        for ypos in range(len(grid[0])):
            options = directions(xpos, ypos)
            for option in options:
                if in_range(option):
                    xmas = ''.join([grid[o[0]][o[1]] for o in option])
                    if xmas == "XMAS":
                        ans += 1

    # part 2
    # for xpos in range(1, len(grid)-1):
    #     for ypos in range(1, len(grid[0])-1):
    #         if grid[xpos][ypos] == "A":
    #             if set([grid[xpos-1][ypos-1], grid[xpos+1][ypos+1]]) == {"M", "S"} and set([grid[xpos-1][ypos+1], grid[xpos+1][ypos-1]]) == {"M", "S"}:
    #                 ans += 1

    return ans

def day5():
    f = open('aoc5.txt', 'r').read().strip().split("\n\n")
    befores = f[0].split("\n")
    pagelists = f[1].split("\n")

    ans = 0
    m = defaultdict(set)
    
    for before in befores:
        bsplit = before.split("|")
        m[bsplit[0]].add(bsplit[1])
    
    def reorder_pagelist(pagelist):
        p = pagelist.split(",")
        # build an adjacency list with only the pages in the list
        mini_graph = defaultdict(set)
        for i in range(len(p)):
            mini_graph[p[i]] = set()
            for j in range(i+1, len(p)):
                if p[j] in m[p[i]]:
                    mini_graph[p[i]].add(p[j])
        
        # builds printing order backwards
        new_order = []
        while len(new_order) < len(p):
            for key in mini_graph:
                # find each key that has no restrictions, add it, and remove it as a restriction from all other sets
                if len(mini_graph[key]) == 0:
                    new_order.append(key)
                    for every_key in mini_graph:
                        mini_graph[every_key].discard(key)
                    del mini_graph[key]
                    break
        
        return int(new_order[len(new_order)//2])

    for pagelist in pagelists:
        p = pagelist.split(",")
        should_add = True
        for i in range(len(p)):
            for j in range(i+1, len(p)):
                if p[i] in m[p[j]]:
                    should_add = False
                    break
        # part 1
        # if should_add:
        #     ans += int(p[len(p)//2])

        # part 2
        if not should_add:
            ans += reorder_pagelist(pagelist)

    return ans

def day6():
    f = open('aoc6.txt', 'r').read().strip().split("\n")
    grid = [list(row) for row in f]
    ans = 0
    startpos = None
    m = {
        (-1,0): (0,1),
        (0,1): (1,0),
        (1,0): (0,-1),
        (0,-1): (-1,0)
    }
    startdir = (-1, 0)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "^":
                startpos = (i, j)
                break

    # part 1
    # visited = set()
    # visited.add(startpos)
    # pos = startpos
    # dir = startdir
    # while True:
    #     newpos = (pos[0] + dir[0], pos[1] + dir[1])
    #     if not (0 <= newpos[0] < len(grid) and 0 <= newpos[1] < len(grid[0])):
    #         break
    #     if grid[newpos[0]][newpos[1]] == "#":
    #         dir = m[dir]  # turn 90 degrees
    #     else:
    #         visited.add(newpos)
    #         pos = newpos
    # return len(visited)
    
    def obstruct(row, col):
        seen = set()
        if grid[row][col] != ".":
            return False

        grid[row][col] = "#"

        pos = startpos
        dir = startdir
        seen.add((pos, dir))
        succeeds = False
        while True:
            newpos = (pos[0] + dir[0], pos[1] + dir[1])
            if not (0 <= newpos[0] < len(grid) and 0 <= newpos[1] < len(grid[0])):
                break
            if (newpos, dir) in seen:
                # if we've been in the same spot traveling in the same direction before, we have a loop
                succeeds = True
                break
            if grid[newpos[0]][newpos[1]] == "#":
                dir = m[dir]  # turn 90 degrees
            else:
                seen.add((newpos, dir))
                pos = newpos
        
        grid[row][col] = "."
        return succeeds

    # there is probably a more clever way but brute force takes a minute or two
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if obstruct(i, j):
                ans += 1
    return ans

def day7():
    f = open('aoc7.txt', 'r').read().strip().split("\n")
    ans = 0
    
    def possible(total, nums):
        if len(nums) == 1:
            return total == nums[0]
        # part 1
        # return possible(total, [nums[0]+nums[1]] + nums[2:]) or possible(total, [nums[0]*nums[1]] + nums[2:])

        # part 2
        return possible(total, [nums[0]+nums[1]] + nums[2:]) or possible(total, [nums[0]*nums[1]] + nums[2:]) or possible(total, [int(str(nums[0]) + str(nums[1]))] + nums[2:])

    for row in f:
        rowsplit = row.split(": ")
        total = int(rowsplit[0])
        nums = [int(x) for x in rowsplit[1].split()]
        if possible(total, nums):
            ans += total
        
    return ans


def day8():
    f = open('aoc8.txt', 'r').read().strip().split("\n")
    m = defaultdict(set)

    for i in range(len(f)):
        for j in range(len(f[0])):
            val = f[i][j]
            if val != ".":
                m[val].add((i,j))

    def get_antinodes(p1, p2):
        # part 1
        # diff = (p1[0] - p2[0], p1[1] - p2[1])
        # a1 = (p1[0] + diff[0], p1[1] + diff[1])
        # a2 = (p2[0] - diff[0], p2[1] - diff[1])
        # return {x for x in [a1, a2] if 0<=x[0]<len(f) and 0<=x[1]<len(f[0])}

        # part 2
        diff = (p1[0]-p2[0], p1[1]-p2[1])
        gcd = math.gcd(diff[0], diff[1])
        newdiff = (diff[0] // gcd, diff[1] // gcd)
        antinodes = set()
        oldp1 = p1
        while (0<=p1[0]<len(f) and 0<=p1[1]<len(f[0])):
            antinodes.add(p1)
            p1 = (p1[0] + newdiff[0], p1[1] + newdiff[1])
        # reset p1, and now move in the other direction
        p1 = oldp1
        while (0<=p1[0]<len(f) and 0<=p1[1]<len(f[0])):
            antinodes.add(p1)
            p1 = (p1[0] - newdiff[0], p1[1] - newdiff[1])

        return antinodes
    
    antinodes = set()
    for key in m:
        pts = list(m[key])
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                antinodes.update(get_antinodes(pts[i], pts[j]))

    return len(antinodes)

def day9():
    f = open('aoc9.txt', 'r').read().strip()
    ans = 0

    # maps each ID to the indices that belong to it on the original disk
    m = defaultdict(list)

    sizes = [int(f[i]) for i in range(0, len(f), 2)]
    totalnums = sum(sizes)

    counter = 0
    allnums = []  # lists out every ID in order, leaving out spaces
    empties = []  # for each empty space, lists a tuple of (starting position, length of space)
    for i in range(0, len(f)):
        if i % 2 == 0:
            for _ in range(int(f[i])):
                allnums.append(i//2)
                m[i//2].append(counter)
                counter += 1
        else:
            empties.append((counter, int(f[i])))
            counter += int(f[i])

    # part 1
    # pos = 0  # index in the input f, moving forwards
    # backpos = len(allnums) - 1  # index in allnums, going backwards
    # pointer = pos  # starting location
    # while pointer < totalnums:
    #     num = int(f[pos])
    #     for _ in range(num):
    #         if pointer >= totalnums:
    #             break
    #         if pos % 2 == 0:
    #             ans += (pointer * (pos // 2))
    #             pointer += 1
    #         else:
    #             ans += (pointer * allnums[backpos])
    #             backpos -= 1
    #             pointer += 1
    #     pos += 1
    
    # part 2
    pos = len(sizes) - 1  # index in "sizes", moving backwards
    while pos > 0:
        moved = False
        for e in range(pos):  # only check empty spaces that come before the block we are moving
            empty = empties[e]
            if sizes[pos] <= empty[1]:
                moved = True
                for k in range(sizes[pos]):
                    ans += ((empty[0] + k) * pos)
                empties[e] = (empty[0] + sizes[pos], empty[1] - sizes[pos])
                break

        # if there was no space for the block, then use its current positions for the total, which are stored in m 
        if not moved:
            for p in m[pos]:
                ans += pos * p

        pos -= 1

    return ans

def day10():
    f = open('aoc10.txt', 'r').read().strip().split("\n")
    ans = 0
    m = defaultdict(set)
    grid = []
    for row in f:
        grid.append([int(i) for i in row])

    trailheads = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                trailheads.add((i, j))
    
    def score(trailhead, pos):
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        val = grid[pos[0]][pos[1]]
        if val == 9:
            m[trailhead].add(pos)
            return 1

        total_score = 0
        for dir in dirs:
            newpos = (dir[0]+pos[0], dir[1]+pos[1])
            if 0 <= newpos[0] < len(grid) and 0 <= newpos[1] < len(grid[0]) and grid[newpos[0]][newpos[1]] == val+1:
                total_score += score(trailhead, newpos)
        return total_score
        
    # part 1
    # for trailhead in trailheads:
    #     score(trailhead, trailhead)
    # for trailhead in m:
    #     ans += len(m[trailhead])

    # part 2
    for trailhead in trailheads:
        ans += score(trailhead, trailhead)

    return ans

def day11():
    f = open('aoc11.txt', 'r').read().strip()
    stones = [int(x) for x in f.split()]
    ans = 0
    dp = {}
    
    def blink(stone, remaining):
        if (stone, remaining) in dp:
            return dp[(stone, remaining)]
        
        if remaining == 0:
            result = 1
        elif stone == 0:
            result = blink(1, remaining-1)
        elif len(str(stone)) % 2 == 0:
            stonestr = str(stone)
            stone1 = int(stonestr[:len(stonestr)//2])
            stone2 = int(stonestr[len(stonestr)//2:])
            result = blink(stone1, remaining-1) + blink(stone2, remaining-1)
        else:
            result = blink(2024*stone, remaining-1)
        
        dp[(stone, remaining)] = result
        return result
            
    for stone in stones:
        # part 1
        # ans += blink(stone, 25)

        # part 2
        ans += blink(stone, 75)
    
    return ans

def day12():
    f = open('aoc12.txt', 'r').read().strip().split("\n")
    ans = 0
    unvisited = set((i, j) for i in range(len(f)) for j in range(len(f[0])))
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    
    # repeatedly take an unvisited location, floodfill out and remove all visited locations from unvisited, compute the score, and repeat
    while unvisited:
        loc = unvisited.pop()
        val = f[loc[0]][loc[1]]
        current_region = set()
        next_candidates = set([loc])
        while next_candidates:
            next_next_candidates = set()
            for candidate in next_candidates:
                current_region.add(candidate)
                unvisited.discard(candidate)
                for dir in dirs:
                    newloc = (candidate[0]+dir[0], candidate[1]+dir[1])
                    if newloc in unvisited and 0<=newloc[0]<len(f) and 0<=newloc[1]<len(f[0]) and f[newloc[0]][newloc[1]] == val:
                        next_next_candidates.add(newloc)
            next_candidates = next_next_candidates
        
        area = len(current_region)
        perimeter = 0
        edges = defaultdict(list)
        for r in current_region:
            for dir in dirs:
                newloc = (r[0]+dir[0], r[1]+dir[1])
                if newloc not in current_region:
                    edges[dir].append(r)
                    perimeter += 1
        
        # part 1
        # ans += area * perimeter

        # part 2
        # to calculate number of sides, we treat vertical / horizontal edges differently
        # group the edges based on which of the 4 directions they face, then within each group, sort values in the direction perpendicular to the direction the edges face
        # once sorted, traverse from left to right -- if two values share one coordinate and the other differs by 1, then they continue to be part of the same edge
        numedges = 0
        for dir in [(1,0), (-1,0)]:
            sortedges = sorted(edges[dir])
            if not sortedges:
                continue
            else:
                numedges += 1
                for i in range(1, len(sortedges)):
                    current = sortedges[i-1]
                    if not (sortedges[i][0] == current[0] and sortedges[i][1] == current[1]+1):
                        numedges += 1
        
        for dir in [(0,1), (0,-1)]:
            sortedges = sorted(edges[dir], key=lambda x: (x[1], x[0]))  # sort this in the reverse direction
            if not sortedges:
                continue
            else:
                numedges += 1
                for i in range(1, len(sortedges)):
                    current = sortedges[i-1]
                    if not (sortedges[i][0] == current[0]+1 and sortedges[i][1] == current[1]):  # compare by the other coordinate this time
                        numedges += 1
        ans += area * numedges
        
    return ans

def day13():
    f = open('aoc13.txt', 'r').read().strip().split("\n\n")
    ans = 0
    for row in f:
        data = row.split("\n")
        amoves = [int(i[2:]) for i in data[0].split(": ")[1].split(", ")]
        bmoves = [int(i[2:]) for i in data[1].split(": ")[1].split(", ")]

        # part 1
        # prizes = [int(i[2:]) for i in data[2].split(": ")[1].split(", ")]
        # minval = None
        # for a in range(min(prizes[0]//amoves[0], prizes[1]//amoves[1])+1):
        #     xdiff = prizes[0] - a*amoves[0]
        #     ydiff = prizes[1] - a*amoves[1]
        #     if xdiff % bmoves[0] == 0 and ydiff % bmoves[1] == 0 and xdiff//bmoves[0] == ydiff//bmoves[1]:
        #         score = 3*a + xdiff//bmoves[0]
        #         if minval is None or score < minval:
        #             minval = score
        # if minval is not None:
        #     ans += minval

        # part 2
        # in theory this shouldn't always work because we're not minimizing 3a+b... but for the given input it seems there is always only one solution for (a,b)
        prizes = [int(i[2:])+10000000000000 for i in data[2].split(": ")[1].split(", ")]
        # initially I used sympy but rewrote this because we can just solve the system of equations directly
        a = (prizes[0]*bmoves[1] - prizes[1]*bmoves[0]) // (amoves[0]*bmoves[1] - amoves[1]*bmoves[0])
        b = (prizes[0]*amoves[1] - prizes[1]*amoves[0]) // (amoves[1]*bmoves[0] - amoves[0]*bmoves[1])
        # check for nonnegative integer solutions
        if a >= 0 and b >= 0 and a*amoves[0] + b*bmoves[0] == prizes[0] and a*amoves[1] + b*bmoves[1] == prizes[1]:
            ans += 3*a + b

    return ans

def day14():
    f = open('aoc14.txt', 'r').read().strip().split("\n")
    robots = []
    width = 101
    height = 103
    for row in f:
        rowsplit = row.split(" ")
        ps = [int(x) for x in rowsplit[0][2:].split(",")]
        vs = [int(x) for x in rowsplit[1][2:].split(",")]
        p0 = (ps[0] + vs[0]*100) % width
        p1 = (ps[1] + vs[1]*100) % height
        robots.append((p0, p1))

    # part 1
    # ans = 1
    # q1 = ((0, width//2-1), (0, height//2-1))
    # q2 = ((0, width//2-1), (height//2+1, height-1))
    # q3 = ((width//2+1, width-1), (0, height//2-1))
    # q4 = ((width//2+1, width-1), (height//2+1, height-1))
    # qs = [q1, q2, q3, q4]
    # for q in qs:
    #     total = 0
    #     for r in robots:
    #         if q[0][0] <= r[0] <= q[0][1] and q[1][0] <= r[1] <= q[1][1]:
    #             total += 1
    #     ans *= total
    # return ans

    # part 2
    grid = [["." for _ in range(width)] for _ in range(height)]
    robots = []
    for row in f:
        rowsplit = row.split(" ")
        ps = [int(x) for x in rowsplit[0][2:].split(",")]
        vs = [int(x) for x in rowsplit[1][2:].split(",")]
        robots.append((ps, vs))
    count = 0
    while True:
        for robot in robots:
            robot[0][0] = (robot[0][0] + robot[1][0]) % width
            robot[0][1] = (robot[0][1] + robot[1][1]) % height
        count += 1
        # not a perfect heuristic, but it turns out the first time none of the robots overlap is the time they show the image
        if len(set(tuple(robot[0]) for robot in robots)) == len(robots):
            for robot in robots:
                grid[robot[0][1]][robot[0][0]] = "A"
            print('\n'.join([''.join(row) for row in grid]))
            break
    return count

print(day14())
