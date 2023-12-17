import heapq
import math
from collections import defaultdict
from functools import cache

def daytemp():
    f = open('input.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    grid = []
    for i in range(len(f)):
        row = f[i]
        
    return ans

def day1():
    f = open('aoc1.txt', 'r').read().strip().split("\n")
    ans = 0

    m = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    
    for i in range(len(f)):
        # Part 1
        # row = [x for x in f[i] if x in "0123456789"]
        # num = int(row[0] + row[-1])
        # ans += num

        row = f[i]
        first = None
        last = None
        for a in range(0, len(row)):
            if row[a] in "0123456789":
                first = int(row[a])
            else:
                for key in m:
                    if row[a:].startswith(key):
                        first = m[key]
            if first is not None:
                break
        for b in range(len(row)-1, -1, -1):
            if row[b] in "0123456789":
                last = int(row[b])
            else:
                for key in m:
                    if row[b:].startswith(key):
                        last = m[key]
            if last is not None:
                break
        
        ans += (10*first+last)
    return ans

def day2():
    f = open('aoc2.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    
    # a = {"red": 12, "green": 13, "blue": 14} # for part 1

    for i in range(len(f)):
        row = f[i]
        s = row.split(": ")
        game = int(s[0].split(" ")[-1])
        sets = s[1].split("; ")
        
        a = {"red": 0, "green": 0, "blue": 0} # for part 2
        
        # ok = True # for part 1
        for x in sets:
            ms = x.split(", ")
            for m in ms:
                msplit = m.split(" ")
                num = int(msplit[0])
                color = msplit[1]
                if num > a[color]:
                    # ok = False # for part 1
                    a[color] = num # for part 2
        # if ok: # for part 1
        #     ans += game
        ans += (a["red"]*a["blue"]*a["green"]) # for part 2

    return ans

def day3():
    f = open('aoc3.txt', 'r').read().strip().split("\n")
    ans = 0
    grid = []
    for i in range(len(f)):
        row = f[i]
        grid.append([x for x in row])
    
    def nbrs(i,j):
        dirs = [(-1,0), (-1,-1), (-1,1), (0,1), (0,-1), (1,1), (1,0), (1,-1)]
        ns = [(i+x[0], j+x[1]) for x in dirs if 0<=(i+x[0])<len(grid) and 0<=(j+x[1])<len(grid[0])]
        # return all(grid[n[0]][n[1]] in "1234567890." for n in ns) # for part 1
        return [n for n in ns if grid[n[0]][n[1]] == "*"]
    
    times = defaultdict(list)

    for i in range(len(f)):
        row = f[i]
        coords = []
        num = ""
        for j in range(len(row)):
            if row[j].isnumeric():
                num += row[j]
                coords.append((i,j))
                if (j == len(row)-1):
                    # if not all(nbrs(c[0], c[1]) for c in coords): # for part 1
                    #     ans += int(num)
                    ns = [nbrs(c[0], c[1]) for c in coords]
                    flat = {item for sublist in ns for item in sublist}
                    for fl in flat:
                        times[fl].append(int(num))

                    coords = []
                    num = ""

            else:
                if num != "":
                    # if not all(nbrs(c[0], c[1]) for c in coords): # for part 1
                    #     ans += int(num)
                    ns = [nbrs(c[0], c[1]) for c in coords]
                    flat = {item for sublist in ns for item in sublist}
                    for fl in flat:
                        times[fl].append(int(num))

                coords = []
                num = ""

    for t in times:
        if len(times[t]) == 2:
            ans += (times[t][0]*times[t][1])

    return ans

def day4():
    f = open('aoc4.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    grid = []

    for i in range(len(f)):
        row = f[i]
        grid.append(row)

        a = row.split(": ")
        nums = a[1].split(" | ")
        winning = nums[0].strip().split()
        haves = nums[1].strip().split()
        
        wins = 0
        for h in haves:
            if h in winning:
                wins += 1
        
        m[i] = wins
    #     if wins > 0: # for part 1
    #         ans += 2**(wins-1)
    # return ans

    counts = defaultdict(int)
    for i in range(len(grid)):
        counts[i] += 1

        if m[i] > 0:
            for j in range(i+1, i+m[i]+1):
                counts[j] += counts[i]

    return sum(counts.values())


def day5():
    f = open('aoc5.txt', 'r').read().strip().split("\n\n")
    
    def parse_section(index):
        return [[int(i) for i in row.split()] for row in f[index].split("\n")[1:]]

    seeds = f[0]
    seeds_split = [int(s) for s in seeds.split()[1:]]
    seed_to_soil = parse_section(1)
    soil_to_fertilizer = parse_section(2)
    fertilizer_to_water = parse_section(3)
    water_to_light = parse_section(4)
    light_to_temperature = parse_section(5)
    temperature_to_humidity = parse_section(6)
    humidity_to_location = parse_section(7)

    def find(seed, rows):
        for vals in rows:
            if vals[1] <= seed < vals[1]+vals[2]:
                return vals[0] + (seed-vals[1])        
        return seed
    
    def loc(seed):
        soil = find(seed, seed_to_soil)
        fertilizer = find(soil, soil_to_fertilizer)
        water = find(fertilizer, fertilizer_to_water)
        light = find(water, water_to_light)
        temperature = find(light, light_to_temperature)
        humidity = find(temperature, temperature_to_humidity)
        location = find(humidity, humidity_to_location)
        return location
    
    # for part 1
    # minscore = float("inf")
    # for seed in seeds_split:
    #     loca = loc(seed)
    #     if loca < minscore:
    #         minscore = loca
    # return minscore

    def backfind(seed, rows):
        for vals in rows:
            if vals[0] <= seed < vals[0]+vals[2]:
                return vals[1] + (seed-vals[0])
        return seed

    def backloc(location):
        humidity = backfind(location, humidity_to_location)
        temperature = backfind(humidity, temperature_to_humidity)
        light = backfind(temperature, light_to_temperature)
        water = backfind(light, water_to_light)
        fertilizer = backfind(water, fertilizer_to_water)
        soil = backfind(fertilizer, soil_to_fertilizer)
        seed = backfind(soil, seed_to_soil)
        return seed

    loc = 103000000 # manually "binary search" this value a bit
    ranges = []
    for i in range(0, len(seeds_split), 2):
        start = seeds_split[i]
        r = seeds_split[i+1]
        ranges.append((start, r))
    while True:
        seed = backloc(loc)
        for r in ranges:
            if r[0] <= seed < r[0]+r[1]:
                print(seed)
                return loc
        loc += 1
        if loc % 100000 == 0:
            print(loc) # keep track of progress   


def day6():
    f = open('aoc6.txt', 'r').read().strip().split("\n")
    ans = 1
    
    # times = [int(i) for i in f[0].split()[1:]] # for part 1
    # distances = [int(i) for i in f[1].split()[1:]]

    # manually entering was faster than trying to parse correctly
    totaltime = 45977295
    totaldistance = 305106211101695
    
    # pairs = [(times[i], distances[i]) for i in range(len(times))] # for part 1
    pairs = [(totaltime, totaldistance)]
    for pair in pairs:
        t = pair[0]
        d = pair[1]

        # this heuristic works okay, but a two-sided binary search would be better
        # on second thought we can just use the quadratic formula, solve j*(t-j)=d and find num of ints between the roots
        for j in range(t+1):
            if j*(t-j) > d:
                minimum = j
                break
        for j in range(t+1, -1, -1):
            if j*(t-j) > d:
                maximum = j
                break
        ans *= (maximum-minimum+1)
        
    return ans


def day7():
    f = open('aoc7.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {"five": [], "four": [], "full": [], "three": [], "twopair": [], "onepair": [], "high": []}
    bids = {}

    def score(cards):
        carddict = defaultdict(int)
        for c in cards:
            carddict[c] += 1
        
        jokervalue = carddict.pop("0", None)
        if jokervalue:
            maxkey = max(carddict, key=carddict.get, default="0")
            carddict[maxkey] += jokervalue

        v = sorted(list(carddict.values()))
        if v == [5]:
            return "five"
        elif v == [1,4]:
            return "four"
        elif v == [2,3]:
            return "full"
        elif v == [1,1,3]:
            return "three"
        elif v == [1,2,2]:
            return "twopair"
        elif v == [1,1,1,2]:
            return "onepair"
        else:
            return "high"

    for i in range(len(f)):
        row = f[i].split()
        # we replace the TJQKA to different values so that natively sorting the hands as strings works correctly
        # cards = row[0].replace("T","V").replace("J","W").replace("Q","X").replace("K","Y").replace("A","Z") # for part 1
        cards = row[0].replace("T","V").replace("J","0").replace("Q","X").replace("K","Y").replace("A","Z") # for part 2
        bids[cards] = int(row[1])
        k = score(cards)
        m[k].append(cards)
    
    sortedhands = []
    for key in ["five", "four", "full", "three", "twopair", "onepair", "high"]:
        allhands = m[key]
        sortedallhands = sorted(allhands, reverse=True)
        sortedhands.extend(sortedallhands)
    
    for i in range(len(sortedhands)):
        yourscore = (len(sortedhands)-i) * bids[sortedhands[i]]
        ans += yourscore

    return ans


def day8():
    f = open('aoc8.txt', 'r').read().strip().split("\n\n")
    m = {}
    lrs = f[0]
    rest = f[1].split("\n")
    for i in range(len(rest)):
        row = rest[i]
        rowsplit = row.split(" = ")
        start = rowsplit[0]
        paths = (rowsplit[1][1:4], rowsplit[1][6:9])  # ugly, but works
        m[start] = paths
    
    # location = "AAA"  # for part 1
    locations = [key for key in m if key.endswith('A')]
    startlocations = locations.copy()
    count = 0
    k = 0
    periods = {}
    while True:
        count += 1
        # part 1
        # if lrs[k] == 'L':
        #     location = m[location][0]
        # else:
        #     location = m[location][1]
        # if location == "ZZZ":
        #     break
        # k = (k+1) % len(lrs)
        
        destinations = []
        for i in range(len(locations)):
            location = locations[i]
            if lrs[k] == 'L':
                loc = m[location][0]
            else:
                loc = m[location][1]
            if loc.endswith('Z') and startlocations[i] not in periods:
                # so this appears to have been lucky... 
                # there is no guarantee this should be the period, but the input happens to be built with this nice property
                # otherwise, a proper solution would require finding an actual period and using the Chinese Remainder Theorem
                periods[startlocations[i]] = count
            destinations.append(loc)
        if len(periods) == len(locations):
            break
        locations = destinations
        k = (k+1) % len(lrs)
    
    # return count  # for part 1
    return math.lcm(*list(periods.values()))


def day9():
    f = open('aoc9.txt', 'r').read().strip().split("\n")
    ans = 0

    def extrapolate(vals):
        if all(v == 0 for v in vals):
            return 0
        else:
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            lastval = extrapolate(diffs)
            # return vals[-1] + lastval  # part 1
            return vals[0] - lastval  # part 2

    for row in f:
        ans += extrapolate([int(i) for i in row.split()])

    return ans


def day10():
    f = open('aoc10.txt', 'r').read().strip().split("\n")

    grid = []
    adj = []

    start = None
    for i in range(len(f)):
        row = f[i]
        grid.append([i for i in row])
        poss = []
        for j in range(len(row)):
            north = (i-1, j)
            south = (i+1, j)
            east = (i, j+1)
            west = (i, j-1)
            val = row[j]
            if val == 'S':
                start = (i, j)
                pos = []

            elif val == '|':
                pos = [north, south]
            elif val == '-':
                pos = [east, west]
            elif val == 'L':
                pos = [north, east]
            elif val == 'J':
                pos = [north, west]
            elif val == '7':
                pos = [south, west]
            elif val == 'F':
                pos = [south, east]
            else: # val == '.':
                pos = []
            poss.append(pos)
        adj.append(poss)
    
    nbrs = []
    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if start in adj[i][j]:
                nbrs.append((i,j))
    
    nbr1 = nbrs[0]  # pick one at random, to traverse the loop in that direction
    current = nbr1
    path = []
    path_set = set()
    adj[nbr1[0]][nbr1[1]].remove(start)
    while current != start:
        path.append(current)
        path_set.add(current)
        for val in adj[current[0]][current[1]]:
            if val not in path_set:  # find the neighbor that hasn't already been visited
                current = val
    # return (len(path)+1)/2  # for part 1

    def shoelace_theorem(path):  # computes the area of a polygon given its lattice point coordinates
        area = 0
        fullpath = [path[-1]] + path
        for i in range(len(fullpath)-1):
            one = fullpath[i]
            two = fullpath[i+1]
            area += (0.5*(one[0]+two[0])*(one[1]-two[1]))
        return abs(area)

    def picks_theorem(area, path):  # computes the number of interior lattice points of a polygon given its area and the number of boundary lattice points (equal to len(path)/2 here) 
        return area + 1 - (len(path)/2)

    area = shoelace_theorem(path)
    interior_points = picks_theorem(area, path)
    return interior_points


def day11():
    f = open('aoc11.txt', 'r').read().strip().split("\n")
    ans = 0
    
    expandrows = []
    for i in range(len(f)):
        row = f[i]
        if all(x == "." for x in row):
            expandrows.append(i)
    
    expandcols = []
    for j in range(len(f[0])):
        if all(row[j] == "." for row in f):
            expandcols.append(j)
    
    locations = []
    for row in range(len(f)):
        for col in range(len(f[0])):
            val = f[row][col]
            if val != ".":
                locations.append((row, col))
    
    def dist(pos1, pos2):
        extrarows = 0
        for row in expandrows:
            if min(pos1[0], pos2[0]) < row < max(pos1[0], pos2[0]):
                extrarows += 1
        extracols = 0
        for col in expandcols:
            if min(pos1[1], pos2[1]) < col < max(pos1[1], pos2[1]):
                extracols += 1
        
        # return abs(pos2[0]-pos1[0]) + abs(pos2[1]-pos1[1]) + (extrarows+extracols)  # part 1
        return abs(pos2[0]-pos1[0]) + abs(pos2[1]-pos1[1]) + (1000000-1)*(extrarows+extracols)  # part 2

    for pos1 in locations:
        for pos2 in locations:
            ans += dist(pos1, pos2)

    return int(ans/2)


def day12():
    f = open('aoc12.txt', 'r').read().strip().split("\n")
    ans = 0

    @cache
    def recurse(row, sizes):
        currentsize = 0
        allsizes = []  # sizes of groups we've seen thus far
        lastdotindex = 0

        for i in range(len(row)):
            if row[i] == ".":
                if currentsize != 0:
                    allsizes.append(currentsize)
                    # exit early if the sizes so far do not match
                    if tuple(allsizes) != sizes[:len(allsizes)]:
                        return 0
                    currentsize = 0

                lastdotindex = i  # the latest index we know of that is a ".", so we can take a substring of `row` starting at `lastdotindex` later, and only check for any sizes we haven't seen yet
            elif row[i] == "#":
                currentsize += 1
            elif row[i] == "?":
                # try either "#" or "." for every instance of "?"
                return recurse(row[lastdotindex:i] + "#" + row[i+1:], sizes[len(allsizes):]) + recurse(row[lastdotindex:i] + "." + row[i+1:], sizes[len(allsizes):])
        
        # if we reach here, there are no "?" marks in the row, so we can check if all sizes seen so far match the list of sizes given
        if currentsize != 0:  # edge case, if the last group is at the edge
            allsizes.append(currentsize)

        # we have 1 successful solution if all sizes match, otherwise we have 0
        return 1 if tuple(allsizes) == sizes else 0  

    for i in range(len(f)):
        rowsplit = f[i].split()
        rowvals = rowsplit[0]
        rowvals = rowvals + "?" + rowvals + "?" + rowvals + "?" + rowvals + "?" + rowvals  # for part 2
        # sizes = tuple(int(i) for i in rowsplit[1].split(","))  # for part 1
        sizes = tuple(int(i) for i in rowsplit[1].split(",")) * 5  # for part 2
        ans += recurse(rowvals, sizes)
    
    return ans


def day13():
    f = open('aoc13.txt', 'r').read().strip().split("\n\n")
    ans = 0

    def transpose(grid):
        rows = []
        for i in range(len(grid[0])):
            newrow = [row[i] for row in grid]
            rows.append(''.join(newrow))
        return rows
    
    def subfind(grid, original_val=None):
        for i in range(1, len(grid)//2+1):
            firstrows = grid[:i]
            lastrows = grid[i:2*i][::-1]
            if firstrows == lastrows:
                if i != original_val:
                    return i

            firstrows = grid[-2*i:-1*i]
            lastrows = grid[-1*i:][::-1]
            if firstrows == lastrows:
                if len(grid)-i != original_val:
                    return len(grid) - i

    def find(grid, original=None):
        original_row = None
        original_col = None
        if original:
            (direction, length) = original
            original_row = length if direction == "h" else None
            original_col = length if direction == "v" else None

        row_find = subfind(grid, original_row)
        if row_find:
            return ("h", row_find)
        
        newgrid = transpose(grid)
        col_find = subfind(newgrid, original_col)
        if col_find:
            return ("v", col_find)

        return None
    
    def bruteforce(grid):
        original = find(grid)
        for i in range(len(grid)):
            row = grid[i]
            for j in range(len(grid[0])):
                if grid[i][j] == "#":
                    grid[i] = grid[i][:j] + "." + grid[i][j+1:]
                else:
                    grid[i] = grid[i][:j] + "#" + grid[i][j+1:]
                tryans = find(grid, original)
                if tryans and tryans != original:
                    return tryans
                grid[i] = row

    for i in range(len(f)):
        grid = f[i].split("\n")
        # direction, length = find(grid)  # part 1
        direction, length = bruteforce(grid)  # part 2
        if direction == "h":
            ans += (100*length)
        else:
            ans += length
        
    return ans


def day14():
    f = open('aoc14.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    grid = []
    for row in f:
        grid.append(list(row))
    
    # part 1
    # for _ in range(len(grid)):
    #     for i in range(1, len(grid)):
    #         for j in range(len(grid[0])):
    #             if grid[i][j] == "O" and grid[i-1][j] == '.':
    #                 grid[i][j] = '.'
    #                 grid[i-1][j] = "O"
    # for i in range(len(grid)):
    #     ans += grid[i].count("O")*(len(grid)-i)
    # return ans

    allgrids = set()
    cycle = 1
    while True:
        # north
        for _ in range(len(grid)):
            for i in range(1, len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == "O" and grid[i-1][j] == '.':
                        grid[i][j] = '.'
                        grid[i-1][j] = "O"
        
        # west
        for _ in range(len(grid)):
            for i in range(len(grid)):
                for j in range(1,len(grid[0])):
                    if grid[i][j] == "O" and grid[i][j-1] == '.':
                        grid[i][j] = '.'
                        grid[i][j-1] = "O"

        # south
        for _ in range(len(grid)):
            for i in range(len(grid)-1):
                for j in range(len(grid[0])):
                    if grid[i][j] == "O" and grid[i+1][j] == '.':
                        grid[i][j] = '.'
                        grid[i+1][j] = "O"

        # east
        for _ in range(len(grid)):
            for i in range(len(grid)):
                for j in range(len(grid[0])-1):
                    if grid[i][j] == "O" and grid[i][j+1] == '.':
                        grid[i][j] = '.'
                        grid[i][j+1] = "O"
        
        strgrid = ''.join([''.join(row) for row in grid])
        # use this to compute the period; on this particular input, cycles 83 and 155 are the same, so the period is 155-83 = 72
        # if strgrid in allgrids:
        #     print(m[strgrid])
        #     print(cycle)
        #     return
        m[strgrid] = cycle
        allgrids.add(strgrid)

        # now, take (1000000000-83) mod 72, which is 53; we now want the grid at cycle 83+53
        if cycle == 83+53:
            for i in range(len(grid)):
                ans += grid[i].count("O")*(len(grid)-i)
            return ans

        cycle += 1


def day15():
    f = open('aoc15.txt', 'r').read().strip().split(",")
    ans = 0
    m = {}
    for word in f:
        # comment out this block for part 1
        if word[-1] == "-":
            word = word[:-1]
        else:
            word = word.split("=")[0]
    
        current = 0
        for c in word:
            current += ord(c)
            current *= 17
            current %= 256
        m[word] = current

        # ans += current  # part 1
    # return ans  # part 1

    boxes = {i: [] for i in range(256)}
    for word in f:
        if word[-1] == "-":
            prefix = word[:-1]
            boxindex = m[prefix]
            if prefix in [b[0] for b in boxes[boxindex]]:
                for b in boxes[boxindex]:
                    if b[0] == prefix:
                        boxes[boxindex].remove(b)
                        break
        else:
            prefix, focallength = word.split("=")
            boxindex = m[prefix]
            focallength = int(focallength)
            if prefix in [b[0] for b in boxes[boxindex]]:
                for i in range(len(boxes[boxindex])):
                    b = boxes[boxindex][i]
                    if b[0] == prefix:
                        boxes[boxindex][i] = (prefix, focallength)
                        break
            else:
                boxes[boxindex].append((prefix, focallength))

    for boxnumber in boxes:
        for slot in range(len(boxes[boxnumber])):
            lens = boxes[boxnumber][slot]
            ans += (boxnumber+1)*(slot+1)*lens[1]
        
    return ans


def day16():
    grid = open('aoc16.txt', 'r').read().strip().split("\n")
    
    EAST = (0,1)
    WEST = (0,-1)
    NORTH = (-1,0)
    SOUTH = (1,0)
    
    mapping = {
        ".": {EAST: [EAST], WEST: [WEST], NORTH: [NORTH], SOUTH: [SOUTH]},
        "-": {EAST: [EAST], WEST: [WEST], NORTH: [WEST, EAST], SOUTH: [WEST, EAST]},
        "|": {NORTH: [NORTH], SOUTH: [SOUTH], WEST: [NORTH, SOUTH], EAST: [NORTH, SOUTH]},
        "/": {NORTH: [EAST], SOUTH: [WEST], EAST: [NORTH], WEST: [SOUTH]},
        "\\": {NORTH: [WEST], SOUTH: [EAST], EAST: [SOUTH], WEST: [NORTH]},
    }

    def is_in(pos):
        row, col = pos
        return 0<=row<len(grid) and 0<=col<len(grid[0])
    
    def try_start(start):
        # tuples are (position, direction)
        # position is (row, col), direction is (row_direction, col_direction)
        litpos = {start}
        lit = set()

        def go(pos, dir):
            nextpos = (pos[0]+dir[0], pos[1]+dir[1])
            if is_in(nextpos):
                litpos.add((nextpos, dir))
        
        while litpos:
            postuple = litpos.pop()
            if postuple in lit:
                continue
            lit.add(postuple)

            pos, dir = postuple
            val = grid[pos[0]][pos[1]]
            m = mapping[val]
            for nextdir in m[dir]:
                go(pos, nextdir)

        return len(set([l[0] for l in lit]))

    # return try_start(((0,0),EAST))  # part 1

    maxlit = 0
    row = 0    
    for col in range(len(grid[0])):  # top edge
        dir = SOUTH
        lit = try_start(((row, col), dir))
        if lit > maxlit:
            maxlit = lit
    row = len(grid)-1
    for col in range(len(grid[0])):  # bottom edge
        dir = NORTH
        lit = try_start(((row, col), dir))
        if lit > maxlit:
            maxlit = lit
    col = 0
    for row in range(len(grid)):  # left edge
        dir = EAST
        lit = try_start(((row, col), dir))
        if lit > maxlit:
            maxlit = lit
    col = len(grid[0])-1
    for row in range(len(grid)):  # right edge
        dir = WEST
        lit = try_start(((row, col), dir))
        if lit > maxlit:
            maxlit = lit

    return maxlit


def day17():
    f = open('aoc17.txt', 'r').read().strip().split("\n")
    grid = []
    for row in f:
        rowints = [int(j) for j in list(row)]
        grid.append(rowints)

    EAST = (0,1)
    WEST = (0,-1)
    NORTH = (-1,0)
    SOUTH = (1,0)
    all_dirs = [NORTH, EAST, SOUTH, WEST]
    opp = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}

    def is_in(pos):
        row, col = pos
        return 0<=row<len(grid) and 0<=col<len(grid[0])

    dists = {(row, col): float("inf") for row in range(len(grid)) for col in range(len(grid[0]))}
    def dijkstra():
        done = set()
        queue = [(0, (0,0), 0, (0,0))]
        heapq.heapify(queue)

        while queue:
            n = heapq.heappop(queue)
            dist, pos, streak, dir = n
            if (pos, streak, dir) in done:
                continue
            # dists[pos] = min(dists[pos], dist)  # part 1 only
            if streak >= 4:  # part 2 only
                dists[pos] = min(dists[pos], dist)  # part 2 only
            done.add((pos, streak, dir))
            row, col = pos

            # forbidden = dir if streak == 3 else None  # part 1
            forbidden = dir if streak == 10 else None  # part 2

            validdirs = [i for i in all_dirs if i != forbidden and (streak == 0 or opp[dir] != i)]
            if 1 <= streak <= 3:  # part 2
                validdirs = [dir]  # part 2
            for newdir in validdirs:
                newrow, newcol = row+newdir[0], col+newdir[1]
                newpos = (newrow, newcol)
                if is_in(newpos):
                    val = grid[newrow][newcol]
                    newstreak = 1 if newdir != dir else streak+1
                    heapq.heappush(queue, (dist+val, newpos, newstreak, newdir))

    dijkstra()
    return dists[(len(grid)-1,len(grid[0])-1)]

print(day17())