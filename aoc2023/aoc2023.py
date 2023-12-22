import heapq
import math
from collections import defaultdict
from functools import cache, reduce

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

    def picks_theorem(area, path):  # computes the number of interior lattice points of a polygon given its area and the number of boundary lattice points (equal to len(path) here) 
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


def day18():
    f = open('aoc18.txt', 'r').read().strip().split("\n")
    m = {"R": (0,1), "L": (0,-1), "U": (-1,0), "D": (1,0)}
    path = []
    pos = (0,0)

    d = {"0": "R", "1": "D", "2": "L", "3": "U"}
    pathlen = 0

    for i in range(len(f)):
        row = f[i].split()
        # dir = row[0]  # part 1
        # dist = int(row[1])  # part 1
        
        hex = row[2][2:-1]  # part 2
        dist = int(hex[:5], 16)  # part 2
        dir = d[hex[-1]]  # part 2

        move = (m[dir][0]*dist, m[dir][1]*dist)
        pathlen += dist
        pos = (pos[0] + move[0], pos[1] + move[1])
        path.append(pos)
    
    def shoelace_theorem(path):  # computes the area of a polygon given its lattice point coordinates
        area = 0
        fullpath = [path[-1]] + path
        for i in range(len(fullpath)-1):
            one = fullpath[i]
            two = fullpath[i+1]
            area += (0.5*(one[0]+two[0])*(one[1]-two[1]))
        return abs(area)

    def picks_theorem(area, boundary_pts):  # computes the number of interior lattice points of a polygon given its area and the number of boundary lattice points 
        return area + 1 - (boundary_pts)/2

    area = shoelace_theorem(path)
    interior_points = picks_theorem(area, pathlen)
    return interior_points + pathlen


def day19():
    f = open('aoc19.txt', 'r').read().strip().split("\n\n")
    ans = 0
    m = {}
    
    workflows = f[0].strip().split("\n")
    parts = f[1].strip().split("\n")

    for w in workflows:
        wsplit = w[:-1].split("{")
        wname = wsplit[0]
        options = wsplit[1].split(",")
        m[wname] = options
    
    def traverse(part, pos):
        if pos in {"A", "R"}:
            return pos
        # part has keys {x:, m:, a:, s:}
        options = m[pos]
        for o in options[:-1]:
            condition, result = o.split(":")
            attribute = condition[0]
            sign = condition[1]
            num = int(condition[2:])
            attributevalue = part[attribute]
            if sign == "<" and attributevalue < num:
                return traverse(part, result)
            elif sign == ">" and attributevalue > num:
                return traverse(part, result)
        return traverse(part, options[-1])
    
    # Part 1
    # for part in parts:
    #     vals = part[1:-1].split(",")
    #     partmap = {}
    #     for v in vals:  # construct mapping of "x", "m", "a", "s" to values
    #         partmap[v[0]] = int(v[2:])
    #     r = traverse(partmap, "in")
    #     if r == "A":
    #         ans += sum(partmap.values())
    # return ans
    
    # these are open intervals for each attribute
    possible = {"x": (0, 4001), "m": (0, 4001), "a": (0, 4001), "s": (0, 4001)}
    def traverse2(pos, possiblemap):
        total = 0
        options = m[pos]

        for o in options[:-1]:
            condition, result = o.split(":")
            attribute = condition[0]
            sign = condition[1]
            num = int(condition[2:])
            minval, maxval = possiblemap[attribute]
            if sign == "<":
                newrange = (min(num, minval), min(num, maxval))
                comp = (max(num-1, minval), max(num-1, maxval))  # tricky off-by-1, since the complement is >=
            elif sign == ">":
                newrange = (max(num, minval), max(num, maxval))
                comp = (min(num+1, minval), min(num+1, maxval))  # tricky off-by-1, since the complement is <=
            
            if newrange[0] != newrange[1]:
                possiblemap[attribute] = newrange
                if result == "A":
                    total += reduce(lambda x, y: x*y, ((x[1]-x[0]-1) for x in possiblemap.values()))
                elif result != "R":  # if result == "R", we ignore it
                    total += traverse2(result, possiblemap.copy())
            
            possiblemap[attribute] = comp  # move to the next condition, so change this attribute to the complement
        
        last = options[-1]
        if last == "A":
            total += reduce(lambda x, y: x*y, [max(0, x[1]-x[0]-1) for x in possiblemap.values()])
        elif last != "R":
            total += traverse2(last, possiblemap.copy())
        
        return total

    return traverse2("in", possible)


def day20():
    f = open('aoc20.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    source_to_dest = defaultdict(list)
    modtype = defaultdict(str)
    for row in f:
        left, right = row.split(" -> ")
        rights = right.split(", ")
        if left == "broadcaster":
            name = left
        else:
            symbol = left[0]
            name = left[1:]
            modtype[name] = symbol
            if symbol == "%":
                m[name] = 0
            else:
                m[name] = {}
        source_to_dest[name] = rights

    for row in f:
        left, right = row.split(" -> ")
        rights = right.split(", ")
        if left == "broadcaster":
            name = left
        else:
            name = left[1:]
        for r in rights:
            if r in m and not isinstance(m[r], int):
                m[r][name] = 0

    pulse_counts = {0: 0, 1: 0}
    def receive_pulse(source, destination, value):
        if modtype[destination] == "%":
            if value == 1:
                return []
            else:
                m[destination] = int(not (m[destination]))
                sendvalue = m[destination]
        elif modtype[destination] == "&":
            m[destination][source] = value
            sendvalue = int(not all(v == 1 for v in m[destination].values()))
        else:
            return []
        pulse_counts[sendvalue] += len(source_to_dest[destination])
        return [(destination, nextdest, sendvalue) for nextdest in source_to_dest[destination]]

    '''
    it gets fuzzy here... we scan the input by hand and discover that the incoming node to &rx is only &cs
    and then the only incoming nodes to &cs are &kh, &lz, &tg, &hn
    and each of these only depends on % nodes, so we try to find the period for kh, lz, tg, hn and take the lcm
    '''
    nodes_to_rx = ["kh", "lz", "tg", "hn"]
    tracking = defaultdict(list)
    count = 0
    while True:
        count += 1
        pulse_counts[0] += len(source_to_dest["broadcaster"]) + 1  # low pulses sent from broadcaster, plus one low pulse from the first button
        queue = [("destination", dest, 0) for dest in source_to_dest["broadcaster"]]
        while queue:
            nextqueue = []
            for q in queue:
                if q[0] in nodes_to_rx and q[2] == 1:
                    tracking[q[0]].append(count)
                next_pulses = receive_pulse(q[0], q[1], q[2])
                nextqueue.extend(next_pulses)
            queue = nextqueue
        
        # part 1
        # if count == 1000: 
        #     break
        
        # part 2
        if count == 20000:  
            # empirically, we observe that all periods luckily are independent from one another!
            print(tracking)  
            break

    # return pulse_counts[0] * pulse_counts[1]  # part 1
    periods = [tracking[key][0] for key in nodes_to_rx]
    return math.lcm(*periods)


def day21():
    grid = open('aoc21.txt', 'r').read().strip().split("\n")
    for i in range(len(grid)):
        row = grid[i]
        for j in range(len(row)):
            if row[j] == "S":
                start = (i, j)
    
    def nbrs(pos):
        row, col = pos
        ns = set()
        moves = [(1,0), (-1,0), (0,1), (0,-1)]
        for m in moves:
            newrow, newcol = row+m[0], col+m[1]
            if 0 <= newrow < len(grid) and 0 <= newcol < len(grid[0]) and grid[newrow][newcol] != "#":
                ns.add((newrow, newcol))
        return ns

    def take_steps(start_loc, num_steps):
        visited = set()
        even_steps = {start_loc}
        odd_steps = set()

        locs = {start_loc}
        for i in range(num_steps):
            newset = set()
            for l in locs:
                newset |= nbrs(l)
            locs = {n for n in newset if n not in visited}
            visited |= locs
            if i%2 == 0:
                odd_steps |= locs
            else:
                even_steps |= locs
            if not locs:  # early return
                break
        return odd_steps, even_steps

    # part 1
    # _, even_steps = take_steps(start, 64)
    # return len(even_steps) 

    '''
    so for part 2... the given grid is 131x131, and when we inspect the input, we notice that all borders,
    and all locations in the same row/column as 'S' are all garden plots, so we can go from the same "position" in adjacent grids
    always in exactly 131 steps; this is convenient because the given number of steps, 26501365, is 202300*(131)+65
    this was by design, because it takes 65 steps to reach the midpoints of each edge in the starting 131x131 "grid"
    so now we need to break up the "infinite" grid and think about all of the possible individual "grids" that are reachable
    things are simplified a bit here because of parity -- no spot reachable in an odd number of steps can ever be reached in an even number of steps
    '''
    # len(grid) = 131
    SIZE = len(grid)-1
    RADIUS = 26501365 // len(grid)  # 202300
    dirsets = {}

    NORTH = (0, SIZE//2)  # midpoint of top edge
    SOUTH = (SIZE, SIZE//2)  # midpoint of bottom edge
    WEST = (SIZE//2, 0)  # midpoint of left edge
    EAST = (SIZE//2, SIZE)  # midpoint of right edge

    # starting from each of these midpoints on a grid, how many locations in an individual grid can be reached?
    for dir in [NORTH, SOUTH, WEST, EAST]:
        odd_steps, even_steps = take_steps(dir, SIZE)
        dirsets[dir] = even_steps  # since SIZE is even
    
    # these are all corners; starting from these corners, how many locations in an individual grid can be reached?
    NORTHEAST = (0, SIZE)
    NORTHWEST = (0, 0)
    SOUTHEAST = (SIZE, SIZE)
    SOUTHWEST = (SIZE, 0)
    for dir in [NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]:
        # when you draw it out, for the outermost grids that we enter from a corner, when you enter it, you only have 64 steps remaining, because it took 67 steps to reach it from an edge midpoint
        odd_steps, even_steps = take_steps(dir, SIZE//2-1)
        dirsets[dir] = even_steps  # since SIZE//2-1 is even
    
    # changing the keys to be more readable
    namemap = {NORTH: "north", SOUTH: "south", EAST: "east", WEST: "west", NORTHWEST: "northwest", NORTHEAST: "northeast", SOUTHWEST: "southwest", SOUTHEAST: "southeast"}
    a = {}
    for dir in [NORTH, SOUTH, WEST, EAST, NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST]:
        a[namemap[dir]] = len(dirsets[dir])
    a["north and west"] = len(dirsets[NORTH] | dirsets[WEST])
    a["south and west"] = len(dirsets[SOUTH] | dirsets[WEST])
    a["north and east"] = len(dirsets[NORTH] | dirsets[EAST])
    a["south and east"] = len(dirsets[SOUTH] | dirsets[EAST])
    
    CENTER = (SIZE//2, SIZE//2)
    odd_steps, even_steps = take_steps(dir, 10000)  # pick a large enough number of iterations, the function will early return
    a["even"] = len(even_steps)  # all locations with the same parity as the starting location, i.e. all locations reachable in an even number of moves from S
    a["odd"] = len(odd_steps)  # all locations with the opposite parity as the starting location, i.e. all locations reachable in an odd number of moves from S

    print(a)
    
    # this summation is almost impossible to explain without a proper diagram...

    # first term: grids with the opposite parity of the center grid
    opposite_parity = RADIUS**2*a["even"]
    # grids with the same parity as the center grid
    same_parity = (RADIUS-1)**2*a["odd"] \
    # grids at the very "tips" of the "diamond" of all reachable grids, i.e. if you only go in one direction the whole time 
    ends = (a["north"] + a["south"] + a["east"] + a["west"])
    # grids on the edge of the "diamond" that are reachable from two directions
    two_directions = (a["north and east"] + a["north and west"] + a["south and east"] + a["south and west"]) * (RADIUS-1)
    # grids on the edge of the diamond that can only be reached by entering from one of its corners
    from_corner = (a["northwest"] + a["northeast"] + a["southwest"] + a["southeast"]) * RADIUS

    # a quick note... so this summation ends up being quadratic in RADIUS, since all values in the dictionary named "a" are constants...
    # so it seems like some people just calculated what this would be for R = 0, 1, and 2, and solved for the coefficients...
    # then plugged R = 202300 into the quadratic... very clever
    return opposite_parity + same_parity + ends + two_directions + from_corner


def day22():
    f = open('aoc22.txt', 'r').read().strip().split("\n")
    ans = 0

    all_blocks = []
    for i in range(len(f)):
        row = f[i]
        left, right = row.split("~")
        ltuple = tuple(int(i) for i in left.split(","))
        rtuple = tuple(int(i) for i in right.split(","))
        all_blocks.append((ltuple, rtuple))
    all_blocks.sort(key=lambda x: x[0][2])  # sort by z coordinate

    def xy_block_coords(block, pos):
        left, right = block
        if pos == "top":
            zcoord = left[2]  # if a "top" block, give all the bottom-most xy coordinates
        elif pos == "bottom":
            zcoord = right[2]  # if a "bottom" block, give all the top-most xy coordinates
        return [(i,j,zcoord) for i in range(left[0], right[0]+1) for j in range(left[1], right[1]+1)]

    def is_atop(top, bottom):
        lefttop, righttop = top
        top_xy = xy_block_coords(top, "top")
        bottom_xy = xy_block_coords(bottom, "bottom")
        for top_coord in top_xy:
            tx, ty, tz = top_coord
            for bottom_coord in bottom_xy:
                bx, by, bz = bottom_coord
                if bz < tz and bx == tx and by == ty:
                    move = tz-bz-1
                    movedblock = ((lefttop[0], lefttop[1], lefttop[2]-move), (righttop[0], righttop[1], righttop[2]-move))
                    return movedblock
        return None
    
    stack = []
    for block in all_blocks:
        appended = False
        finalmovedblock = None
        for stackblockindex in range(len(stack)):
            stackblock = stack[stackblockindex]
            movedblock = is_atop(block, stackblock)
            if movedblock is not None:
                # need to find the block that this block stacks the "highest" on top of
                if finalmovedblock is None or movedblock[0][2] > finalmovedblock[0][2]: 
                    finalmovedblock = movedblock
        if finalmovedblock:
            stack.append(finalmovedblock)
        else:  # no other block lies below it, so we need to put the block on the ground
            lefttop, righttop = block 
            move = lefttop[2]-1
            finalmovedblock = ((lefttop[0], lefttop[1], lefttop[2]-move), (righttop[0], righttop[1], righttop[2]-move))
            stack.append(finalmovedblock)
    
    def directly_atop(top, bottom):
        lefttop, righttop = top
        top_xy = xy_block_coords(top, "top")
        bottom_xy = xy_block_coords(bottom, "bottom")
        for top_coord in top_xy:
            tx, ty, tz = top_coord
            for bottom_coord in bottom_xy:
                bx, by, bz = bottom_coord
                if bz == (tz-1) and bx == tx and by == ty:  # z-coordinates differ by exactly 1
                    return True
        return False

    lies_below = defaultdict(list)
    lies_atop = defaultdict(list)
    for top in stack:
        for bottom in stack:
            if directly_atop(top, bottom):
                lies_atop[top].append(bottom)
                lies_below[bottom].append(top)
    
    # part 1
    # not_allowed = set()
    # for block in stack:
    #     if len(lies_atop[block]) == 1:
    #         not_allowed.add(lies_atop[block][0])
    # return len(stack) - len(not_allowed)
    
    for block in stack:
        removed = set()
        nextremove = set([block])
        while nextremove:
            nextnextremove = set()
            for n in nextremove:
                removed.add(n)
                for b in lies_below[n]:
                    if all(x in removed for x in lies_atop[b]):
                        nextnextremove.add(b)
            nextremove = nextnextremove
        ans += len(removed)-1  # -1 to ignore the disintegrated block
    return ans
    
print(day22())
