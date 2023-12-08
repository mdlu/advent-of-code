import math
from collections import defaultdict

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

print(day8())