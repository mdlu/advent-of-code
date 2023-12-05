def findDoublet(nums, total):
    for num in nums:
        if total-num in nums:
            return num, total-num
    return None

def findTriplet():
    numbers = open('aoc1.txt', 'r').read().split("\n")[:-1]
    numberSet = set([int(num) for num in numbers])
    for number in numberSet:
        doublet = findDoublet(numberSet, 2020-number)
        if doublet:
            print(number, doublet[0], doublet[1], number*doublet[0]*doublet[1])
            break

def day2():
    f = open('aoc2.txt', 'r').read().split("\n")
    count = 0
    for line in f[:-1]:
        split = line.split(" ")
        ranges = split[0].split('-')
        letter = split[1][0]
        password = split[-1]

        if (password[int(ranges[0])-1] == letter) ^ (password[int(ranges[1])-1] == letter):
            count += 1
        # if int(ranges[0]) <= password.count(letter) <= int(ranges[1]):
        #     count += 1
    print(count)


def day13():
    time = 1000303
    bs = "41,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,37,x,x,x,x,x,541,x,x,x,x,x,x,x,23,x,x,x,x,13,x,x,x,17,x,x,x,x,x,x,x,x,x,x,x,29,x,983,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,19"

    allbuses = list(enumerate(bs.split(",")))
    buses = [(bus[0], int(bus[1])) for bus in allbuses if bus[1] != 'x']

    nums = [bus[1] for bus in buses]
    rems = [bus[0] for bus in buses]
    k = len(buses)
    print(buses)
    minx = findMinX(nums, rems, k) # crt
    # for bus in buses:
    #     print(bus)
    #     print(minx % bus[1])
    return minx
    # # largest = max(buses, key=lambda x: x[1])

    # i = 1
    # while True:
    #     num = largest[1]*i - largest[0]
    #     # found = False
    #     count = 0
    #     for bus in buses:
    #         if (num-bus[0]) % bus[1] != 0:
    #             break
    #         count += 1

    #     if count == len(buses):
    #         break 
    #     i += 1
    #     # if found:
    #     #     break
    # return num

    # gap = float("inf")
    # bestbus = None
    # for bus in buses:
    #     bustime = (time//bus + 1)*bus
        
    #     if bustime - time < gap:
    #         gap = bustime-time
    #         bestbus = bus
    # return bestbus * gap

def day14():
    f = open('aoc14.txt', 'r').read().split("mask = ")[1:]
    # f = """mask = 000000000000000000000000000000X1001X
    # mem[42] = 100
    # mask = 00000000000000000000000000000000X0XX
    # mem[26] = 1
    # """.split("mask = ")[1:]
    mem = {}
    for item in f:
        elements = item.split("\n")[:-1]
        bitmask = elements[0]
        valueloc = []
        for locs in elements[1:]:
            left = locs.index("[")
            right = locs.index("]")
            equal = locs.index("=")
            location = int(locs[left+1:right])
            value = int(locs[equal+2:])
            valueloc.append((value, location))
        
        # mask1 = ""
        # mask2 = ""
        # for b in bitmask:
        #     if b == 'X':
        #         mask1 += '0'
        #         mask2 += '1'
        #     else:
        #         mask1 += b
        #         mask2 += b

        # mask1 = int(mask1, 2)
        # mask2 = int(mask2, 2)
        
        # for pair in valueloc:
        #     newvalue = pair[0] | mask1 
        #     newvalue = newvalue & mask2
        #     mem[pair[1]] = newvalue
        
        mask1 = "" # for and
        mask2 = "" # for or
        for b in bitmask:
            if b == 'X':
                mask1 += '0'
                mask2 += '0'
            else:
                mask1 += '1'
                mask2 += b

        mask1 = int(mask1, 2)
        mask2 = int(mask2, 2)

        for pair in valueloc:
            value = pair[0]
            loc = pair[1]

            baseloc = (loc & mask1) | mask2
            vals = []
            for i in range(len(bitmask)):
                if bitmask[-(i+1)] == 'X':
                    vals.append(2**i)
            
            combos = getCombos(vals)
            for combo in combos:
                newloc = baseloc + combo
                mem[newloc] = value

    return sum(mem.values())

def getCombos(vals, hold = [0]):
    if len(vals) == 0:
        return hold
    else:
        newhold = hold + [h + vals[0] for h in hold]
        return getCombos(vals[1:], newhold)

def encounterTrees(lines, right, down):
    length = len(lines[0])
    counter = 0
    trees = 0
    newlines = [lines[i] for i in range(len(lines)) if i%down == 0]
    for line in newlines:
        if line[counter] == '#':
            trees += 1
        counter = (counter+right)% length
    return trees

def day3():
    f = open('aoc3.txt', 'r').read().split("\n")[:-1]
    length = len(f[0])
    counter = 0
    tree1 = encounterTrees(f, 1, 1)
    tree2 = encounterTrees(f, 3, 1)
    tree3 = encounterTrees(f, 5, 1)
    tree4 = encounterTrees(f, 7, 1)
    tree5 = encounterTrees(f, 1, 2)

    return tree1*tree2*tree3*tree4*tree5

def day15():
    mem = {2:0, 20:1, 0:2, 4:3, 1:4}
    lastval = 17
    for i in range(6, 30000000):
        newval = 0 if lastval not in mem else (i-1) - mem[lastval]
        mem[lastval] = i-1
        lastval = newval
    return lastval

def day16():
    f = open('aoc16.txt', 'r').read().split("\n\nyour ticket:\n")
    positions = f[0]
    locations = {}
    for row in positions.split("\n"):
        leftright = row.split(": ")
        key = leftright[0]
        ranges = []
        for value in leftright[1].split(" or "):
            split = value.split("-")
            ranges.append((int(split[0]), int(split[1])))
        locations[key] = tuple(ranges)

    g = f[1].split("\n\nnearby tickets:\n")
    yours = [int(num) for num in g[0].split(",")]
    tickets = g[1].split("\n")[:-1]
    allnumbers = [[int(num) for num in ticket.split(",")] for ticket in tickets]
    total = 0
    valids = []
    for numbers in allnumbers:
        invalid = False
        for num in numbers:
            if all([num < r[0] or num>r[1] for r in ranges for ranges in locations.values()]):
                total += num
                invalid = True
        if not invalid:
            valids.append(numbers)
    
    result = valid_vals(locations, valids)
    sortresult = sorted(list(enumerate(result)), key=lambda x: len(x[1]))
    answers = {sortresult[0][1].pop(): sortresult[0][0]}
    for i in range(1, len(sortresult)):
        answers[(sortresult[i][1] - sortresult[i-1][1]).pop()] = sortresult[i][0]
    
    product = 1
    for key in answers.keys():
        if key[:9] == 'departure':
            product *= yours[answers[key]]
    return product
    
def valid_vals(locations, tickets):
    valids = []
    for i in range(len(tickets[0])):
        keys = []
        ith = [ticket[i] for ticket in tickets]
        for key in locations.keys():
            r = locations[key]
            if all([(r[0][0] <= val <= r[0][1] or r[1][0] <= val <= r[1][1]) for val in ith]):
                keys.append(key)
        valids.append(set(keys))
    return valids

def day17():
    states = set()
    start = """....#...
.#..###.
.#.#.###
.#....#.
...#.#.#
#.......
##....#.
.##..#.#""".split("\n")
    for x in range(len(start)):
        for y in range(len(start[0])):
            if start[x][y] == '#':
                states.add((x,y,0,0))
    
    for it in range(6):
        minx = min(states, key=lambda x:x[0])[0]-1
        maxx = max(states, key=lambda x:x[0])[0]+2
        miny = min(states, key=lambda x:x[1])[1]-1
        maxy = max(states, key=lambda x:x[1])[1]+2
        minz = min(states, key=lambda x:x[2])[2]-1
        maxz = max(states, key=lambda x:x[2])[2]+2
        minw = min(states, key=lambda x:x[3])[3]-1
        maxw = max(states, key=lambda x:x[3])[3]+2

        newstates = states.copy()
        for x in range(minx, maxx):
            for y in range(miny, maxy):
                for z in range(minz, maxz):
                    for w in range(minw, maxw):
                        count = checkneighbors(x,y,z,w,states)
                        if (x,y,z,w) in states:
                            if not (count == 2 or count == 3):
                                newstates.remove((x,y,z,w))
                        elif count == 3:
                            newstates.add((x,y,z,w))
        
        states = newstates
    
    return len(states)
    
def checkneighbors(x, y, z, w, states):
    ans = set()
    for a in [x-1, x, x+1]:
        for b in [y-1,y,y+1]:
            for c in [z-1,z,z+1]:
                for d in [w-1,w,w+1]:
                    ans.add((a,b,c,d))
    ans.remove((x,y,z,w))

    return len(ans.intersection(states))

def day5():
    f = open('aoc5.txt', 'r').read().strip().split("\n")
    ids = set()
    for seat in f:
        ids.add(seatID(seat))
    # return max(ids)
    for i in ids:
        if i+1 not in ids:
            return i+1

def seatID(seat):
    row = seat[:-3]
    col = seat[-3:]
    rownum = 0
    colnum = 0
    for i in range(len(row)):
        if row[-(i+1)] == 'B':
            rownum += (2**i)
    for j in range(len(col)):
        if col[-(j+1)] == 'R':
            colnum += (2**j)
    return rownum*8+colnum

def day6():
    f = open('aoc6.txt', 'r').read().strip().split("\n\n")
    total = 0
    for group in f:
        # qs = len(set(''.join(group.split("\n"))))
        # total += qs
        qs = group.split("\n")
        init = set(qs[0])
        for q in qs:
            init = init.intersection(set(q))
        total += len(init)
    return total
    
def day9():
    f = open('aoc9.txt', 'r').read().strip().split("\n")
    # queue = []
    # for i in range(25):
    #     queue.append(int(f[i]))
    
    # j = 25
    # while True:
    #     s = set(queue)
    #     val = int(f[j])
    #     found = False
    #     for q in queue:
    #         if val-q in s:
    #             found = True
    #             break
    #     if not found:
    #         return val
    #     queue.pop(0)
    #     queue.append(val)
    #     j += 1

    answer = 2089807806
    sums = {(i,i): int(f[i]) for i in range(len(f))}
    for i in range(len(f)):
        for j in range(i+1, len(f)):
            sums[(i,j)] = sums[(i,j-1)] + int(f[j])
    for key in sums:
        if sums[key] == answer and key[0] != key[1]:
            interval = [int(x) for x in f[key[0]:key[1]+1]]
            print(key)
            print(interval)
            print(sum(interval))
            return min(interval) + max(interval)

def day10():
    f = open('aoc10.txt', 'r').read().strip().split("\n")
    ints = sorted([int(i) for i in f])
    ints = [0] + ints + [max(ints) + 3] # added for part 2
    # ones = 0
    # threes = 0
    # for i in range(1, len(ints)):
    #     diff = ints[i] - ints[i-1]
    #     if diff == 1:
    #         ones += 1
    #     elif diff == 3:
    #         threes += 1
    # return ones*threes
    totals = {}
    totals[-1] = 1
    for i in range(2, len(ints)+1):
        ans = totals[-i+1]
        if (-i+2) < 0:
            if ints[-i+2] - ints[-i] <= 3:
                ans += totals[-i+2]
        if (-i+3) < 0:
            if ints[-i+3] - ints[-i] <= 3:
                ans += totals[-i+3]
        totals[-i] = ans
    return totals[-len(ints)]


def day11():
    f = open('aoc11.txt', 'r').read().strip().split("\n")
    # for i in range(len(f)):
    #     f[i] = list(f[i])
    while True:
        change = False
        newf = []
        for r in range(len(f)):
            row = ""
            for c in range(len(f[0])):
                ns = checkNeighs(r,c,f)
                seat = f[r][c]
                if seat == 'L' and ns == 0:
                    # f[r][c] = '#'
                    row += '#'
                    change = True
                elif seat == '#' and ns >= 5:
                    # f[r][c] = 'L'
                    row += 'L'
                    change = True
                else:
                    row += seat
            newf.append(row)
        f = newf
        if not change:
            break
    count = 0
    for row in f:
        count += row.count('#')
    return count
    
def checkNeighs(r,c,seats):
    # neighs = []
    # for a in [r-1, r, r+1]:
    #     for b in [c-1, c, c+1]:
    #         if 0 <= a < len(seats) and 0 <= b < len(seats[0]) and (a,b) != (r,c):
    #             neighs.append((a,b))
    # count = 0
    # for n in neighs:
    #     if seats[n[0]][n[1]] == "#":
    #         count += 1
    # return count
    dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,-1),(-1,1),(1,1),(-1,-1)]
    count = 0
    for d in dirs:
        x = r + d[0]
        y = c + d[1]
        found = False
        while 0 <= x < len(seats) and 0 <= y < len(seats[0]):
            if seats[x][y] != '.':
                if seats[x][y] == '#':
                    found = True
                break
            x += d[0]
            y += d[1]
        if found:
            count += 1
    return count

import math
def day12():
    f = open('aoc12.txt', 'r').read().strip().split("\n")
    # ang = 0
    # x = 0
    # y = 0
    # for r in f:
    #     c = r[0]
    #     n = int(r[1:])
    #     if c == 'N':
    #         y += n
    #     elif c == 'S':
    #         y -= n
    #     elif c == 'E':
    #         x += n
    #     elif c == 'W':
    #         x -= n
    #     elif c == 'L':
    #         ang += n
    #     elif c == 'R':
    #         ang -= n
    #     elif c == 'F':
    #         x += n*math.cos(math.radians(ang))
    #         y += n*math.sin(math.radians(ang))
    # return abs(x) + abs(y)
    wx = 10
    wy = 1

    x = 0
    y = 0
    for r in f:
        c = r[0]
        n = int(r[1:])
        if c == 'N':
            wy += n
        elif c == 'S':
            wy -= n
        elif c == 'E':
            wx += n
        elif c == 'W':
            wx -= n
        elif c == 'L':
            ang = math.atan2(wy, wx) + math.radians(n)
            scale = math.sqrt(wx**2 + wy**2)
            wx = math.cos(ang)*scale
            wy = math.sin(ang)*scale
        elif c == 'R':
            ang = math.atan2(wy, wx) - math.radians(n)
            scale = math.sqrt(wx**2 + wy**2)
            wx = math.cos(ang)*scale
            wy = math.sin(ang)*scale
        elif c == 'F':
            x += wx * n
            y += wy * n

    return abs(x) + abs(y)

import re
def day4():
    f = open('aoc4.txt', 'r').read().strip().split("\n\n")
    fields = ["byr","iyr","eyr","hgt","hcl","ecl","pid"] # ignore cid
    count = 0
    valids = []
    for p in f:
        if all([(field+":") in p for field in fields]):
            count += 1
            valids.append(p)
    # return count

    total = 0
    for p in valids:
        joined = ' '.join(p.split("\n")).split(' ')
        pport = {}
        for item in joined:
            split = item.split(":")
            key = split[0]
            val = split[1]
            pport[key] = val
        print(pport)
        if not (len(pport["byr"]) == 4 and 1920 <= int(pport["byr"]) <= 2002):
            continue
        if not (len(pport["iyr"]) == 4 and 2010 <= int(pport["iyr"]) <= 2020):
            continue
        if not (len(pport["eyr"]) == 4 and 2020 <= int(pport["eyr"]) <= 2030):
            continue
        if not ((pport["hgt"][-2:] == "cm" and 150 <= int(pport["hgt"][:-2]) <=193) or (pport["hgt"][-2:] == "in" and 59 <= int(pport["hgt"][:-2]) <= 76)):
            continue
        if not re.match("^#[0-9a-f]{6}$", pport["hcl"]):
            continue
        if pport["ecl"] not in ["amb","blu","brn","gry","grn","hzl","oth"]:
            continue
        if not re.match("^[0-9]{9}$", pport["pid"]):
            continue
        total += 1
    return total


def day8():
    f = open('aoc8.txt', 'r').read().strip().split("\n")
    i = 0
    val = 0
    seen = set()

    def check(i2, val2, seen2):
        i = i2
        val = val2
        seen = seen2.copy()
        while i < len(f):
            if i in seen:
                return None
            seen.add(i)
            v = f[i]
            if v[:3] == "jmp":
                i += int(v.split(" ")[1])
            else:
                i += 1
                if v[:3] == "acc":
                    val += int(v.split(" ")[1])
        return val

    while True:
        # if i in seen:
        #     return val
        seen.add(i)

        v = f[i]
        if v[:3] == "jmp":
            res = check(i+1, val, seen)
            if res:
                return res
            i += int(v.split(" ")[1])
        elif v[:3] == 'nop':
            res = check(i+int(v.split(" ")[1]), val, seen)
            if res:
                return res
            i += 1
        elif v[:3] == "acc":
            val += int(v.split(" ")[1])
            i += 1

def day7():
#     f = """shiny gold bags contain 2 dark red bags.
# dark red bags contain 2 dark orange bags.
# dark orange bags contain 2 dark yellow bags.
# dark yellow bags contain 2 dark green bags.
# dark green bags contain 2 dark blue bags.
# dark blue bags contain 2 dark violet bags.
# dark violet bags contain no other bags.""".strip().split("\n")
    f = open('aoc7.txt', 'r').read().strip().split("\n")
    contains = {}
    # for rule in f:
    #     split = rule[:-1].split(" contain ")
    #     contains[split[0][:-5]] = set([i[2:-4].strip() for i in split[1].split(", ")])
    # colors = set(['shiny gold'])
    # while True:
    #     changed = False
    #     for color in set(contains.keys()) - colors:
    #         l = len(contains[color].intersection(colors))
    #         if l != 0:
    #             colors.add(color)
    #             changed = True
    #     if not changed:
    #         break
    # return len(colors)-1
    
    for rule in f:
        split = rule[:-1].split(" contain ")
        contains[split[0][:-5]] = set([i[:-4].strip() for i in split[1].split(", ")])

    counts = {}
    def countBags(bag):
        print(bag, contains[bag])
        if bag in counts:
            return counts[bag]
        elif len(contains[bag]) == 1 and list(contains[bag])[0][:2] == "no":
            counts[bag] = 1
            return 1

        total = 1
        for b in contains[bag]:
            num = int(b[0])
            color = b[2:]
            total += num*countBags(color)
        counts[bag] = total
        return total
    
    result = countBags("shiny gold")
    return result - 1

def day18():
    f = open('aoc18.txt', 'r').read().strip().split("\n")
    total = 0
    for row in f:
        total += eval2(row.strip())
    return total

def eval2(e):
    if "(" in e:
        l = e.index("(")
        opens = 1
        i = l+1
        while True:
            if e[i] == "(":
                opens += 1
            elif e[i] == ")":
                opens -= 1
            if opens == 0:
                break
            i += 1
        r = i

        before = e[:l]
        after = e[r+1:]
        middle = e[l+1:r]
        result = eval2(middle)
        combined = before + str(result) + after
        return eval2(combined)
    else:
        ops = e.split(" ")
        # i = 1
        # ans = int(ops[0])
        # while i < len(ops):
        #     if ops[i] == "*":
        #         ans *= int(ops[i+1])
        #     else:
        #         ans += int(ops[i+1])
        #     i += 2
        # return ans
        
        i = 1
        while i < len(ops):
            if ops[i] == "+":
                ops[i-1] = int(ops[i-1]) + int(ops[i+1])
                ops.pop(i)
                ops.pop(i)
            else:
                i += 2
        
        total = 1
        for i in range(0, len(ops), 2):
            total *= int(ops[i])
        return total

class C(int):
    def __init__(self, n):
        self.n = n

    def __add__(self, o):
        return C(self.n * o.n)
        # return C(self.n + o.n)
    
    def __mul__(self, o):
        return C(self.n + o.n)

    def __sub__(self, o):
        return C(self.n * o.n)

def day18_2():
    f = open('aoc18.txt', 'r').read().strip().split("\n")
    total = 0

    def convert(expr):
        a = ""
        for s in expr:
            if s == '+':
                a += '*'
                # a += '+'
            elif s == '*':
                a += '+'
                # a += '-'
            elif s in ' ()':
                a += s
            else:
                a += f'C({s})'
        return a

    for row in f:
        total += eval(convert(row))
    return total


def day19():
    f = open('aoc19.txt', 'r').read().strip().split("\n\n")
    # rules = f[0].split("\n")
    rules = (f[0] + """\n8: 42 | 42 8
11: 42 31 | 42 300
300: 11 31""").split("\n") # 300 is arbitrary, convert ternary rule to binary
    words = f[1].split("\n")
    
#     rules = """0: 4 1
# 1: 2 3 | 3 2
# 2: 4 4 | 5 5
# 3: 4 5 | 5 4
# 4: 'a'
# 5: 'b'""".split("\n")
#     words = """ababb
# babab
# abbba
# aaabb
# aaaabb""".split("\n")

    r = {}
    for rule in rules:
        spl = rule.split(": ")
        key = eval(spl[0])
        vals = [tuple([eval(i) for i in v.split(" ")]) for v in spl[1].split(" | ")]
        r[key] = vals

    memo = {}
    def parse(word, rule):
        if (word, rule) in memo:
            return memo[(word, rule)]
        for x in r[rule]:
            if len(x) == 1:
                if isinstance(x[0], str):
                    if (word == x[0]):
                        memo[(word, rule)] = True
                        return True
                elif parse(word, x[0]):
                    memo[(word, rule)] = True
                    return True
            else:
                for i in range(1, len(word)):
                    left = word[:i]
                    right = word[i:]
                    if (parse(left, x[0]) and parse(right, x[1])):
                        memo[(word, rule)] = True
                        return True
        memo[(word, rule)] = False
        return False

    total = 0
    for word in words:
        if parse(word, 0):
            total += 1
    return total

def day20():
    f = open('aoc20.txt', 'r').read().strip().split("\n\n")
    
    sqs = {}
    m = {}
    for square in f:
        spl = square.split(":\n")
        num = int(spl[0].split(' ')[1])
        tiles = spl[1].split("\n")
        sqs[num] = tiles
        s1 = tiles[0]
        s2 = tiles[-1]
        s3 = ''.join([s[0] for s in tiles])
        s4 = ''.join([s[-1] for s in tiles])
        for s in [s1, s2, s3, s4]:
            if s in m:
                m[s].add(num)
            else:
                m[s] = {num}
            srev = s[::-1]
            if srev in m:
                m[srev].add(num)
            else:
                m[srev] = {num}

    corners = []
    for square in f:
        spl = square.split(":\n")
        num = int(spl[0].split(' ')[1])
        tiles = spl[1].split("\n")
        s1 = tiles[0]
        s2 = tiles[-1]
        s3 = ''.join([s[0] for s in tiles])
        s4 = ''.join([s[-1] for s in tiles])
        count = 0
        for s in [s1, s2, s3, s4]:
            if len(m[s]) == 1:
                count += 1
            if len(m[s[::-1]]) == 1:
                count += 1
        if count == 4:
            corners.append(num)
    
    edges = []
    for square in f:
        spl = square.split(":\n")
        num = int(spl[0].split(' ')[1])
        tiles = spl[1].split("\n")
        s1 = tiles[0]
        s2 = tiles[-1]
        s3 = ''.join([s[0] for s in tiles])
        s4 = ''.join([s[-1] for s in tiles])
        count = 0
        for s in [s1, s2, s3, s4]:
            if len(m[s]) == 1:
                count += 1
            if len(m[s[::-1]]) == 1:
                count += 1
        if count == 2:
            edges.append(num)
    print(len(edges))

    # alledges = corners + edges

    grid = [[None for i in range(12)] for i in range(12)]
    keys = [[None for i in range(12)] for i in range(12)]
    grid[0][0] = sqs[corners[0]]
    keys[0][0] = corners[0]

    def rotateLeft(left, num, newgrid=False):
        tiles = sqs[num]
        if newgrid: # hacky
            tiles = [''.join([tiles[i][j] for i in range(10)]) for j in range(9, -1, -1)]

        s1 = tiles[0][::-1]
        s2 = tiles[-1]
        s3 = ''.join([s[0] for s in tiles])
        s4 = ''.join([s[-1] for s in tiles])[::-1]
        if s1 == left:
            return [''.join([tiles[i][j] for i in range(10)]) for j in range(9, -1, -1)]
        elif s2 == left:
            return [''.join([tiles[i][j] for i in range(9, -1, -1)]) for j in range(10)]
        elif s3 == left:
            return tiles
        elif s4 == left:
            return [''.join([tiles[i][j] for j in range(9, -1, -1)]) for i in range(9, -1, -1)]
        
        s1 = tiles[0]
        s2 = tiles[-1][::-1]
        s3 = ''.join([s[0] for s in tiles])[::-1]
        s4 = ''.join([s[-1] for s in tiles])
        if s1 == left:
            return [''.join([tiles[i][j] for i in range(10)]) for j in range(10)]
        elif s2 == left:
            return [''.join([tiles[i][j] for i in range(9, -1, -1)]) for j in range(9, -1, -1)]
        elif s3 == left:
            return [''.join([tiles[i][j] for j in range(10)]) for i in range(9, -1, -1)]
        elif s4 == left:
            return [''.join([tiles[i][j] for j in range(9, -1, -1)]) for i in range(10)]
        else:
            print(left, num, "SOMETHING BROKE")

    # tiles = sqs[corners[1]]
    # # tiles = square.split("\n")
    # s1 = tiles[0]
    # s2 = tiles[-1]
    # s3 = ''.join([s[0] for s in tiles])
    # s4 = ''.join([s[-1] for s in tiles])
    # print(m[s1], m[s2], m[s3], m[s4])
    for i in range(1, 12):
        s4 = ''.join([s[-1] for s in grid[0][i-1]])
        m[s4].remove(keys[0][i-1])
        keys[0][i] = m[s4].pop()
        grid[0][i] = rotateLeft(s4, keys[0][i])

    for i in range(1, 12):
        s4 = grid[i-1][0][-1]
        m[s4].remove(keys[i-1][0])
        keys[i][0] = m[s4].pop()
        newgrid = rotateLeft(s4[::-1], keys[i][0], True)
        grid[i][0] = [''.join([newgrid[i][j] for i in range(9,-1,-1)]) for j in range(10)]

    for i in range(1, 12):
        for j in range(1, 12):
            tiles = grid[i][j-1]
            s1 = tiles[0]
            s2 = tiles[-1][::-1]
            s3 = ''.join([s[0] for s in tiles])[::-1]
            s4 = ''.join([s[-1] for s in tiles])
            m[s4].remove(keys[i][j-1])
            keys[i][j] = m[s4].pop()
            grid[i][j] = rotateLeft(s4, keys[i][j])
    
    for i in range(12):
        for j in range(12):
            val = grid[i][j]
            grid[i][j] = [''.join([val[k][l] for l in range(1, 9)]) for k in range(1, 9)]
    
    wholegrid = []
    for i in range(12):
        for j in range(8):
            wholegrid.append(''.join([grid[i][k][j] for k in range(12)]))
    print(wholegrid)

    def monster(tiles):
        o1 = [''.join([tiles[i][j] for i in range(96)]) for j in range(95, -1, -1)]
        o2 = [''.join([tiles[i][j] for i in range(95, -1, -1)]) for j in range(96)]
        o3 = grid
        o4 = [''.join([tiles[i][j] for j in range(95, -1, -1)]) for i in range(95, -1, -1)]
        o5 = [''.join([tiles[i][j] for i in range(96)]) for j in range(10)]
        o6 = [''.join([tiles[i][j] for i in range(95, -1, -1)]) for j in range(95, -1, -1)]
        o7 = [''.join([tiles[i][j] for j in range(96)]) for i in range(95, -1, -1)]
        o8 = [''.join([tiles[i][j] for j in range(95, -1, -1)]) for i in range(96)]

        os = [o1, o2, o3, o4, o5, o6, o7, o8]
        ss = set()
        mons = """                  # 
#    ##    ##    ###
 #  #  #  #  #  #   """.split("\n")
        for i in range(3):
            for j in range(len(mons[i])):
                if mons[i][j] == "#":
                    ss.add((i, j))
        h = 3
        w = 20
        for o in os:
            total = 0
            for i in range(len(o)-h):
                for j in range(len(o[0])-w):
                    if all([o[i+k][j+l]=='#' for (k,l) in ss]):
                        total += 1
            print(total)
            if total > 0:
                return total
    
    ms = monster(wholegrid)
    size = 15
    return sum([x.count("#") for x in wholegrid]) - size*ms


    # for key in m:
    #     if len(m[key]) == 1:
    #         print(key,m[key])

def day21():
    f = open('aoc21.txt', 'r').read().strip().split("\n")

    lines = []
    allergens = set()
    allfoods = []

    for i in range(len(f)):
        line = f[i]
        spl = line.split(" (contains ")
        foods = spl[0].split(" ")
        algs = spl[1][:-1].split(", ") # take out contains and )
        lines.append((set(foods), set(algs)))
        allergens = allergens.union(algs)
        allfoods.extend(foods)
    
    afoods = {}
    for a in allergens:
        foods = None
        for l in lines:
            if a in l[1]:
                if foods:
                    foods = foods.intersection(l[0])
                else:
                    foods = l[0]
        afoods[a] = foods
    
    # allergics = set()
    # for key in afoods:
    #     allergics = allergics.union(afoods[key])
    # count = 0
    # for f in allfoods:
    #     if f not in allergics:
    #         count += 1
    # return count

    found = True
    answers = {}
    while found:
        found = False
        for key in afoods:
            if len(afoods[key]) == 1:
                found = True
                food = list(afoods[key])[0]
                answers[key] = food
                for k2 in afoods:
                    afoods[k2].discard(food)
    
    result = []
    for key in sorted(answers.keys()):
        result.append((key, answers[key]))
    return result

def day22():
    f = open('aoc22.txt', 'r').read().strip().split("\n\n")
    p1 = [int(x) for x in f[0].split("\n")[1:]]
    p2 = [int(x) for x in f[1].split("\n")[1:]]
    
    # p1 = [int(x) for x in "9, 2, 6, 3, 1".split(", ")]
    # p2 = [int(x) for x in "5, 8, 4, 7, 10".split(", ")]
    
    def won1(copyp1, copyp2):
        seen = set()
        p1, p2 = copyp1, copyp2
        while len(p1) > 0 and len(p2) > 0:
            t1 = p1.pop(0)
            t2 = p2.pop(0)
            if (tuple(sorted(p1)), tuple(sorted(p2))) in seen:
                p1.extend([t1, t2])
            else:
                seen.add((tuple(sorted(p1)), tuple(sorted(p2))))
                if t1 <= len(p1) and t2 <= len(p2):
                    one = won1(p1[:t1].copy(), p2[:t2].copy())[0]
                    if one:
                        p1.extend([t1, t2])
                    else:
                        p2.extend([t2, t1])
                else:
                    if t1 > t2:
                        p1.extend([t1, t2])
                    else:
                        p2.extend([t2, t1])
 
        ans = (True, p1) if len(p1) > 0 else (False, p2)
        return ans

    # def won1(copyp1, copyp2):
    #     seen = set()
    #     p1, p2 = copyp1, copyp2

    #     while len(p1) > 0 and len(p2) > 0:
    #         # t1 = p1.pop(0)
    #         # t2 = p2.pop(0)
    #         t1 = p1[0]
    #         t2 = p2[0]
    #         # if serial(p1, p2) in seen:
    #         #     p1 += (t1, t2)
    #         if ((p1, p2)) in seen:
    #             p1 += (t1,) + (t2,)
    #         else:
    #             # seen.add(serial(p1, p2))
    #             seen.add((p1, p2))
    #             if t1 <= len(p1) and t2 <= len(p2):
    #                 one = won1(p1[:t1], p2[:t2])[0]
    #                 if one:
    #                     p1 += (t1,) + (t2,)
    #                 else:
    #                     p2 += (t2,) + (t1,)
    #             else:
    #                 if t1 > t2:
    #                     p1 += (t1,) + (t2,)
    #                 else:
    #                     p2 += (t2,) + (t1,)
    #         print(p1, p2)
 
    #     ans = (True, p1) if len(p1) > 0 else (False, p2)
    #     return ans

    won1, l = won1(p1, p2)
    return sum([i*l[-i] for i in range(1, len(l)+1)])

def day23():
    cups = [int(x) for x in list("193467258")]
    # cups = [int(x) for x in list("389125467")]
    cups += [i for i in range(10, 1000001)]
    l = len(cups)

    nxt = {cups[i]: cups[i+1] for i in range(l-1)}
    nxt[cups[-1]] = cups[0]

    s = cups[0]
    a = nxt[s]
    b = nxt[a]
    c = nxt[b]
    for it in range(10000000):
        x = (s-1) % l
        if x == 0:
            x = l
        while (x in [a,b,c]):
            x = (x-1)%l
            if x == 0:
                x = l
        nxt[s] = nxt[c]
        nxt[c] = nxt[x]
        nxt[x] = a

        s = nxt[s]
        a = nxt[s]
        b = nxt[a]
        c = nxt[b]
    return nxt[1] * nxt[nxt[1]]

        # ind = 0
        # current = cups[ind]
        # pickup = cups[ind+1:ind+4]
        # x = (current-1)%l
        # if x == 0:
        #     x = l
        # while (x in pickup):
        #     x = (x-1)%l
        #     if x == 0:
        #         x = l
        # i = cups.index(x)
        # cups = cups[ind+4:i+1] + pickup + cups[i+1:] + cups[:ind+1]
    # return cups

def day24():
    f = open('aoc24.txt', 'r').read().strip().split("\n")
    
    def parse(s):
        i = 0
        dirs = []
        while i < len(s):
            if s[i:i+2] in ["se", "sw", "nw", "ne"]:
                dirs.append(s[i:i+2])
                i += 2
            else:
                dirs.append(s[i])
                i += 1
        return dirs
    
    locs = set()
    amts = {"e": (2, 0), "w": (-2, 0), "ne": (1, 1), "nw": (-1, 1), "se": (1, -1), "sw": (-1, -1)}
    for s in f:
        dirs = parse(s)
        x = 0
        y = 0
        for d in dirs:
            x += amts[d][0]
            y += amts[d][1]
        if (x,y) in locs:
            locs.remove((x,y))
        else:
            locs.add((x,y))
    
    def nbrs(x,y):
        return [(x+val[0], y+val[1]) for val in amts.values()]

    for it in range(100):
        xmax = max(locs, key = lambda x: x[0])[0]
        ymax = max(locs, key = lambda x: x[1])[1]
        xmin = min(locs, key = lambda x: x[0])[0]
        ymin = min(locs, key = lambda x: x[1])[1]

        up = set()
        for y in range(ymin-1, ymax+2):
            if y%2 == 0:
                r = range(xmin//2*2-2, xmax//2*2+4, 2)
            else:
                r = range(xmin//2*2-3, xmax//2*2+3, 2)
            for x in r:
                count = 0
                for n in nbrs(x, y):
                    if n in locs:
                        count += 1
                if (x,y) in locs:
                    if count == 0 or count > 2:
                        up.add((x,y))
                else:
                    if count == 2:
                        up.add((x,y))
        for pt in up:
            if pt in locs:
                locs.remove(pt)
            else:
                locs.add(pt)
    
    return len(locs)

def day25():
    k1 = 1717001
    k2 = 523731
    subj = 7
    div = 20201227
    
    loopc = 0
    loopd = 0 
    val1 = 1
    count = 0
    while True:
        val1 *= subj
        val1 %= div
        count += 1
        if val1 == k1:
            break
    print(count)
    val2 = 1
    count = 0
    while True:
        val2 *= subj
        val2 %= div
        count += 1
        if val2 == k2:
            break
    print(count)

    # k1 = 17907740
    # k2 = 2337351
    ans1 = 1
    ans2 = 1
    for i in range(2337351):
        ans1 *= k1
        ans1 %= div
    for j in range(17907740):
        ans2 *= k2
        ans2 %= div
    return ans1, ans2

# print(day25())

a = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 9, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 22, 24]
d = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 7, 7, 7, 8, 8, 8, 8, 9, 10, 11, 12, 14, 16, 17, 18, 20, 23]
aa = [1, 8, 3, 2, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 16, 18, 19, 21, 22, 24, 2, 4, 6, 7, 1, 2, 3, 4, 1, 2, 4, 6, 9, 10, 12, 2, 3, 4, 5, 6, 7]
dd = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 1, 8, 10, 12, 14, 16, 17, 18, 20, 23, 1, 1, 2, 3, 5, 7, 8, 9, 5, 1, 3, 5, 7, 8, 11, 1]
# print(sorted(a) == sorted(aa))
# print(sorted(d) == sorted(dd))

txt = """Salve! Let me tell you I had two large groups, one of twenty kids and one of nineteen kids. Don't have any left over from my $38 of candy.
Kaixo! There were just four waves of twenty visitors that I saw. And then another group of thirteen. I did give them good candy, after all I spent $88 on it.
Bonjour! I had an okay night, but was a bit disappointed by the turnout. I saw negative five trick or treaters and twenty more that night. I paid $13 for my candy.
Ẹ n lẹ! Thanks for stopping by like you said you would. I kept a close count, and there were only nine kids. Well plus ten kids. Oh and did I mention that this was counting out of the fourth group of twenty? I also bought exactly $68 of candy.
K'uxi! It's a bit of a long story, but I had twenty groups of two kids and one other. Later, I had five groups of two and one kids. Finally at the end of the night, another two and one kids visited. That was enough for my $65 of candy.
Hifa yifunger! I think total that night there were four groups of twenty. Then another fifteen trick or treaters. Oh and another two. I used just about all of my $70 worth of candy.
Niltze! I was doing okay, how about you? Oh I just had two groups of nine kids, as well as group of twenty stop by. Good deal for only $34 worth of candy!
Su'mae! Well, I had nine trick or treaters, and I thought that was going to be it until half of a hundred stopped by. Good thing I spent enough at $57!
Demat! I saw just one lone trick or treater. Oh and then four and half times, I saw a pod of twenty kids come by. I spent $89 on my candy.
Hej med jer! Let's see, there was three sets of twenty visitors, and then there were just four more than ten that came later. I spent $67 on my candy and you can't have any of it!
Gamarjoba! I didn't see that many come by, but those that did were great! I saw just two short of twenty kids. Good thing I only paid $7 for my candy."""
texts = txt.split("\n")
inds = [1, 5, 2, 11, 13, 27, 4, 2, 2, 7, 11]
for i in range(len(texts)):
    t = texts[i]
    j = t.index('!')
    ti = t[j+2:]
    # print(len(ti.split(" ")))
    print(ti.split(" ")[inds[i]-1])