
def daytemp():
    f = open('aoctemp.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    grid = []
    for i in range(len(f)):
        row = f[i]
        pass
    for row in f:
        pass
    return ans

def day1():
    f = open('aoc1.txt', 'r').read().strip()
    fs = f.split("\n\n")
    sums = sorted([sum(int(i) for i in s.strip().split("\n")) for s in fs])
    print(sums[-1])
    print(sum(sums[-3:]))


def day2():
    # scores = {"A X": 3, "B Y": 3, "C Z": 3, "A Y": 6, "A Z": 0, "B X": 0, "B Z": 6, "C X": 6, "C Y": 0}
    # shapes = {"X": 1, "Y": 2, "Z": 3}
    scores = {"A X": 3, "B Y": 2, "C Z": 1, "A Y": 1, "A Z": 2, "B X": 1, "B Z": 3, "C X": 2, "C Y": 3}
    shapes = {"X": 0, "Y": 3, "Z": 6}
    f = open('aoc2.txt', 'r').read().strip().split("\n")
    score = 0
    for row in f:
        score += scores[row]
        xs = row.split(" ")
        x,y = xs[0], xs[1]
        score += shapes[y]
    print(score)


def day3():
    f = open('aoc3.txt', 'r').read().strip().split("\n")
    ans = 0
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # for row in f:
    #     half = len(row)//2
    #     x, y = row[:half], row[half:]
    #     letter = set(x).intersection(set(y)).pop()
    #     ans += letters.index(letter)+1
    # return ans
    for i in range(len(f)//3):
        x,y,z = f[i*3], f[i*3+1], f[i*3+2]
        letter = set(x).intersection(set(y)).intersection(set(z)).pop()
        ans += letters.index(letter)+1
    return ans


def day4():
    f = open('aoc4.txt', 'r').read().strip().split("\n")
    ans = 0
    for row in f:
        two = row.split(",")
        x,y = two[0], two[1]
        xsplit = x.split("-")
        ysplit = y.split("-")
        x1, x2 = int(xsplit[0]), int(xsplit[1])
        y1, y2 = int(ysplit[0]), int(ysplit[1])
        # if (x1 <= y1 and y2 <= x2) or (y1 <= x1 and x2 <= y2):
        if not ((y2 < x1) or (x2 < y1)):
            ans += 1

    return ans

def day5():
    f = open('aoc5.txt', 'r').read().strip().split("\n\n")[1]
    boxes = {1: list("JZGVTDBN"), 2:list("FPWDMRS"),3:list("ZSRCV"),4:list("GHPZJTR"),5:list("FQZDNJCT"),6:list("MFSGWPVN"),6:list("MFSGWPVN"),7:list("QPBVCG"),8:list("NPBZ"),9:list("JPW")}
    ans = ""
    for row in f.split("\n"):
        row = row[5:]
        s = row.split(" from ")
        num = int(s[0])
        s2 = s[1].split(" to ")
        a = int(s2[0])
        b = int(s2[1])
        bs = boxes[a][:num]
        boxes[a] = boxes[a][num:]
        boxes[b] = bs + (boxes[b])
    
    for i in range(1,10):
        ans += boxes[i][0]

    return ans

def day6():
    f = open('aoc6.txt', 'r').read().strip()
    for i in range(len(f)-14):
        if len(set(f[i:i+14])) == 14:
            return i+14


class Dire:
    def __init__(self, name, parent, dirs, files):
        self.name = name
        self.parent = parent
        self.dirs = dirs
        self.files = files

def day7():
    f = open('aoc7.txt', 'r').read().strip()[2:].split("\n$ ")

    root = Dire("/",None,set(),set())
    curdir = root

    for chunk in f:
        lines = chunk.split("\n")
        com = lines[0]
        resp = lines[1:]
        if com == "ls":
            for r in resp:
                q = r.split(" ")
                if q[0] == "dir":
                    if q[1] not in [d.name for d in curdir.dirs]:
                        curdir.dirs.add(Dire(q[1],curdir,set(),set()))
                else:
                    curdir.files.add((int(q[0]),q[1]))
                    
        else: # cd
            d = com[3:]
            if d == "/":
                curdir = root
            elif d == "..":
                curdir = curdir.parent
            else:
                for i in curdir.dirs:
                    if i.name == d:
                        curdir = i
                        break
    
    dp = {}
    def sumdire(x):
        if x in dp:
            return dp[x]
        ans = sum([i[0] for i in x.files])
        for d in x.dirs:
            ans += sumdire(d)

        dp[x] = ans
        return ans
    
    sumdire(root)

    # totals = 0
    # for key in dp:
    #     if dp[key] <= 100000:
    #         totals += dp[key]
    # return totals

    minans = float("inf")
    for key in dp:
        if dp[root] - dp[key] <= 40000000 and dp[key] < minans:
            minans = dp[key]

    return minans


def day8():
    f = open('aoc8.txt', 'r').read().strip().split("\n")
    ans = 0
    m = {}
    grid = []
    for row in f:
        trees = [int(i) for i in row]
        grid.append(trees)
    
    def isvisible(grid, i, j):
        val = grid[i][j]
        a = all([grid[x][j] < val for x in range(i)])
        b = all([grid[x][j] < val for x in range(i+1,len(grid))])
        c = all([grid[i][y] < val for y in range(j)])
        d = all([grid[i][y] < val for y in range(j+1,len(grid[0]))])
        return a or b or c or d

    def score(grid, i, j):
        val = grid[i][j]
        a = i
        for x in range(i-1, -1, -1):
            if grid[x][j] >= val:
                a = i-x
                break
        b = len(grid)-i-1
        for x in range(i+1, len(grid)):
            if grid[x][j] >= val:
                b = x-i
                break
        c = j
        for y in range(j-1, -1, -1):
            if grid[i][y] >= val:
                c = j-y
                break
        d = len(grid[0])-j-1
        for y in range(j+1, len(grid[0])):
            if grid[i][y] >= val:
                d = y-j
                break
        
        return a*b*c*d

    # for i in range(len(grid)):
    #     for j in range(len(grid[0])):
    #         if isvisible(grid,i,j):
    #             ans += 1
    # return ans

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            m[(i,j)] = score(grid,i,j)
    
    return max(m.values())


def day9():
    f = open('aoc9.txt', 'r').read().strip().split("\n")
    s = {(0,0)}
    for i in range(len(f)):
        pass

    # head = [0,0]
    # tail = [0,0]
    h0 = [0,0]
    h1 = [0,0]
    h2 = [0,0]
    h3 = [0,0]
    h4 = [0,0]
    h5 = [0,0]
    h6 = [0,0]
    h7 = [0,0]
    h8 = [0,0]
    h9 = [0,0]

    def up(x, y):
        head = x.copy()
        tail = y.copy()
        if head[0] == tail[0]:
            if head[1] - tail[1] >= 2:
                tail[1] = head[1] - 1
            elif tail[1] - head[1] >= 2:
                tail[1] = head[1] + 1
        elif head[1] == tail[1]:
            if head[0] - tail[0] >= 2:
                tail[0] = head[0] - 1
            elif tail[0] - head[0] >= 2:
                tail[0] = head[0] + 1
        else:
            if tail[0] - head[0] >= 2:
                tail[0] -= 1
                if tail[1] > head[1]:
                    tail[1] -= 1
                else:
                    tail[1] += 1
            elif tail[0] - head[0] <= -2:
                tail[0] += 1
                if tail[1] > head[1]:
                    tail[1] -= 1
                else:
                    tail[1] += 1
            elif tail[1] - head[1] >= 2:
                tail[1] -= 1
                if tail[0] > head[0]:
                    tail[0] -= 1
                else:
                    tail[0] += 1
            elif tail[1] - head[1] <= -2:
                tail[1] += 1
                if tail[0] > head[0]:
                    tail[0] -= 1
                else:
                    tail[0] += 1

            # if head[0] - tail[0] >= 2:
            #     tail = [head[0]-1, head[1]]
            # elif head[1] - tail[1] >= 2:
            #     tail = [head[0], head[1]-1]
            # elif head[0] - tail[0] <= -2:
            #     tail = [head[0]+1, head[1]]
            # elif head[1] - tail[1] <= -2:
            #     tail = [head[0], head[1]+1]

        return head, tail

    dirs = {"D": (0, -1), "U": (0,1), "R": (1,0), "L": (-1,0)}
    # for row in f:
    #     x = row.split(" ")
    #     dir = x[0]
    #     dist = int(x[1])
    #     for i in range(dist):
    #         d = dirs[dir]
    #         head[0] += d[0]
    #         head[1] += d[1]
    #         # print(head)
    #         head, tail = up(head, tail)
    #         # print(tail)
    #         s.add(tuple(tail))
    
    # return len(s)

    for row in f:
        x = row.split(" ")
        dir = x[0]
        dist = int(x[1])
        for i in range(dist):
            d = dirs[dir]
            h0[0] += d[0]
            h0[1] += d[1]
            h0, h1 = up(h0, h1)
            h1, h2 = up(h1, h2)
            h2, h3 = up(h2, h3)
            h3, h4 = up(h3, h4)
            h4, h5 = up(h4, h5)
            h5, h6 = up(h5, h6)
            h6, h7 = up(h6, h7)
            h7, h8 = up(h7, h8)
            h8, h9 = up(h8, h9)

            s.add(tuple(h9))

    return len(s)

def day10():
    f = open('aoc10.txt', 'r').read().strip().split("\n")
    m = {}
    cycle = 1
    v = 1
    for i in range(len(f)):
        pass
    # for row in f:
    #     m[cycle] = v
    #     if row == "noop":
    #         cycle += 1
    #     else:
    #         num = int(row[5:])
    #         cycle += 1
    #         m[cycle] = v
    #         cycle += 1
    #         v += num
    # return m[20]*20 + m[60]*60 + m[100]*100 + m[140]*140 + m[180]*180 + m[220]*220

    chars = ""
    sprite = [0,1,2]
    cycle = 0
    for row in f:
        if (cycle)%40 in sprite:
            chars += "#"
        else:
            chars += "."
        if row == "noop":
            cycle += 1
        else:
            num = int(row[5:])
            cycle += 1
            if (cycle)%40 in sprite:
                chars += "#"
            else:
                chars += "."
            cycle += 1
            newc = sprite[1] + num
            sprite = [newc-1, newc, newc+1]

    for i in range(len(chars)//40):
        print(chars[i*40:i*40+40])


def day11():
    f = open('aoc11.txt', 'r').read().strip().split("\n\n")
    ans = 0
    lcm = 3*17*2*19*11*5*13*7

    items = {}
    ops = {0: (lambda x: x*5), 1: (lambda x: x+6), 2: (lambda x: x+5), 3: (lambda x: x+2), 4: (lambda x: x*7), 5: (lambda x: x+7), 6: (lambda x: x+1), 7: (lambda x: x*x)}
    testdivs = {}
    trues = {}
    falses = {}
    counts = {i: 0 for i in range(8)}

    for monkey in f:
        rows = monkey.split("\n")
        monkeynum = int(rows[0][-2])
        starting = [int(i) for i in rows[1].split(": ")[1].split(", ")]
        items[monkeynum] = starting

    for monkey in f:
        rows = monkey.split("\n")
        monkeynum = int(rows[0][-2])
        testdiv = int(rows[3].split(" ")[-1])
        testdivs[monkeynum] = testdiv
        truemonkey = int(rows[4].split(" ")[-1])
        trues[monkeynum] = truemonkey
        falsemonkey = int(rows[5].split(" ")[-1])
        falses[monkeynum] = falsemonkey

    for _ in range(10000):
        for mon in range(8):
            while len(items[mon]) > 0:
                item = items[mon].pop(0)
                counts[mon] += 1
                newval = ops[mon](item) % lcm
                if newval % testdivs[mon] == 0:
                    items[trues[mon]].append(newval)
                else:
                    items[falses[mon]].append(newval)

    anss = sorted(counts.values())
    ans = anss[-1] * anss[-2]
    return ans


def day12():
    f = open('aoc12.txt', 'r').read().strip().split("\n")
    # start = (0,0)
    starts = []
    end = (0,0)
    grid = []

    chars = "abcdefghijklmnopqrstuvwxyz"
    m = {chars[i]:i for i in range(len(chars))}

    for i in range(len(f)):
        row = list(f[i])
        for j in range(len(row)):
            if row[j] == "S" or row[j] == "a":
                starts.append((i,j))
                row[j] = "a"
            if row[j] == "E":
                end = (i,j)
                row[j] = "z"
        grid.append(list(m[i] for i in row))

    def steps(pos):
        i = pos[0]
        j = pos[1]
        poss = [(i+1, j), (i-1,j), (i, j-1), (i,j+1)]
        return [x for x in poss if 0<= x[0] < len(grid) and 0<=x[1]<len(grid[0]) and grid[x[0]][x[1]] - grid[pos[0]][pos[1]] <= 1]

    def bfs(start, end):
        seen = set()
        level = {start}
        
        count = 0
        while True:
            count += 1
            level2 = set()
            for item in level:
                if item not in seen:
                    seen.add(item)
                    for step in steps(item):
                        if step == end:
                            return count
                        level2.add(step)
            level = level2.copy()
            if count > 425:
                break
        
        return count

    # ans = bfs(start, end)
    # return ans
    
    vals = []
    for start in starts:
        vals.append(bfs(start, end))

    return min(vals)


import functools
from http.client import CONTINUE
from turtle import Turtle

def day13():
    f = open('aoc13.txt', 'r').read().strip().split("\n\n")

    def isordered(x, y):
        if isinstance(x, int) and isinstance(y, int):
            if x < y:
                return 1
            elif x > y:
                return -1
            else:
                return 0

        if isinstance(x, int):
            newx = [x]
        else:
            newx = x.copy()
        if isinstance(y, int):
            newy = [y]
        else:
            newy = y.copy()

        for i in range(len(newx)):
            if i >= len(newy):
                return -1
            result = isordered(newx[i], newy[i])
            if result != 0:
                return result
        
        if len(newx) == len(newy):
            return 0
        return 1

    # ans = 0
    # for i in range(len(f)):
    #     pair = f[i]
    #     s = pair.split("\n")
    #     x = eval(s[0])
    #     y = eval(s[1])
    #     if isordered(x,y) == 1:
    #         ans += (i+1)

    packets = [[[2]], [[6]]]
    for i in range(len(f)):
        pair = f[i]
        s = pair.split("\n")
        x = eval(s[0])
        y = eval(s[1])
        packets.append(x)
        packets.append(y)
    
    sorts = sorted(packets, key=functools.cmp_to_key(isordered), reverse=True)    
    return (sorts.index([[2]])+1)*(sorts.index([[6]])+1)


def day14():
    f = open('aoc14.txt', 'r').read().strip().split("\n")
    
    grid = set()
    for row in f:
        points = row.split(" -> ")
        for i in range(len(points)-1):
            p1 = [int(x) for x in points[i].split(",")]
            p2 = [int(x) for x in points[i+1].split(",")]
            if p1[0] == p2[0]:
                for j in range(min(p1[1], p2[1]),max(p1[1],p2[1])+1):
                    grid.add((p1[0], j))
            else:
                for j in range(min(p1[0], p2[0]),max(p1[0],p2[0])+1):
                    grid.add((j, p1[1]))
    
    height = max(p[1] for p in grid)
    count = 0

    def fall1(p):
        if p[1] > height:
            return None

        down = (p[0],p[1]+1)
        right = (p[0]+1,p[1]+1)
        left = (p[0]-1,p[1]+1)
        if (down in grid) and (right in grid) and (left in grid):
            grid.add(p)
            return p
        if down not in grid:
            return fall1(down)
        if left not in grid:
            return fall1(left)
        return fall1(right)

    def fall2(p):
        if p[1] == height+1:
            grid.add(p)
            return p

        down = (p[0],p[1]+1)
        right = (p[0]+1,p[1]+1)
        left = (p[0]-1,p[1]+1)
        if (down in grid) and (right in grid) and (left in grid):
            grid.add(p)
            return p
        if down not in grid:
            return fall2(down)
        if left not in grid:
            return fall2(left)
        return fall2(right)
    
    p = (500,0)
    # while True:
    #     result = fall1(p)
    #     if result is not None:
    #         count += 1
    #     else:
    #         break
    
    while True:
        result = fall2(p)
        count += 1
        if result == p:
            break

    return count

def day15():
    f = open('aoc15.txt', 'r').read().strip().split("\n")

    def manhattan(pt1, pt2):
        return abs(pt1[0]-pt2[0]) + abs(pt1[1]-pt2[1])

    allpts = set()
    def pts(sensor, beacon):
        m = manhattan(sensor, beacon)
        for i in range(sensor[0]-m, sensor[0]+m+1):
            j = 2000000
            if (sensor[1]-m) <= j <= (sensor[1]+m):
                if (i,j) != beacon and manhattan(sensor, (i,j)) <= m:
                    allpts.add((i,j))

    def pts2(y, sensor, beacon):
        m = manhattan(sensor, beacon)
        if abs(y-sensor[1]) > m:
            return None
        x1 = sensor[0]-(m-abs(y-sensor[1]))
        x2 = sensor[0]+(m-abs(y-sensor[1]))
        x1a = min(x1, x2)
        x2a = max(x1, x2)
        return (x1a, x2a)
        
    sensors = []
    beacons = []
    
    allpts = set()
    for row in f:
        print("new row!")
        print(row)
        row2 = row[10:]
        s = row2.split(": closest beacon is at ")
        sensor = s[0]
        beacon = s[1]
        sensorspl = sensor.split(", ")
        sensorx = int(sensorspl[0][2:])
        sensory = int(sensorspl[1][2:])
        se = (sensorx, sensory)
        beaconspl = beacon.split(", ")
        beaconx = int(beaconspl[0][2:])
        beacony = int(beaconspl[1][2:])
        be = (beaconx, beacony)
        
        # pts(se, be) # FOR PART 1

        sensors.append(se)
        beacons.append(be)

    # FOR PART 1
    # count = 0
    # for pt in allpts:
    #     if pt[1] == 2000000:
    #         count += 1
    # return count

    for p in range(4000001):
        if (p%10000 == 0):
            print(p) # just to keep track of progress
        ins = []
        for j in range(len(sensors)):
            s = sensors[j]
            b = beacons[j]
            pts = pts2(p,s,b)
            if pts:
                ins.append(pts)
        
        ins.sort(key = lambda x: x[0])
        total_interval = None
        for i in range(len(ins)):
            interval1 = ins[i]
            if interval1[1] < 0:
                continue
            if interval1[0] >= 4000001:
                break
            if total_interval is None:
                total_interval = interval1
                continue
            if interval1[0] <= total_interval[1]+1:
                total_interval = (total_interval[0], max(interval1[1], total_interval[1]))
            else:
                print("FOUND!!!")
                print(p)
                print(total_interval)
                print(interval1)
        
        if not (total_interval[0] <= 0 <= total_interval[1]) and (total_interval[0] <= 4000000 <= total_interval[1]):
            print("FOUND!!!")
            print(total_interval)


def day16():
    f = open('aoc16.txt', 'r').read().strip().split("\n")
#     f = """Valve AA has flow rate=0; tunnels lead to valves DD, II, BB
# Valve BB has flow rate=13; tunnels lead to valves CC, AA
# Valve CC has flow rate=2; tunnels lead to valves DD, BB
# Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE
# Valve EE has flow rate=3; tunnels lead to valves FF, DD
# Valve FF has flow rate=0; tunnels lead to valves EE, GG
# Valve GG has flow rate=0; tunnels lead to valves FF, HH
# Valve HH has flow rate=22; tunnel leads to valve GG
# Valve II has flow rate=0; tunnels lead to valves AA, JJ
# Valve JJ has flow rate=21; tunnel leads to valve II""".split("\n")

    flows = {}
    tunnels = {}
    for row in f:
        start = row[6:8]
        equals = row.index("=")
        semi = row.index(";")
        rate = int(row[equals+1:semi])
        flows[start] = rate
        endstext = row.split("to valve")[1]
        ends = endstext[1:].strip().split(", ")
        tunnels[start] = ends
        
    # point, step, opened
    dp = {("AA", 1, tuple()): 0}
    queue = [("AA", 1, tuple())]
    seen = set()
    maxval = 0

    # PART 1
    # STEP_CAP = 29

    # PART 2
    STEP_CAP = 25

    while len(queue) > 0:
        item = queue.pop(0)
        pt = item[0]
        step = item[1]
        opened = item[2]
        total = dp[item]
        if flows[pt] != 0 and pt not in opened: # case where we open a valve
            opened2 = tuple(sorted(list(opened)+[pt]))
            sumflows = sum(flows[o] for o in opened2)
            if (pt, step+1, opened2) not in dp or sumflows + total > dp[(pt, step+1, opened2)]:
                dp[(pt, step+1, opened2)] = sumflows + total
                if total+sumflows > maxval:
                    maxval = total+sumflows
                if step < STEP_CAP and (pt, step+1, opened2) not in seen:
                    seen.add((pt, step+1, opened2))
                    queue.append((pt, step+1, opened2))
        for nbr in tunnels[pt]:
            sumflows = sum(flows[o] for o in opened)
            if (nbr, step+1, opened) not in dp or sumflows + total > dp[(nbr, step+1, opened)]:
                dp[(nbr, step+1, opened)] = total + sumflows
                if total+sumflows > maxval:
                    maxval = total+sumflows
                if step < STEP_CAP and (nbr, step+1, opened) not in seen:
                    seen.add((nbr, step+1, opened))
                    queue.append((nbr, step+1, opened))
        del dp[item]
    
    # PART 1
    # return maxval
    
    # PART 2
    vals = {}
    maxval2 = 0
    for i in dp:
        if i[2] not in vals:
            vals[i[2]] = dp[i]
        elif vals[i[2]] < dp[i]:
            vals[i[2]] = dp[i]
    for i in vals:
        for j in vals:
            if len(set(i).intersection(set(j))) == 0:
                total = vals[i] + vals[j]
                if total > maxval2:
                    maxval2 = total   
    return maxval2 



def day17():
    f = open('aoc17.txt', 'r').read().strip()
    # f = """>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"""

    dirs = []
    for i in f:
        dirs.append(i)
        dirs.append("v")
    
    rock0 = {(0,0),(1,0),(2,0),(3,0)}
    rock1 = {(0,1),(1,0),(1,1),(1,2),(2,1)}
    rock2 = {(0,0),(1,0),(2,0),(2,1),(2,2)}
    rock3 = {(0,0),(0,1),(0,2),(0,3)}
    rock4 = {(0,0),(0,1),(1,0),(1,1)}

    rockset = [rock0, rock1, rock2, rock3, rock4]
    widths = [4, 3, 3, 1, 2]
    bottom0 = {(0,-1),(1,-1),(2,-1),(3,-1)}
    bottom1 = {(0,0),(1,-1),(2,0)}
    bottom2 = {(0,-1),(1,-1),(2,-1)}
    bottom3 = {(0,-1)}
    bottom4 = {(0,-1),(1,-1)}
    bottoms = [bottom0, bottom1, bottom2, bottom3, bottom4]

    right0 = {(4,0)}
    left0 = {(-1,0)}
    right1 = {(2,0),(3,1),(2,2)}
    left1 = {(-1,1),(0,0),(0,2)}
    right2 = {(3,0),(3,1),(3,2)}
    left2 = {(-1,0),(1,1),(1,2)}
    right3 = {(1,0),(1,1),(1,2),(1,3)}
    left3 = {(-1,0),(-1,1),(-1,2),(-1,3)}
    right4 = {(2,0), (2,1)}
    left4 = {(-1,0), (-1,1)}
    rights = [right0, right1, right2, right3, right4]
    lefts = [left0, left1, left2, left3, left4]

    space = set()
    rocks = 0
    rockindex = 0
    dirindex = 0
    
    height = 0
    rockpos = (2,3)

    seen = {}

    def rockcoords(rockpos, rockset):
        return {(a[0]+rockpos[0], a[1]+rockpos[1]) for a in rockset}

    currentrock = rockcoords(rockpos, rock0)

    def moverock(rockpos, rockindex, currentrock, dir):
        nonlocal space
        nonlocal height
        nonlocal seen

        if dir == ">":
            rs = {(a[0]+rockpos[0],a[1]+rockpos[1]) for a in rights[rockindex]}
            if rockpos[0] + widths[rockindex] == 7 or any(b in space for b in rs):
                return rockpos, currentrock
            else:
                return (rockpos[0]+1, rockpos[1]), {(a[0]+1, a[1]) for a in currentrock}
        if dir == "<":
            ls = {(a[0]+rockpos[0],a[1]+rockpos[1]) for a in lefts[rockindex]}
            if rockpos[0] == 0 or any(b in space for b in ls):
                return rockpos, currentrock
            else:
                return (rockpos[0]-1, rockpos[1]), {(a[0]-1, a[1]) for a in currentrock}
        if dir == "v":
            bot = {(a[0]+rockpos[0],a[1]+rockpos[1]) for a in bottoms[rockindex]}
            if rockpos[1] == 0 or any(b in space for b in bot):
                space = space.union(currentrock)                

                height = max(height, max(a[1] for a in currentrock)+1)

                # USED FOR PART 2
                # spacecache = frozenset((s[0], s[1]-height) for s in space if (height-30) <= s[1] <= height)
                # cache = (rockindex, dirindex, spacecache)
                # if cache in seen:
                #     print("FOUND!!!")
                #     print(seen[cache])
                #     print(rocks)
                #     print(height)
                #     return
                # seen[cache] = (rocks, height)

                return None, None
            else:
                return (rockpos[0], rockpos[1]-1), {(a[0], a[1]-1) for a in currentrock}

    # some numbers found for part 2
    # PERIOD = 1715
    # HEIGHT = 2616
    # MODREMAINING = 1614
    # while rocks != (116+MODREMAINING):
    while rocks != 2022: # for part 1
        dir = dirs[dirindex]
        dirindex = (dirindex+1) % len(dirs)

        rockpos, currentrock = moverock(rockpos, rockindex, currentrock, dir)
        if currentrock is None:
            rocks += 1
            rockindex = (rockindex+1) % len(rockset)
            rockpos = (2,height+3)
            currentrock = rockcoords(rockpos, rockset[rockindex])

    return height


def day18():
    f = open('aoc18.txt', 'r').read().strip().split("\n")

    cubes = set()

    def sides(cube):
        nonlocal cubes
        adds = [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]
        ns = [(cube[0]+i[0], cube[1]+i[1], cube[2]+i[2]) for i in adds]
        nexist = [n for n in ns if n in cubes]
        return 6-len(nexist)
    
    for row in f:
        coords = [int(i) for i in row.split(",")]
        cubes.add((coords[0], coords[1], coords[2]))
    
    # PART 1
    # area = 0
    # for cube in cubes:
    #     area += sides(cube)
    # return area
    
    (minx, maxx) = (min(i[0] for i in cubes)-1, max(i[0] for i in cubes)+1)
    (miny, maxy) = (min(i[1] for i in cubes)-1, max(i[1] for i in cubes)+1)
    (minz, maxz) = (min(i[2] for i in cubes)-1, max(i[2] for i in cubes)+1)

    outpoint = (minx, miny, minz)
    outside = {outpoint}
    layer = {outpoint}

    def outsidenbrs(cube):
        nonlocal cubes
        nonlocal outside
        nonlocal minx
        nonlocal maxx
        nonlocal miny
        nonlocal maxy
        nonlocal minx
        nonlocal maxz
        adds = [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]
        ns = [(cube[0]+i[0], cube[1]+i[1], cube[2]+i[2]) for i in adds]
        nexist = {n for n in ns if n not in cubes and n not in outside and minx <= n[0] <= maxx and miny <= n[1] <= maxy and minz <= n[2] <= maxz}
        return nexist

    while len(layer) > 0:
        newlayer = set()
        for l in layer:
            ns = outsidenbrs(l)
            newlayer = newlayer.union(ns)
            outside = outside.union(ns)
        layer = newlayer.copy()
    
    def sides2(cube):
        nonlocal cubes
        nonlocal outside
        adds = [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]
        ns = [(cube[0]+i[0], cube[1]+i[1], cube[2]+i[2]) for i in adds]
        nexist = [n for n in ns if n in outside and n not in cubes]
        return len(nexist)
    
    area = 0
    for cube in cubes:
        area += sides2(cube)
    return area


def day19():
    f = open('aoc19.txt', 'r').read().strip().split("\n")
    
    blueprints = {}
    # all tuples are (ore, clay, obsidian, geode)

    for i in range(len(f)):
        row = f[i]
        blueprintnum = i+1
        items = row.split(" ")
        orecost = (int(items[6]),0,0,0) 
        claycost = (int(items[12]),0,0,0) 
        obsidiancost = (int(items[18]), int(items[21]), 0,0) 
        geodecost = (int(items[27]), 0, int(items[30]),0) 
        blueprints[blueprintnum] = (orecost, claycost, obsidiancost, geodecost)
    
    dp = {}
    robots = (1,0,0,0)
    resources = (0,0,0,0)

    # MAXMINUTE = 24 # PART 1
    MAXMINUTE = 32 # PART 2

    def maxGeodes(blueprintnum, robots, resources, minute):
        b = blueprints[blueprintnum]
        maxMaxGeodes = 0
        maxore = max(i[0] for i in b)
        maxclay = max(i[1] for i in b)
        maxobsidian = max(i[2] for i in b)
        
        if robots[0] > maxore or robots[1] > maxclay or robots[2] > maxobsidian:
            return 0

        oremax = (MAXMINUTE-minute)*maxore - robots[0]*(MAXMINUTE-1-minute)
        claymax = (MAXMINUTE-minute)*maxclay - robots[1]*(MAXMINUTE-1-minute)
        obsidianmax = (MAXMINUTE-minute)*maxobsidian - robots[2]*(MAXMINUTE-1-minute)
        r1 = oremax if resources[0] >= oremax else resources[0]
        r2 = claymax if resources[1] >= claymax else resources[1]
        r3 = obsidianmax if resources[2] >= obsidianmax else resources[2]
        resources = (r1,r2,r3,resources[3])

        if (blueprintnum, robots, resources, minute) in dp:
            return dp[(blueprintnum, robots, resources, minute)]

        if minute == MAXMINUTE:
            dp[(blueprintnum, robots, resources, minute)] = resources[3]
            return resources[3]

        for i in range(3, -1, -1):
            bot = b[i]
            if all(bot[k] <= resources[k] for k in range(4)):
                newresources = [resources[k]-bot[k] for k in range(4)]
                for z in range(4):
                    # collect resources
                    newresources[z] += robots[z]
                newrobots = list(robots)
                newrobots[i] += 1
                result = maxGeodes(blueprintnum, tuple(newrobots), tuple(newresources), minute+1)
                if result > maxMaxGeodes:
                    maxMaxGeodes = result
            
        newresources = list(resources)
        for z in range(4):
            # collect resources
            newresources[z] += robots[z]
        result = maxGeodes(blueprintnum, robots, tuple(newresources), minute+1)
        if result > maxMaxGeodes:
            maxMaxGeodes = result

        dp[(blueprintnum, robots, resources, minute)] = maxMaxGeodes
        return maxMaxGeodes

    # ans = 0 
    # for i in blueprints: # PART 1

    ans = 1
    for i in range(1,4): # PART 2
        print(i)
        maxG = maxGeodes(i, robots, resources, 0)
        dp = {}
        print(maxG)
        # ans += (maxG*i) # PART 1
        ans *= (maxG) # PART 2
    return ans


def day20():
    f = open('aoc20.txt', 'r').read().strip().split("\n")

    originalnums = []
    nums = []
    for i in range(len(f)):
        row = f[i]
        # nums.append((int(row), i)) # PART 1
        # originalnums.append(int(row))

        nums.append((int(row)*811589153%(len(f)-1), i)) # PART 2
        originalnums.append(int(row)*811589153)

    numorder = nums.copy()
    for _ in range(10): # ONLY FOR PART 2
        for n in numorder:
            ind = nums.index(n)
            newind = (ind + n[0]) % (len(nums)-1)
            del nums[ind]
            nums.insert(newind, n)
    
    zeroind = nums.index((0,originalnums.index(0)))
    x = (zeroind+1000)%len(nums)
    y = (zeroind+2000)%len(nums)
    z = (zeroind+3000)%len(nums)
    # return nums[x][0]+nums[y][0]+nums[z][0] # PART 1
    return originalnums[nums[x][1]] + originalnums[nums[y][1]] + originalnums[nums[z][1]] # PART 2


def day21():
    f = open('aoc21.txt', 'r').read().strip().split("\n")
    m = {}

    for row in f:
        nums = row.split(" ")
        if len(nums) == 2:
            n = int(nums[1])
            m[nums[0][:-1]] = n
        else:
            first = nums[1]
            second = nums[3]
            op = nums[2]
            m[nums[0][:-1]] = (first,op,second)
    
    dp = {}
    def evaluate(elem):
        if elem == "humn": # FOR PART 2
            return None

        val = m[elem]

        if isinstance(val, int) or isinstance(val,str):
            dp[elem] = val
            return val
        
        if elem in dp:
            return dp[elem]
        
        op = val[1]
        first = evaluate(val[0])
        second = evaluate(val[2])

        if first is None or second is None: # FOR PART 2
            return None

        if op == "+":
            ans = first+second
        elif op == "-":
            ans = first-second
        elif op == "*":
            ans = first*second
        elif op == "/":
            ans = first/second

        dp[elem] = ans
        return ans
    
    def solve(elem, equals): # used for PART 2
        print(elem, m[elem], equals)
        if elem == "humn":
            return None

        val = m[elem]
        op = val[1]

        first = evaluate(val[0])
        second = evaluate(val[2])
        if first is None:
            if op == "+":
                neweq = equals-second
            elif op == "-":
                neweq = equals+second
            elif op == "*":
                neweq = equals/second
            elif op == "/":
                neweq = equals*second
            
            return solve(val[0], neweq)
        elif second is None:
            if op == "+":
                neweq = equals-first
            elif op == "-":
                neweq = first-equals
            elif op == "*":
                neweq = equals/first
            elif op == "/":
                neweq = first/equals
            
            return solve(val[2], neweq)

    # PART 1
    # return evaluate("root")

    # PART 2
    first = "sbtm"
    second = "bmgf"

    a = evaluate(first)
    b = evaluate(second)
    if a is None:
        solve(first, b)
    elif b is None:
        solve(second, a)


def day22():
    f = open('aoc22.txt', 'r').read().split("\n\n")
    grid = []

    for row in f[0].split("\n"):
        rowg = row.replace(" ","X")
        if len(rowg) < 150:
            rowg += "X"*(150-len(rowg)) # make all rows 150 long
        grid.append(list(rowg))
    
    # (row,col)
    dirs = {">": (0,1), "v":(1,0), "<":(0,-1), "^":(-1,0)}
    
    posrow = 0
    poscol = 50
    dirstr = ">v<^"
    
    rows = len(grid)
    cols = len(grid[0])

    curdir = 0
    def move(newdir):
        nonlocal curdir
        nonlocal posrow
        nonlocal poscol
        if newdir == "L":
            curdir = (curdir-1)%4
        elif newdir == "R":
            curdir = (curdir+1)%4
        else:
            count = newdir
            while count > 0:
                dir = dirs[dirstr[curdir]]
                nextrow = (posrow + dir[0]) % rows
                nextcol = (poscol + dir[1]) % cols
                if grid[nextrow][nextcol] == "#":
                    break

                # FOR PART 1
                # while grid[nextrow][nextcol] == "X":
                #     nextrow = (nextrow + dir[0]) % rows
                #     nextcol = (nextcol + dir[1]) % cols
                # if grid[nextrow][nextcol] == "#":
                #     break
                # else:
                #     posrow = nextrow
                #     poscol = nextcol
                #     count -= 1

                # FOR PART 2
                elif grid[nextrow][nextcol] == "X":
                    if curdir == 3: # up
                        if posrow == 0 and 50 <= poscol < 100: #A
                            nextrow = 150+(poscol-50)
                            nextcol = 0
                            nextdir = 0
                        elif posrow == 0 and 100 <= poscol < 150: #B
                            nextrow = 199
                            nextcol = poscol-100
                            nextdir = 3
                        elif posrow == 100 and 0 <= poscol < 50: #D
                            nextdir = 0
                            nextrow = 50+poscol
                            nextcol = 50
                    elif curdir == 2: # left
                        if poscol == 50 and 0 <= posrow < 50: #C
                            nextdir = 0
                            nextrow = 100+(49-posrow) 
                            nextcol = 0
                        elif poscol == 50 and 50 <= posrow < 100: #D
                            nextdir = 1
                            nextrow = 100
                            nextcol = (posrow-50)
                        elif poscol == 0 and 100 <= posrow < 150: #C
                            nextdir = 0
                            nextrow = 49-(posrow-100) 
                            nextcol = 50
                        elif poscol == 0 and 150 <= posrow < 200: #A
                            nextdir = 1
                            nextrow = 0 
                            nextcol = 50+(posrow-150)
                    elif curdir == 1: # down
                        if posrow == 199 and 0 <= poscol < 50: #B
                            nextdir = 1
                            nextrow = 0
                            nextcol = 100+poscol
                        elif posrow == 149 and 50 <= poscol < 100: #G
                            nextdir = 2
                            nextrow = 150+(poscol-50)
                            nextcol = 49
                        elif posrow == 49 and 100 <= poscol < 150: #E
                            nextdir = 2
                            nextrow = 50+(poscol-100)
                            nextcol = 99
                    elif curdir == 0: # right
                        if poscol == 149 and 0 <= posrow < 50: #F
                            nextdir = 2
                            nextrow = 100+(49-posrow)
                            nextcol = 99
                        elif poscol == 99 and 50 <= posrow < 100: #E
                            nextdir = 3
                            nextrow = 49
                            nextcol = 100+(posrow-50)
                        elif poscol == 99 and 100 <= posrow < 150: #F
                            nextdir = 2
                            nextrow = 49-(posrow-100)
                            nextcol = 149
                        elif poscol == 49 and 150 <= posrow < 200: #G
                            nextdir = 3
                            nextrow = 149
                            nextcol = 50+(posrow-150)
                    else:
                        print("SOMETHING HAS GONE WRONG")
                        break
                    
                    if grid[nextrow][nextcol] == "#":
                        break
                    else:
                        posrow = nextrow
                        poscol = nextcol
                        curdir = nextdir
                        count -= 1
                else:
                    posrow = nextrow
                    poscol = nextcol
                    count -= 1
    
    
    inss = []
    instrs = f[1].strip()
    s = "4" # hard-coded
    i = 1
    while i < len(instrs):
        c = instrs[i]
        if c == "L" or c == "R":
            inss.append(int(s))
            s = ""
            inss.append(c)
        else:
            s += c
        i += 1
    inss.append(int(s))

    for i in range(len(inss)):
        ins = inss[i]
        move(ins)
    
    return 1000*(posrow+1) + 4*(poscol+1) + curdir


def day23():
    f = open('aoc23.txt', 'r').read().strip().split("\n")
    grid = []
    for row in f:
        grid.append(list(row))

    elves = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "#":
                elves.add((i,j))
    
    # (row, col)
    ncheck = [(-1,0), (-1,1), (-1,-1)]
    scheck = [(1,0),(1,1),(1,-1)]
    wcheck = [(0,-1),(-1,-1),(1,-1)]
    echeck = [(0,1),(-1,1),(1,1)]
    north = (-1,0)
    south = (1,0)
    west = (0,-1)
    east = (0,1)
    checks = {"n":ncheck,"s":scheck,"w":wcheck,"e":echeck}
    dirs = {"n":north,"s":south,"w":west,"e":east}
    dirsorder = "nswe"
    allnbrs = [(1,0),(1,1),(1,-1),(-1,0),(-1,-1),(-1,1),(0,1),(0,-1)]

    def move():
        nonlocal dirsorder
        nonlocal checks
        nonlocal dirs
        nonlocal grid
        nonlocal elves
        nonlocal allnbrs

        anymoved = False
        moves = {}
        movescount = {}
        for elf in elves:
            row = elf[0]
            col = elf[1]

            nbrs = [(c[0]+row, c[1]+col) for c in allnbrs]
            if all(nbr not in elves for nbr in nbrs):
                # nobody in neighbors
                moves[elf] = elf
                movescount[elf] = 1
                continue
            
            setted = False
            for i in range(4):
                d = dirsorder[i]
                
                check = checks[d]
                dir = dirs[d]
                nbrs = [(c[0]+row, c[1]+col) for c in check]
                if all(nbr not in elves for nbr in nbrs):
                    newelf = (row+dir[0], col+dir[1])
                    moves[elf] = newelf
                    if newelf in movescount:
                        movescount[newelf] += 1
                    else:
                        movescount[newelf] = 1
                    
                    setted = True
                    break
            if not setted:
                # no valid moves
                moves[elf] = elf
                movescount[elf] = 1
        
        for elf in moves:
            if movescount[moves[elf]] > 1:
                continue
            else:
                if elf != moves[elf]:
                    anymoved = True
                    elves.remove(elf)
                    elves.add(moves[elf])

        dirsorder = dirsorder[1:] + dirsorder[0]

        return anymoved

    # PART 2
    count = 0
    while True:
        count += 1
        if count % 10 == 0:
            print(count)
        anymoved = move()
        if not anymoved:
            return count

    # PART 1
    # for _ in range(10):
    #     move()
    # minrow = min(i[0] for i in elves)
    # maxrow = max(i[0] for i in elves)
    # mincol = min(i[1] for i in elves)
    # maxcol = max(i[1] for i in elves)

    # print(minrow,maxrow,mincol,maxcol,elves)
    # return (maxcol-mincol+1)*(maxrow-minrow+1) - len(elves)


print(day23())
