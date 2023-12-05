import re
from typing import no_type_check

def day1():
    f = open('aoc1.txt', 'r').read().strip().split("\n")
    f = [int(i) for i in f]
    count = 0
    for i in range(len(f)-3):
        if sum(f[i:i+3]) < sum(f[i+1:i+4]):
            count +=1

    return count



def day2():
    f = open('aoc2.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    x = 0
    y = 0
    aim = 0
    for l in f:
        (d, n) = l.split(" ")
        if d == "forward":
            x += int(n)
            y += aim*int(n)
        elif d == "up":
            aim -= int(n)
        elif d == "down":
            aim += int(n)
    return x*y

def day3():
    f2 = open('aoc3.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # print(len(f[0])) 
    beta = ""
    gamma = ""

    f = f2.copy()
    for i in range(12):
        ones = []
        zeros = []
        count1 = 0
        for j in range(len(f)):
            if f[j][i] == '1':
                count1 += 1
                ones.append(f[j])
            else:
                zeros.append(f[j])
        count0 = len(f) - count1
        if count1 >= count0:
            f = ones
        else:
            f = zeros
            
    print(f)
    beta = f[0]
    
    f2 = open('aoc3.txt', 'r').read().strip().split("\n")
    f = f2.copy()
    for i in range(12):
        if len(f) == 1:
            break
        ones = []
        zeros = []
        count1 = 0
        for j in range(len(f)):
            if f[j][i] == '1':
                count1 += 1
                ones.append(f[j])
            else:
                zeros.append(f[j])
        count0 = len(f) - count1
        if count1 < count0:
            f = ones
        else:
            f = zeros
    
    print(f)
    gamma = f[0]

    beta = int(beta, 2)
    gamma = int(gamma, 2)
    return beta*gamma


def checkBingo(board):
    for row in board:
        if all([i<0 for i in row]):
            return True
    for i in range(len(board[0])):
        col = [row[i] for row in board]
        if all([i<0 for i in col]):
            return True
    return False
            
def day4():
    f = open('aoc4.txt', 'r').read().strip().split("\n\n")
    nums = f[0]
    nums = [int(i) for i in nums.split(",")]

    boards = f[1:]
    bs = []
    for b in boards:
        board = []
        s = b.split("\n")
        for row in s:
            board.append([int(i) for i in row.split()])
        bs.append(board)
    # print(bs[0])
    for num in nums:
        for b in bs:
            for i in range(len(b)):
                for j in range(len(b[0])):
                    if b[i][j] == num:
                        b[i][j] *= -1

        nextbs = []
        for b in bs:
            if not checkBingo(b):
                nextbs.append(b)
            elif len(bs) == 1:
                total = 0
                for row in b:
                    for elem in row:
                        if elem > 0:
                            total += elem
                return total*num
        bs = nextbs
    return None

    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    total = 0
    # for e in f:
    #     pass
    # for i in range(len(f)):
    #     e = f[i]

    return total


def day5():
    f = open('aoc5.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    f = [re.split(' -> |,', i) for i in f]
    counts = {}
    for r in f:
        row = [int(i) for i in r]
        if row[0] == row[2]:
            for i in range(min(row[1], row[3]), max(row[1], row[3])+1):
                if (row[0], i) in counts:
                    counts[(row[0], i)] += 1
                else:
                    counts[(row[0], i)] = 1
        elif row[1] == row[3]:
            for i in range(min(row[0], row[2]), max(row[0], row[2])+1):
                if (i, row[1]) in counts:
                    counts[(i, row[1])] += 1
                else:
                    counts[(i, row[1])] = 1
        else:
            if row[0] < row[2]:
                x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            else:
                x1, y1, x2, y2 = row[2], row[3], row[0], row[1]

            if y1 < y2:
                for i in range(x2-x1+1):
                    if (x1+i, y1+i) in counts:
                        counts[(x1+i, y1+i)] += 1
                    else:
                        counts[(x1+i, y1+i)] = 1
            else:
                for i in range(x2-x1+1):
                    if (x1+i, y1-i) in counts:
                        counts[(x1+i, y1-i)] += 1
                    else:
                        counts[(x1+i, y1-i)] = 1
                
    total = 0
    for key in counts:
        if counts[key] > 1:
            total += 1

    return total


def day6():
    f = open('aoc6.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    fish = [int(i) for i in f[0].split(",")]
    total = 0
    # for e in f:
    #     pass
    # for i in range(len(f)):
    #     e = f[i]
    fdict = {i:0 for i in range(9)}
    for f in fish:
        fdict[f] += 1
    
    for i in range(256):
        newfdict = {i:0 for i in range(9)}
        for j in range(1, 9):
            newfdict[j-1] = fdict[j]
        newfdict[6] += fdict[0]
        newfdict[8] += fdict[0]
        fdict = newfdict
    
    return sum(fdict.values())
    
    
    # for i in range(256):
    #     newfish = []
    #     newnewfish = []
    #     for j in range(len(fish)):
    #         if fish[j] == 0:
    #             newfish.append(6)
    #             newnewfish.append(8)
    #         else:
    #             newfish.append(fish[j]-1)
    #     fish = newfish+newnewfish
    # return len(fish)

tris = {}
def tri(num):
    return num*(num+1)/2

def day7():
    f = open('aoc7.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    f = [int(i) for i in f[0].split(",")]

    best = 0
    besttotal = float("inf")
    for i in range(min(f), max(f)+1):
        total = sum([tri(abs(x-i)) for x in f])
        if total < besttotal:
            besttotal = total
            best = i

    return besttotal
    # for e in f:
    #     pass
    # for i in range(len(f)):
    #     e = f[i]


def day8():
    f = open('aoc8.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    f = [i.split(" | ") for i in f]

    total = 0
    for e in f:
        left = [sorted(i) for i in e[0].split(" ")]
        right = [sorted(i) for i in e[1].split(" ")]
        d = {i:0 for i in "abcdefg"}
        for char in e[0]:
            if char != ' ':
                d[char] += 1
        
        m = {}
        for c in "abcdefg":
            if d[c] == 6:
                m[c] = "b"
            elif d[c] == 4:
                m[c] = "e"
            elif d[c] == 9:
                m[c] = "f"

        for i in left:
            if len(i) == 2:
                for c in i:
                    if c not in m:
                        m[c] = "c"
                        break
        for c in "abcdefg":
            if d[c] == 8 and c not in m:
                m[c] = "a"

        for i in left:
            if len(i) == 4:
                for c in i:
                    if c not in m:
                        m[c] = "d"
                        break
        for c in "abcdefg":
            if d[c] == 7 and c not in m:
                m[c] = "g"
        
        nums = {"abcefg":0, "cf":1, "acdeg":2, "acdfg":3, "bcdf":4, "abdfg":5, "abdefg":6, "acf":7, "abcdefg":8, "abcdfg":9}
        
        number = 0
        for n in right:
            s = ""
            for c in n:
                s += m[c]
            # if nums[''.join(sorted(s))] in [1, 4, 7, 8]:
            #     total += 1
            number = 10*number+nums[''.join(sorted(s))]
        total += number

    return total


def day9():
    f = open('aoc9.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    total = 0
    lows = set()
    rows = []
    for i in f:
        rows.append([int(r) for r in i])
    width = len(rows)
    height = len(rows[0])
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            neighbors = [n for n in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)] if n[0] >=0 and n[0] < width and n[1] >= 0 and n[1] < height]
            count = 0
            for n in neighbors:
                if rows[i][j] < rows[n[0]][n[1]]:
                    count += 1
            if count == len(neighbors):
                total += rows[i][j]+1
                lows.add((i,j))
    basins = []
    for low in lows:
        basin = set()
        queue = [low]
        while len(queue) > 0:
            (i,j) = queue.pop()
            neighbors = [n for n in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)] if n not in basin and n[0] >=0 and n[0] < width and n[1] >= 0 and n[1] < height and rows[n[0]][n[1]]!=9]
            queue.extend(neighbors)
            basin.add((i,j))

        basins.append(len(basin))
    return sorted(basins)[-3:]


def day10():
    f = open('aoc10.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    total = 0
    m = {')':3, ']':57,'}':1197,'>':25137}
    valids = []
    for e in f:
        stack = []
        valid = True
        for c in e:
            if c in ['{','(','<','[']:
                stack.append(c)
            elif c == '}':
                if stack[-1] == '{':
                    del stack[-1]
                else:
                    total += m[c]
                    valid = False
                    break
            elif c == ']':
                if stack[-1] == '[':
                    del stack[-1]
                else:
                    total += m[c]
                    valid = False
                    break
            elif c == '>':
                if stack[-1] == '<':
                    del stack[-1]
                else:
                    total += m[c]
                    valid = False
                    break
            elif c == ')':
                if stack[-1] == '(':
                    del stack[-1]
                else:
                    total += m[c]
                    valid = False
                    break
        if valid:
            valids.append(stack)
    
    scores = []
    m = {'(':1,'[':2,'{':3,'<':4}
    for v in valids:
        x = 0
        s = v[::-1]
        for c in s:
            x *= 5
            x += m[c]
        scores.append(x)
    return sorted(scores)[len(scores)//2]



def day11():
    f = open('aoc11.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    total = 0
    r = []
    for e in f:
        row = [int(i) for i in e]
        r.append(row)
    x = len(r)
    y = len(r[0])
    def nbrs(i,j):
        ns = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
        return [n for n in ns if n[0] >= 0 and n[0] < x and n[1] >= 0 and n[1] < y]

    count = 0
    while True:
        count += 1
        queue = set()
        flashes = set()
        for i in range(len(r)):
            for j in range(len(r[0])):
                r[i][j] += 1
                if r[i][j] == 10:
                    queue.add((i,j))

        while len(queue) > 0:
            z = queue.pop()
            flashes.add(z)
            for n in nbrs(z[0],z[1]):
                r[n[0]][n[1]] += 1
                if r[n[0]][n[1]] > 9 and n not in flashes:
                    queue.add(n)
        
        for f in flashes:
            r[f[0]][f[1]] = 0
        
        if len(flashes) == x*y:
            return count

def nbrs8(i,j,x,y):
    ns = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
    return [n for n in ns if n[0] >= 0 and n[0] < x and n[1] >= 0 and n[1] < y]

def nbrs4(i,j,x,y):
    ns = [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]
    return [n for n in ns if n[0] >= 0 and n[0] < x and n[1] >= 0 and n[1] < y]


def day12():
    f = open('aoc12.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    total = 0

    g = {}
    for e in f:
        s = e.split("-")
        if s[0] in g:
            g[s[0]].add(s[1])
        else:
            g[s[0]] = {s[1]}
        if s[1] in g:
            g[s[1]].add(s[0])
        else:
            g[s[1]] = {s[0]}
    
    def help(s, t, path=set(), double=False):
        count = 0
        for n in g[s]:
            if n == t:
                count += 1
            elif (n.islower() and n not in path) or n.isupper():
                count += help(n, t, path.union({s}), double)
            elif (n.islower() and not double and n in path and n not in ["start", "end"]):
                count += help(n,t,path.union({s}),True)
        return count
    
    total = help("start", "end")


    return total

def day13():
    f = open('aoc13.txt', 'r').read().strip().split("\n\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    points = f[0].split("\n")
    folds = f[1].split("\n")
    folds = [i[10:].strip() for i in folds]
    total = 0

    a = set()
    for p in points:
        ps = p.split(",")
        x,y = int(ps[0]), int(ps[1])
        a.add((x,y))
    
    for fold in folds:
        newa = set()
        s = fold.split("=")
        d, val = s[0], int(s[1])
        for p in a:
            x,y = p[0], p[1]
            if d == 'x':
                if x == val:
                    continue
                elif x < val:
                    newa.add((x,y))
                else:
                    newa.add((val-(x-val), y))
            elif d == 'y':
                if y == val:
                    continue
                elif y < val:
                    newa.add((x,y))
                else:
                    newa.add((x,(val-(y-val))))
        a = newa
    print(a)
    x = max(a, key=lambda x: x[0])
    y = max(a, key=lambda y: y[1])
    grid = []
    for i in range(50):
        grid.append([" "]*50)
    # for i in range(x[0]):
    #     l = [0]*y[1]
    #     grid.append(l)
    for p in a:
        grid[p[1]][p[0]] = "O"
    for row in grid:
        print(''.join(row))
    # print(grid)
    return len(a)

def day14():
    f = open('aoc14.txt', 'r').read().strip().split("\n\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]
    start = f[0]
    ins = f[1].split("\n")

    maps = {}
    for i in ins:
        s = i.split(" -> ")
        maps[s[0]] = s[1]
    
    pairs = {}
    for j in range(len(start)-1):
        if start[j:j+2] in pairs:
            pairs[start[j:j+2]] += 1
        else:
            pairs[start[j:j+2]] = 1

    for _ in range(40):
        newpairs = {}
        for p in pairs:
            if p in maps:
                left, right = p[0] + maps[p], maps[p] + p[1]
                for x in [left, right]:
                    if x in newpairs:
                        newpairs[x] += pairs[p]
                    else:
                        newpairs[x] = pairs[p]
            else:
                if p in newpairs:
                    newpairs[p] += pairs[p]
                else:
                    newpairs[p] = pairs[p]
        pairs = newpairs.copy()
    
    counts = {}
    for p in pairs:
        if p[0] in counts:
            counts[p[0]] += pairs[p]
        else:
            counts[p[0]] = pairs[p]
        if p[1] in counts:
            counts[p[1]] += pairs[p]
        else:
            counts[p[1]] = pairs[p]
    counts[start[0]] += 1
    counts[start[-1]] += 1

    return (max(counts.values()) - min(counts.values()))/2
        

    # for _ in range(40):
    #     newstart = ""
    #     for j in range(len(start)-1):
    #         if start[j:j+2] in maps:
    #             newstart += start[j] + maps[start[j:j+2]]
    #         else:
    #             newstart += start[j]
    #     newstart += start[-1]
    #     start = newstart
    
    counts = {}
    for s in start:
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
    
    return max(counts.values()) - min(counts.values())
            
import heapq
def day15():
    f = open('aoc15.txt', 'r').read().strip().split("\n")
    x = []
    for e in f:
        row = [int(i) for i in e]
        newrow = row + [i%9+1 for i in row] + [(i+1)%9+1 for i in row] + [(i+2)%9+1 for i in row] + [(i+3)%9+1 for i in row]
        x.append(newrow)
    
    m = []
    for i in range(5):
        for row in x:
            m.append([(p+i-1)%9+1 for p in row])
    
    m[0][0] = 0

    ds = {(i,j): float("inf") for i in range(len(m)) for j in range(len(m[0]))}
    ds[(0,0)] = 0
    unvisited = {(i,j) for i in range(len(m)) for j in range(len(m[0]))}
    q = [(0,(0,0))]

    while len(q) > 0:
        dist, p = heapq.heappop(q)
        if p not in unvisited:
            continue
        
        unvisited.remove(p)

        ns = [i for i in nbrs4(p[0], p[1], len(m), len(m[0])) if i in unvisited]
        for n in ns:
            if ds[p] + m[n[0]][n[1]] < ds[n]:
                ds[n] = ds[p] + m[n[0]][n[1]]
                heapq.heappush(q, (ds[n], n))
    
    return ds[(len(m)-1,len(m)-1)]

import math
from functools import reduce
def day16():
    f = open('aoc16.txt', 'r').read().strip().split("\n")
    i = 0

    m = {"0":"0000","1":"0001","2":"0010","3":"0011","4":"0100","5":"0101","6":"0110","7":"0111","8":"1000","9":"1001","A":"1010","B":"1011","C":"1100","D":"1101","E":"1110","F":"1111"}
    s = ''.join([m[e] for e in f[0]])
    versions = []

    def evaluate(vals, ptype):
        if ptype == 0:
            ans = sum(vals)
        elif ptype == 1:
            ans = reduce((lambda x,y: x*y), vals)
        elif ptype == 2:
            ans = min(vals)
        elif ptype == 3:
            ans = max(vals)
        elif ptype == 5:
            ans = (vals[0] > vals[1])
        elif ptype == 6:
            ans = (vals[0] < vals[1])
        elif ptype == 7:
            ans = (vals[0] == vals[1])
        return ans

    def findLiteral(i):
        ans = ""
        while True:
            ans += s[i+1:i+5]
            i += 5
            if s[i-5] == "0":
                break
            
        return int(ans,2), i
    
    def findZero(i,ptype):
        length = int(s[i:i+15],2)

        i += 15
        total = 0
        vals = []
        while total < length:
            val, newi = decode(i)
            vals.append(val)
            total += (newi-i)
            i = newi
        
        ans = evaluate(vals, ptype)
        return ans, i
    
    def findOne(i,ptype):
        numPackets = int(s[i:i+11],2)
        i += 11
        vals = []
        for _ in range(numPackets):
            val, i = decode(i)
            vals.append(val)
        ans = evaluate(vals, ptype)
        return ans, i

    def decode(i):
        p = s[i:i+7]
        i += 7

        version = int(p[:3],2)
        versions.append(version)

        ptype = int(p[3:6],2)
        if ptype == 4:
            val, i = findLiteral(i-1)

        else:
            ltype = p[6]
            if ltype == "0":
                val, i = findZero(i,ptype)
            else:
                val, i = findOne(i,ptype)
        
        return val, i

    while i < len(s)-7:
        val, i = decode(i)
        return val
        i = math.ceil(i / 4)*4

    return sum(versions)

def day17():
    # f = open('aoc17.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]

    total = 0
    xmin = 253
    xmax = 280
    ymin = -73
    ymax = -46

    def travel(i,j):
        x = 0
        y = 0
        maxy = 0
        while True:
            x += i
            y += j
            if y > maxy:
                maxy = y
            if i < 0:
                i += 1
            elif i > 0:
                i -= 1
            j -= 1
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return maxy, True
            if x > xmax or y < ymin:
                return None, False
    
    ans = 0
    count = 0
    for i in range(22,281):
        for j in range(-73,100):
            my, good = travel(i,j)
            if good:
                count += 1
                print(count)
            if good and my > ans:
                ans = my
                # print(ans)
                # print(i,j)
            # print(i,j)
    return ans


def day18():
    f = open('aoc18.txt', 'r').read().strip().split("\n")
    # f = [int(i) for i in f]
    # f = [i.split(" ") for i in f]
    # f = [int(i) for i in f[0].split(",")]

    total = 0
    
    # for e in f:
    #     pass
    # for i in range(len(f)):
    #     e = f[i]

    return total
    

print(day18())

# def day4():
#     f = open('aoc4.txt', 'r').read().strip().split("\n")
#     # f = [int(i) for i in f]
#     # f = [i.split(" ") for i in f]
#     # f = [int(i) for i in f[0].split(",")]
#     total = 0
#     # for e in f:
#     #     pass
#     # for i in range(len(f)):
#     #     e = f[i]

#     return total