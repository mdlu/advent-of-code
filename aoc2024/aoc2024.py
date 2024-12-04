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

print(day4())
