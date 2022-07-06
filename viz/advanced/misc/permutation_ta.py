import itertools
import math
from icecream import ic
import copy

import numpy as np

l = [(29, 'mengenyangkan', 0.0014),
    (8, 'terkenal', 0.0003),
    (16, 'pesan', 0.0002),
    (26, 'porsi', 0.0002),
    (0, 'lokasi', 0.0002),
    (13, 'gule', 0.0002),
    (5, 'padang', 0.0002),
    (17, 'nasi', 0.0002),
    (21, 'rendang', 0.0002),
    (19, 'padang', 0.0002),
    (12, 'kakap', 0.0002),
    (2, 'alun', 0.0001),
    (11, 'ikan', 0.0001),
    (4, 'masakan', 0.0001),
    (18, 'bungkus', 0.0001),
    (23, 'pop', 0.0001),
    (10, 'kepala', 0.0001),
    (3, 'alun', 0.0001),
    (25, 'perkedel', 0.0001),
    (22, 'ayam', 9.3827e-05),
    (20, 'berisikan', 7.0279e-05)]

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

# def least_important_words(words):
#     print(words[2])

# Driver Code
# s = {1, 2, 3}
n = int(math.floor(len(l) * 0.4))
words_perturb = l[:n]

print(words_perturb)

print(n)

# print(min(words_perturb, key = lambda t: t[2]))
def get_minimums(word_tups):
    arr = []
    for wt in word_tups:
        if wt[2] == min(words_perturb, key = lambda t: t[2])[2]:
            arr.append(wt)
    return arr


# minimum_import = get_minimums(words_perturb)
# print(minimum_import)
# # for a in findsubsets(l, n):
# #     if len(intersection(words_perturb, a)) <= n:
# #         print(a)

# def swap(A,B,i,j):
#     A[i] = B[j]
#     return A

def swap_minimum(l, words_perturb):
    def get_minimums(word_tups):
        arr = []
        for wt in word_tups:
            if wt[2] == min(words_perturb, key = lambda t: t[2])[2]:
                arr.append(wt)
        return arr
    minimum_import = get_minimums(words_perturb)
    unlisted = list(set(l).symmetric_difference(set(words_perturb)))

    len_wp = len(words_perturb)
    len_ul = len(unlisted)
    # ic(swap(words_perturb, unlisted, -1, 0))
    res = []
    for i in range(len_wp):
        if words_perturb[i] in minimum_import:
            temp_wp = list(copy.deepcopy(words_perturb))
            temp_wp.pop(i)
            swapped_wp = np.array([temp_wp for i in range(len_ul)])
            for j in range(len(swapped_wp)):
                temp_sm = np.append(swapped_wp[j], unlisted[j])
                res.append(temp_sm)
                # ic(unlisted[j])
                # swapped_wp.append(unlisted[j])
    return res

ic(words_perturb)
ic(len(swap_minimum(l, words_perturb)))

# f = [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]
# f[1].append(99)
# print(f)

# a = [1,2,3,4,5]
# b = [4,5]
# c = [7,8,9]
# r = []
# res = []
# for i in range(len(a)):
#     if a[i] in b:
#         temp_a = copy.deepcopy(a)
#         temp_a.pop(i)
#         swapped_matrix = np.array([temp_a for i in range(len(c))])
#         for j in range(len(swapped_matrix)):
#             temp_sm = np.append(swapped_matrix[j], c[j])
#             res.append(temp_sm)

# ic(res)