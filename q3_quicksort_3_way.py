# Source: Sedgewick, R. and Wayne, K. (2011). Algorithms. 4th ed. Addison-Wesley Professional, p.299.

import random


def swap(arr, i, j):
    """
    Swaps the elements at indices i and j in the list
    """
    arr[i], arr[j] = arr[j], arr[i]


def quicksort_3_way(arr, lo, hi):
    """
    Sorts the list in-place using quicksort:
        - arr: the list to sort
        - lo: the lower bound of the list
        - hi: the upper bound of the list
    """
    if hi <= lo:
        return

    lt, i, gt = lo, lo + 1, hi
    pivot = arr[lo]

    while i <= gt:
        if arr[i] < pivot:
            swap(arr, lt, i)
            lt += 1
            i += 1
        elif arr[i] > pivot:
            swap(arr, i, gt)
            gt -= 1
        else:
            i += 1

    quicksort_3_way(arr, lo, lt - 1)
    quicksort_3_way(arr, gt + 1, hi)


def shuffle(arr):
    """
    Randomly shuffles the list in-place to guarantee O(nlogn) time complexity regardless of the input
    """
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i)
        swap(arr, i, j)


def sortArray(arr):
    """
    Sorts a list using quicksort:
        - arr: the list to sort
    Shuffling the list randomizes pivot selection at the i-th position, guaranteeing O(nlogn) time complexity
    """
    shuffle(arr)
    quicksort_3_way(arr, 0, len(arr) - 1)

    return arr
