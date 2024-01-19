import random


def swap(arr, i, j):
    """
    Swaps the elements at indices i and j in the list
    """
    arr[i], arr[j] = arr[j], arr[i]


def partition(arr, lo, hi):
    """
    Partitions the list around the pivot and returns the index of the pivot:
        - arr: the list to partition
        - lo: the lower bound of the list
        - hi: the upper bound of the list
    """
    # Question constraint: the pivot is always the second element in the list
    swap(arr, lo, lo + 1)
    pivot = arr[lo]

    i, j = lo + 1, hi

    while True:
        while arr[i] < pivot and i < hi:
            i += 1

        while arr[j] > pivot and j > lo:
            j -= 1

        if i >= j:
            break

        swap(arr, i, j)
        i += 1
        j -= 1

    swap(arr, lo, j)
    return j


def quicksort(arr, lo, hi):
    """
    Sorts the list in-place using quicksort:
        - arr: the list to sort
        - lo: the lower bound of the list
        - hi: the upper bound of the list
    """
    if lo >= hi:
        return

    pivot_idx = partition(arr, lo, hi)
    quicksort(arr, lo, pivot_idx - 1)  # Lower half of the list
    quicksort(arr, pivot_idx + 1, hi)  # Upper half of the list


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
    quicksort(arr, 0, len(arr) - 1)

    return arr

print(sortArray([5,2,3,1]))