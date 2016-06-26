def quicksort(L):
    if (len(L) < 2):
        return L
    else:
        first = L[0]  # pivot
        rest = L[1:]
        lo = [x for x in rest if x < first]
        hi = [x for x in rest if x >= first]
        return quicksort(lo) + [first] + quicksort(hi)
