from functools import reduce
from collections import defaultdict

# Define matrices
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8],
     [9, 10],
     [11, 12]]

# Map step: emit key-value pairs
def mapper(A, B):
    mapped = []

    # Map from A
    for i in range(len(A)):
        for k in range(len(A[0])):
            mapped.append(( (i, 0), ('A', k, A[i][k]) ))
            mapped.append(( (i, 1), ('A', k, A[i][k]) ))

    # Map from B
    for k in range(len(B)):
        for j in range(len(B[0])):
            mapped.append(( (0, j), ('B', k, B[k][j]) ))
            mapped.append(( (1, j), ('B', k, B[k][j]) ))

    return mapped

# Shuffle and group by key
def shuffle(mapped):
    grouped = defaultdict(list)
    for key, value in mapped:
        grouped[key].append(value)
    return grouped

# Reduce step: compute final values
def reducer(grouped):
    result = {}
    for key, values in grouped.items():
        a_vals = {k: val for tag, k, val in values if tag == 'A'}
        b_vals = {k: val for tag, k, val in values if tag == 'B'}

        # Dot product
        total = sum(a_vals.get(k, 0) * b_vals.get(k, 0) for k in range(len(B)))
        result[key] = total

    return result

# Run MapReduce
mapped = mapper(A, B)
grouped = shuffle(mapped)
reduced = reducer(grouped)

# Print Result Matrix
print("Result Matrix (C):")
C = [[reduced.get((i, j), 0) for j in range(len(B[0]))] for i in range(len(A))]
for row in C:
    print(row)
