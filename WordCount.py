from collections import defaultdict

# Simulated Mapper
def mapper(line):
    for word in line.strip().split():
        yield (word.lower(), 1)

# Simulated Reducer
def reducer(pairs):
    word_count = defaultdict(int)
    for word, count in pairs:
        word_count[word] += count
    return word_count

# Read input file and apply map-reduce
def map_reduce_word_count(filename):
    intermediate_pairs = []

    with open(filename, 'r') as file:
        for line in file:
            for pair in mapper(line):
                intermediate_pairs.append(pair)

    result = reducer(intermediate_pairs)
    return result

# Example usage
if __name__ == "__main__":
    filename = "sample.txt"  # Replace with your file
    word_counts = map_reduce_word_count(filename)
    
    for word, count in sorted(word_counts.items()):
        print(f"{word}: {count}")
