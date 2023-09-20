import re
import collections
import numpy as np


def tokenize(text):
    # Tokenize the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def ngram_counts(tokens, n):
    # Generate n-gram counts from a list of tokens
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return collections.Counter(ngrams)


def chrF_plus_plus(reference, candidate, max_n=5, beta=3.0):
    # Tokenize the reference and candidate translations
    ref_tokens = tokenize(reference)
    can_tokens = tokenize(candidate)

    # Compute n-gram counts for reference and candidate
    ref_counts = {n: ngram_counts(ref_tokens, n) for n in range(1, max_n + 1)}
    can_counts = {n: ngram_counts(can_tokens, n) for n in range(1, max_n + 1)}

    # Calculate the harmonic mean of precision and recall
    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        ref_count = ref_counts[n]
        can_count = can_counts[n]

        common_count = sum((ref_count & can_count).values())

        if common_count == 0 or sum(can_count.values()) == 0 or sum(ref_count.values()) == 0:
            prec = 0.0
            rec = 0.0
        else:
            prec = common_count / sum(can_count.values())
            rec = common_count / sum(ref_count.values())

        precisions.append(prec)
        recalls.append(rec)

    # Calculate the F_beta score
    num_precisions = len(precisions)
    f_beta = ((1 + beta ** 2) * np.prod(precisions) * np.prod(recalls)) / (
            np.prod(precisions) + beta ** 2 * np.prod(recalls))

    return f_beta


# Sample usage
reference = "Hello, my name is Rudra"
candidate = "Hello, my name is Rudra"

chrF_plus_plus_score = chrF_plus_plus(reference, candidate)
print("chrF++ Score: ", chrF_plus_plus_score)
