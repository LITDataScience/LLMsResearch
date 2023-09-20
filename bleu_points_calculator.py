import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download the NLTK data if you haven't already
nltk.download("punkt")
nltk.download("perluniprops")


def calculate_bleu(reference, candidate):
    # Tokenize the reference and candidate sentences
    reference_sentence = reference.split()
    candidate_sentence = candidate.split()

    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_sentence], candidate_sentence, smoothing_function=smoothing)

    return bleu_score


# Example usage
reference = "My name is Rudra"
candidate = "My name is Rudra"

bleu_score = calculate_bleu(reference, candidate)
print("BLEU Score:", bleu_score)
