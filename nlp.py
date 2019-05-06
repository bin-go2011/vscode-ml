#%%
import numpy as np

sentence = """Thomas Jefferson began building Monticello at the
age of 26."""

print(sentence)

#%%
token_sequence = sentence.split()
vocab = sorted(set(token_sequence))

#%%
num_tokens = len(token_sequence)
print("num_tokens", num_tokens)
vocab_size = len(vocab)
print("vocab", vocab_size)

#%%
onehot_vectors = np.zeros((num_tokens, vocab_size), int)
for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1
onehot_vectors

#%%
import pandas as pd
pd.DataFrame(onehot_vectors, columns=vocab)
