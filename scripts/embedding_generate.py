import os
import tqdm
import numpy as np
import argparse


def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    print("words:", len(words))
    return vocab_dict


def generate_emb(dataset):
    glove_file = '../data/glove.840B.300d.txt'
    vocab_file = '../data/vocabulary_spacy_{}.txt'.format(dataset)
    npy_file = '../data/{}_spacy_emb.npy'.format(dataset)

    vocab_dict = load_vocab_dict_from_file(vocab_file)
    vocab_size = 8226 if dataset == 'referit' else 21692
    emb_size = 300
    emb_dict = {}

    with open(glove_file, 'r') as glo:
        print("Loading Glove File...")
        all_emb = glo.readlines()

    print("Loaded Gloves.")

    for line in tqdm.tqdm(all_emb):
        items = line.strip().split(' ')
        word, emb = items[0], items[1:]
        if word in vocab_dict.keys():
            emb_dict[word] = np.array(emb, np.float32)

    # build vocab npy
    vocab_emb_array = np.zeros((vocab_size, emb_size), dtype=np.float32)
    print("shape:", vocab_emb_array.shape)

    no = open('not_exist.txt', 'w')
    for w, idx in vocab_dict.items():
        if w in emb_dict.keys():
            vocab_emb_array[idx, :] =emb_dict[w]
        else:
            no.write(w+'\n')
            vocab_emb_array[idx, :] = np.random.normal(scale=0.6, size=(emb_size, ))
    no.close()
    np.save(npy_file, vocab_emb_array)
    print("Saving to npy file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='referit')  # or 'Gref'
    args = parser.parse_args()
    generate_emb(args.d)
