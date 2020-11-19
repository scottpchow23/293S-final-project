from pyserini.search import SimpleSearcher

searcher = SimpleSearcher('indexes/index-robust04-20191213/')
hits = searcher.search('hubble space telescope')

# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')