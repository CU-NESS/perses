class InfiniteIndexerIterator(object):
    def __init__(self, infinite_indexer):
        self.value = infinite_indexer.value
        self.done = False
    def next(self):
        if self.done:
            raise StopIteration
        else:
            return self.value
            self.done = True

class InfiniteIndexer(object):
    def __init__(self, value):
        self.value = value
    def __getitem__(self, index):
        return self.value
    def __len__(self):
        return 1
    def __iter__(self):
        return InfiniteIndexIterator(self)

class DoubleInfiniteIndexer(InfiniteIndexer):
    def __init__(self, value):
        self.value = InfiniteIndexer(value)

