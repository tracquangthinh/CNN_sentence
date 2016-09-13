import gensim, logging, os
from labeledlinesentence import LabeledLineSentence
class Doc2Vec:
  def __init__(self, pathCorpus, pathVector, sizeVector, window, min_count):
    self.pathCorpus = pathCorpus
    self.pathVector = pathVector
    self.sizeVector = sizeVector
    self.window = window
    self.min_count = min_count
    self.sentences = LabeledLineSentence(self.pathCorpus)

  def make(self):
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',
      level = logging.INFO)
    model = gensim.models.Doc2Vec(self.sentences, size = self.sizeVector,
      min_count = self.min_count, window = self.window)
    model.save_word2vec_format(self.pathVector, binary = True)
