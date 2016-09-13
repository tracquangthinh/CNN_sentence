import gensim
class LabeledLineSentence(object):
  def __init__(self, pathCorpus):
    self.pathCorpus = pathCorpus
  def __iter__(self):
    for uid, line in enumerate(open(self.pathCorpus)):
      # objLine = line.split(":")
      yield gensim.models.doc2vec.LabeledSentence(words=line.split(),
        tags=['SENT_%s' % uid])
