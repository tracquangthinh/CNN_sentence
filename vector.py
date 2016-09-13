from corpus import ELMCorpus
from doc2vec import Doc2Vec
class ELMVector:
  def __init__(self, pathELM, pathCorpus="corpus", pathVector="[V]element",
    sizeVector=300, windowSplit=1, windowWordToVec=3, min_count=0,
    kinase="PKA_group", nAround=3, biggerNegative=3,
    filePos="kinase.pos", fileNeg="kinase.neg"):
    self.pathELM = pathELM
    self.pathCorpus = pathCorpus
    self.pathVector = pathVector
    self.windowSplit = windowSplit
    self.windowWordToVec = windowWordToVec
    self.sizeVector = sizeVector
    self.min_count = min_count
    self.elmCorpus = ELMCorpus(self.pathELM)
    self.kinase = kinase
    self.nAround = nAround
    self.biggerNegative = biggerNegative
    # create corpus from ELM file
    self.elmCorpus.toCorpus(output=self.pathCorpus,
      window=self.windowSplit, kinase=self.kinase, biggerNegative=self.biggerNegative,
      filePos="kinase.pos", fileNeg="kinase.neg")
    # run word2vec for created corpus
    doc2vec = Doc2Vec(pathCorpus=self.pathCorpus, pathVector=self.pathVector,
      sizeVector=self.sizeVector, window=self.windowWordToVec,
      min_count=self.min_count)
    doc2vec.make()
