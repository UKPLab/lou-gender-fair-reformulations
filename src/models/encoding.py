from pandas import Series
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

class Encoding:
    def __init__(self, model_name, pooling_mode):
        self.transforming = Transformer(model_name)

        self.pooling = Pooling(
            word_embedding_dimension=self.transforming.auto_model.config.hidden_size,
            pooling_mode=pooling_mode
        )
        self.encoding = SentenceTransformer(modules=[self.transforming,self.pooling])

    def encode(self, texts):
        if type(texts) == Series:
            texts = list(texts)

        return self.encoding.encode(sentences=texts)

