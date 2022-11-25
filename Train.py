import os
import gensim
from typing import List
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Preproccesser:

    @staticmethod
    def preprocess_doc(filename: str, batch_size: int, regex: str = None):
        with open(filename) as f:
            for i, line in enumerate(f.readlines()):
                if i == batch_size:
                    break
                yield list(gensim.utils.tokenize(line, lower=True))


class Train:
    def __init__(self, sentences: List[str], embedding_size: int, window_size: int, min_word_count: int,
                 number_of_workers: int, number_of_epochs: int, useSG: bool):

        self.model_type = None
        self.sentences = sentences
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.min_word_count = min_word_count
        self.number_of_epochs = number_of_epochs
        self.useSG = useSG

        if 0 < number_of_workers <= os.cpu_count():
            self.number_of_workers = number_of_workers
        elif number_of_workers == -1:
            self.number_of_workers = os.cpu_count()
        else:
            raise IndexError("Number of workers value out of range: expected -1 or positive number less than maximum "
                             "number of threads")

        self.model = None

    def get_model(self):
        return self.model

    def set_model_type(self, model_type):
        self.model_type = model_type

    def train(self, model_type: int):
        if model_type == 0:
            self.model = Word2Vec(
                self.sentences,
                vector_size=self.embedding_size,
                window=self.window_size,
                min_count=self.min_word_count,
                workers=self.number_of_workers,
                epochs=self.number_of_epochs,
                sg=self.useSG)

        elif model_type == 1:
            self.model = FastText(
                self.sentences,
                vector_size=self.embedding_size,
                window=self.window_size,
                min_count=self.min_word_count,
                workers=self.number_of_workers,
                epochs=self.number_of_epochs,
                sg=self.useSG
            )

        elif model_type == 2:
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.sentences)]
            self.model = Doc2Vec(
                documents,
                vector_size=self.embedding_size,
                window=self.window_size,
                min_count=self.min_word_count,
                workers=self.number_of_workers,
                epochs=self.number_of_epochs,
                sg=self.useSG
            )

    def save(self, filename: str, saveJustWV: bool = False):
        if self.model is None:
            return False, "Model is None"

        if saveJustWV:
            self.model.wv.save(filename)
        else:
            self.model.save(filename)

        return True, None
