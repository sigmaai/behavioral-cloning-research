
import cPickle as pickle


class TrainHistory():

    def __init__(self, model_type, iteration):

        self.model_type = model_type
        self.iteration = iteration

    def save_history(self, path=None):

        if path:
            with open(path, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_history(path=None):
        with open(path, 'rb') as input:
            return pickle.load(input)