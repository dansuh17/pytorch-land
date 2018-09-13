class NetworkTrainer:
    """Base trainer for neural net training.

    This provides a minimal set of methods that any trainer
    should implement.
    """
    def __init__(self):
        pass

    def test(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def save_model(self, filename: str):
        raise NotImplementedError

    def save_checkpoint(self, filename: str):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError
