class FakeSignal:
    def __init__(self):
        pass

    def emit(self, data_tuple):
        worker_id, string = data_tuple
        print(string)
