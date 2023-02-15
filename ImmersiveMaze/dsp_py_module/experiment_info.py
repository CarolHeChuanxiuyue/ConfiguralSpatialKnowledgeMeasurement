class ExperimentInfo:
    def __init__(self, names, values):
        self.properties = dict()
        for name, value in zip(names, values):
            self.properties[name] = value

    def ordered_keys(self, order):
        return [self.properties[x] for x in order]

    def keys(self):
        return self.properties.keys()

    def values(self):
        return self.properties.values()
