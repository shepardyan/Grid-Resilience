class PFC:
    def index(self):
        ...

    def __init__(self, base, bus, gen, branch):
        self.baseMVA = base
        self.bus = bus
        self.gen = gen
        self.branch = branch
