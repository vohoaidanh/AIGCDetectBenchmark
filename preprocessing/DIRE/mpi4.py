# -*- coding: utf-8 -*-


class COMM_WORLD:
    def __init__(self, size = 1, rank = 0):
        self.size = size
        self.rank = rank
        
    def Get_rank(self):
        return self.rank

class MPI:
    def __init__(self, size = 1, rank = 0):
        self.COMM_WORLD = COMM_WORLD(size, rank)



MPI = MPI()

MPI.COMM_WORLD.size
