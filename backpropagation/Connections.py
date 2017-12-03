# Connections is the set object of Connection
# it support set operations on Connection
from Connection import Connection


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, conn):
        self.connections.append(conn)

    def dump(self):
        '''
        print info of this connections
        :return: 
        '''
        for conn in self.connections:
            print conn