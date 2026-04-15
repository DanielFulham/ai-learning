import chromadb
from interfaces.client_interface import ClientInterface


class Client(ClientInterface):

    def get_client(self):
        return chromadb.Client()