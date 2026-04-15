from typing import Protocol

class ClientInterface(Protocol):

    def get_client(self):
        ...