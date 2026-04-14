from abc import ABC, abstractmethod

# This make rach retrieval apply the same function
class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, question: str, top_k: int):
        pass