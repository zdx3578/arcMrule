import abc
from typing import List
class BaseExtractor(metaclass=abc.ABCMeta):
    name: str = ''
    @abc.abstractmethod
    def extract(self, grid: 'np.ndarray') -> List[str]: ...
    @staticmethod
    def fact(name: str, *args) -> str:
        argstr = ','.join(map(str, args))
        return f"{name}({argstr})."
