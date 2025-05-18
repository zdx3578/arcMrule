import abc
class BaseExtractor(metaclass=abc.ABCMeta):
    name: str = ''
    @abc.abstractmethod
    def extract(self, grid: 'np.ndarray') -> list[str]:
        """Return list of Prolog fact strings"""
    @staticmethod
    def fact(name: str, *args) -> str:
        argstr = ','.join(map(str, args))
        return f"{name}({argstr})."