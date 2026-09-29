from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PolicyConfig(ABC):
    name: str = ""


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass