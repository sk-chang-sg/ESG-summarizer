import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import transformers

from abc import ABC, abstractmethod


class LLMBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    # Define on which transformer to use on each model
    @abstractmethod
    def model_origin(self) -> None:
        pass


class SummurizerModel(LLMBase, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def model_origin(self) -> None:
        return super().model_origin()
