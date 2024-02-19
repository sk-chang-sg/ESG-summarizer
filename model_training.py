import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import transformers

from abc import ABC, abstractmethod


class LLMBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define on which transformer to use on each model
    @abstractmethod
    def model_origin(self) -> None:
        pass


class SumSeqModel(LLMBase, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def model_origin(self) -> None:
        return super().model_origin()
