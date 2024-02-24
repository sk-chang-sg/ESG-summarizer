import numpy as np
import pandas as pd

from dotenv import dotenv_values
from abc import ABC, abstractmethod

# Lang chain packages
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

config = dotenv_values(".env")


class LLMBase(ABC):
    def __init__(self) -> None:
        super().__init__()


class SumSeqModel(LLMBase):
    def __init__(self) -> None:
        super().__init__()
        self.hf = HuggingFaceEndpoint(
            repo_id="https://api-inference.huggingface.co/models/Falconsai/text_summarization",
            huggingfacehub_api_token=config["HF_API_KEY"],
        )
        self.prompt = PromptTemplate.from_template("Summarize this text: {text}")

    def summarize(self, text_input) -> str:
        text_out = load_summarize_chain(self.hf, chain_type="map_reduce")

        return text_out.invoke({"text_input"})
