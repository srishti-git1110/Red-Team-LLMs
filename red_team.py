# section 4.2 of https://arxiv.org/pdf/2302.08582.pdf
import math
import argparse
import Optional
from srsly import read_jsonl
from pydantic import BaseModel

import wandb
from transformers import pipeline, TextGenerationPipeline

class CandidatePrompt(BaseModel):
  '''
  Represents a single (adversarial) prompt.

  Attributes:
    text: str - the textual prompt
    scores: list[float] - list of scores for each token in the prompt [-R(xi)]
    own_score: Optional[float] - 
  '''
  text: str
  scores: list[float]
  own_score: Optional[float] = None

  def mean(self):
    return sum(self.scores) / len(self.scores)

  def std(self):
    return (sum((score - self.mean()) ** 2 for score in self.scores) / len(self.scores)) ** 0.5

  def __str__(self):
    return f'{self.text[:40]} ({self.mean():.3f} ± {self.std():.3f})'

  # figure out hashing based on self.text


class PromptPool(BaseModel):
  '''
  Represents the prompt pool P.

  Attributes:
    prompts: dict[str, CandidatePrompt] - dict storing the prompt pool P. 
    beta: float - scaling parameter used to calculate prompt weights (step 2 of the algo)
  '''
  prompts: dict[str, CandidatePrompt]
  beta: float = 1.0

  @classmethod
  def from_json(cls, path:str, limit:int = 20, **kwargs):
    prompts = {
      prompt['text']: CandidatePrompt(text=prompt['text'], scores=[1e-3])
      for prompt in list(read_jsonl(path))[:limit]
    }

    return cls(prompts=prompts, **kwargs)



  
  

