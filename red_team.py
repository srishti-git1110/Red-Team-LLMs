from pydantic import BaseModel
from transformers import pipeline, TextGenerationPipeline
import argparse

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
  

