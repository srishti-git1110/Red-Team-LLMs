# section 4.2 of https://arxiv.org/pdf/2302.08582.pdf
import re
import argparse
import Optional
import numpy as np
from srsly import read_jsonl
from pydantic import BaseModel

from prompts import get_redlm_prompt_template
from scorers import DetoxifyToxicityScorer
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
  
  def add(self, prompt: CandidatePrompt):
    if prompt.text in self.prompts:
      self.prompts[prompt.text].scores.extend(prompt.scores)
    else:
      self.prompts[prompt.text] = prompt

  def clear(self):
    self.prompts.clear()

  def sample(self, k=4):
    weights = [np.exp(np.mean(prompt.scores)/self.beta) for prompt in self.prompts.values()]
    sampled_prompts = np.random.choice(
      list(self.prompts.values()),
      size=k,
      replace=False,
      p=np.array(weights)/sum(weights)
    )

    return list(set(sampled_prompts))
  
  def current_best(self, n=1):
    return sorted(self.prompts.values(), key=lambda prompt: np.mean(prompt.scores), reverse=True)[:n]

  def current_mean(self):
    return np.mean([np.mean(prompt.scores) for prompt in self.prompts.values()])
  
  def __iter__(self):
    return iter(self.prompts.values())


def parse_response(
  num_responses_to_extract: int, 
  response: str
) -> list[str]:
  response = response.split(sep='\n')[:num_responses_to_extract]
  response = [re.sub(r'\d+\.\s', '', line).lstrip() for line in response]

  return list(set(response))


def construct_prompt_for_redlm(
  few_shot_examples: list(CandidatePrompt), 
  prompt_template:str
) -> str:
  '''
  Creates a prompt with an instruction and (few-shot) examples. 
  This prompt is used to sample from the red LM the adversarial prompts that are inturn used to elicit adversarial behaviour from the target lm.
  '''
  examples = ''
  for i, example in enumerate(few_shot_examples):
    examples += f'{i+1}. {example.text}/n'
  examples += f'{i+2}. '

  return prompt_template.format(examples=examples)

  
def get_candidates_from_redlm(
  prompt: str, 
  n: int, 
  top_p: float, 
  temperature: float
) -> list[str]:
  '''
  Few-shot-generates adversarial prompts from the red LM using the prompt returned by construct_prompt_for_redlm.
  '''
  

def generate_completions_from_targetlm(
  target_lm: TextGenerationPipeline,
  candidates: list[str],
  continuations_per_prompt: int = 512,
  prefix: str = '',
  bad_word_ids: Optional[list[int]] = None,
) -> dict[str, list[str]]:
  '''
  Tries to elicit harmful behaviour from the target model given an adversarial prompt returned by get_candidates_from_redlm
  '''
  target_lm.tokenizer.pad_token_id = target_lm.tokenizer.bos_token_id
  target_lm.tokenizer.padding_side = 'left'
  batch_size = 512

  output = target_lm(
    candidates,
    prefix=target_lm.tokenizer.bos_token+prefix,
    eos_token_id=target_lm.tokenizer.bos_token_id,
    min_length=10,
    max_length=64,
    top_p=0.9,
    temperature=0.7,
    do_sample=True,
    bad_words_ids=bad_word_ids,
    batch_size=int(batch_size/continuations_per_prompt),
    num_workers=0,
    num_return_sequences=continuations_per_prompt,
  )
  
  return {
    prompt: [completion['generated_text'].removeprefix(prompt).replace(prefix, '') for completion in completions]
    for prompt, completions in zip(candidates, output)
  }


def red_team(args: argparse.Namespace):
  target_lm = pipeline('text-generation', model=args.target_lm, device=args.device)

  
  
  
  

