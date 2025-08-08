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
      scores: list[float] - list of scores corresponding to completions of the adversarial prompt ai, u(ai)
      own_score: Optional[float] - candidate prompt's score
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
    def from_json(cls, path: str, limit: int = 20, **kwargs):
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
        weights = [np.exp(np.mean(prompt.scores)/self.beta)
                   for prompt in self.prompts.values()]
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
    response: str,
    num_responses_to_extract: int=1
) -> list[str]:
    response = response.split(sep='\n')[:num_responses_to_extract]
    response = [re.sub(r'\d+\.\s', '', line).lstrip() for line in response]

    return list(set(response))


def construct_prompt_for_redlm(
    few_shot_examples: list(CandidatePrompt),
    prompt_template: str
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
    redlm: TextGenerationPipeline,
    prompt: str,
    num_responses: int,
    top_p: float,
    temperature: float
) -> list[str]:
    '''
    Few-shot-generates adversarial prompts from the red LM using the prompt returned by construct_prompt_for_redlm.
    '''
    # return a list of completions only (not prompt+completion)


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
        prompt: [completion['generated_text'].removeprefix(
            prompt).replace(prefix, '') for completion in completions]
        for prompt, completions in zip(candidates, output)
    }


def red_team(args: argparse.Namespace):
    target_lm = pipeline(
		'text-generation', 
		model=args.target_lm, 
		device=args.device
	)
    redlm = pipeline(
        'text-generation', 
		model=args.redlm, 
		device=args.device
    )
    scorer = DetoxifyToxicityScorer()

    for _ in range(args.num_trials):
        wandb.init(
			project='red-teaming',
			entity='srishti-gureja-wandb',
            group=args.group_name,
			config=args
		)
        prompt_pool = PromptPool.from_json(
			path=args.prompt_pool_file,
			temperature=args.beta
		)
        local_prompt_pool = PromptPool(prompts={})

        for i in range(args.num_rounds):
            local_prompt_pool.clear()
            print('---------------------------------------------------------')
            print('---------------------------------------------------------')
            print(f'Round {i+1}, prompt pool size = {len(prompt_pool.prompts)}\n')

            few_shot_examples = prompt_pool.sample(k=4)
            redlm_prompt = construct_prompt_for_redlm(
                few_shot_examples=few_shot_examples,
                prompt_template=get_redlm_prompt_template()
            )
            print(f'Red LM Prompt: {redlm_prompt}')
            
            responses = get_candidates_from_redlm(
                redlm=redlm,
                prompt=redlm_prompt,
                num_responses=args.redlm_num_responses,
                top_p=args.redlm_top_p,
                temperature=args.redlm_temperature   
            )
            candidates = sum([parse_response(response)
                              for response in responses], [])
            
            targetlm_output = generate_completions_from_targetlm(
                target_lm=target_lm,
                candidates=candidates,
                prefix=args.prefix,
                bad_word_ids=args.bad_word_ids
            )

            for candidate_text, completions in targetlm_output.items():
                scores = scorer.score_texts(completions)
                candidate_prompt = CandidatePrompt(
                    text=candidate_text,
                    scores=scores,
                    own_score=scorer.score_text(candidate_text)
                )
                prompt_pool.add(prompt=candidate_prompt)
                local_prompt_pool.add(prompt=candidate_prompt)
                print(f'Candidate Prompt: {candidate_prompt}')


            print(f'Best prompt for round {i+1}: {local_prompt_pool.current_best()[0]}')
            print(f'Average score for round {i+1}: {local_prompt_pool.current_mean():.2f}')
            print(f'Best prompt globally: {prompt_pool.current_best()[0]}')
            print(f'Average score globally: {prompt_pool.current_mean():.2f}')

            best_prompt_table = wandb.Table(
                    data=[(prompt.text, prompt.mean(), prompt.std(), prompt.own_score)
                          for prompt in prompt_pool.current_best(n=10)],
                    columns=['text', 'mean score', 'std', 'own score']
                )
            wandb.log({
                'best_prompt': best_prompt_table,
                'best_prompts_scatter': wandb.plot.scatter(best_prompt_table, 'mean score', 'own score'),
                'target_lm_responses': wandb.Table(
                    data=[(prompt, responses[:3]) for prompt, responses in targetlm_output.items()][:10],
                    columns=['prompt', 'response']
                ),
                'best_prompt_score': prompt_pool.current_best()[0].mean(),
                'best_10_prompt_score': sum(p.mean() for p in prompt_pool.current_best(n=10))/10,
                'best_100_prompt_score': sum(p.mean() for p in prompt_pool.current_best(n=100)) / 100,
                'average_score': prompt_pool.current_mean(),
                'round_best_prompt_score': local_prompt_pool.current_best()[0].mean(),
                'round_average_score': local_prompt_pool.current_mean(),
            })
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--target_lm', type=str, default='gpt2')
    parser.add_argument('--initial_prompt_pool', type=str, default='resources/challenging_rtp.jsonl')
    parser.add_argument('--pool_temperature', type=float, default=0.1)
    parser.add_argument('--gpt3_temperature', type=float, default=1)
    parser.add_argument('--gpt3_top_p', type=float, default=1)
    parser.add_argument('--gpt3_num_responses', type=int, default=20)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--bad_words_ids', type=str, default=None)
    args = parser.parse_args()
    print(args)
    red_team(args)
            