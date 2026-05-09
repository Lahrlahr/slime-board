from collections.abc import Callable
from typing import Any

import numpy as np
import pybase64
import sglang_router
import torch
import asyncio
import copy
import inspect
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_function, trace_span
from slime.utils.types import Sample

import httpx
import requests
from patch.board.board import Board
from collections import defaultdict


class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args: Namespace = None) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint if args else '/data/huangguang/model/Qwen/Qwen2.5-1.5B-expand/', trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(8)


client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=512),
    timeout=httpx.Timeout(None),
    trust_env=False,
)
url = "http://172.17.0.37:15000/open_session"
url1 = "http://172.17.0.37:15000/generate"
url2 = "http://172.17.0.37:15000/close_session"


async def open_session():
    response = await client.post(url, json={"capacity_of_str_len": 1024}, headers=None)
    return response.json()


async def close_session(session_id):
    response = await client.post(url2, json={"session_id": session_id}, headers=None)
    return response.json()


async def generate(payload):
    response = await client.post(url1, json=payload, headers=None)
    return response.json()

uurl = "http://172.17.0.37:17000/open_session"
uurl1 = "http://172.17.0.37:17000/generate"
uurl2 = "http://172.17.0.37:17000/close_session"

async def open_session1():
    response = await client.post(uurl, json={"capacity_of_str_len": 1024}, headers=None)
    return response.json()


async def close_session1(session_id):
    response = await client.post(uurl2, json={"session_id": session_id}, headers=None)
    return response.json()


async def generate1(payload):
    response = await client.post(uurl1, json=payload, headers=None)
    return response.json()

flags_ids = [157600, 157601, 157602, 157603, 157604, 157605, 157606, 157607]
dead_ids = [157608, 157609, 157610, 157611]
result_ids = [157612, 157613, 157614, 157615]

# 0雷 1司 2军 3师 4旅 5团 6营 7连 8排 9兵 10炸 11旗
role_map = ['Mine', 'Commander', 'General', 'Division Commander', 'Brigade Commander', 'Regiment Commander',
            'Battalion Commander', 'Company Commander', 'Platoon Leader', 'Engineer', 'Bomb', 'Flag']


def generate_prompt(layout):
    role = [role_map[idx] for idx in layout]
    lower = role[:25]
    upper = role[25:]

    content = (
        "You are playing Chinese Military Chess. "
        "Your layout is divided into two parts: "
        f"The lower half consists of {', '.join(lower)}. "
        f"The upper half consists of {', '.join(upper)}."
    )
    prompt = [{"role": "user", "content": content}]

    state = GenerateState()
    prompt1 = state.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    prompt_ids = state.tokenizer.encode(prompt1, add_special_tokens=False)
    return prompt_ids


async def generate_junqi():
    state = GenerateState()
    async with state.semaphore:
        session_ids = await asyncio.gather(open_session(), open_session())

        req_id = [None, None]
        buffer = []
        idx = [0, 0]

        board = Board()
        prompt = [generate_prompt(board.get_layout(0)), generate_prompt(board.get_layout(1))]

        loss_mask = [[], []]
        logits_masks = [[], []]
        reward_list = [[], []]
        rollout_log_probs = [[], []]
        dead_players_set = set()

        is_complete = False
        for i in range(400):
            player = i % 4
            group = i % 2
            if player in dead_players_set:
                buffer.append(157599)
                loss_mask[group].append(0)
                loss_mask[1 - group].append(0)
                rollout_log_probs[group].append(0)
                rollout_log_probs[1 - group].append(0)
                continue

            action_mask = board.get_action_mask(player)
            assert action_mask is not None
            logits_mask = (np.where(action_mask)[0] + 151665).tolist()
            buf_slice = buffer[idx[group]:]
            idx[group] = len(buffer)
            input_ids = prompt[group] + buf_slice if req_id[group] is None else buf_slice

            response = await generate({
                'input_ids': input_ids,
                'session_params': {'id': session_ids[group], 'rid': req_id[group], },
                'sampling_params': {'max_new_tokens': 1, 'logits_mask': logits_mask, },
                'return_logprob': True,
            })

            req_id[group] = response['meta_info']['id']  # new_response_log_probs =
            action = response['output_ids'][0]
            result_id, _, flags, dead_players, reward = board.update(player, action - 151665)

            if result_id == 4:
                assert action == 157599
                buffer.append(157599)
                loss_mask[group].append(1)
                loss_mask[1 - group].append(0)
                rollout_log_probs[group].append(response["meta_info"]["output_token_logprobs"][0][0])
                rollout_log_probs[1 - group].append(0)
            else:
                result = result_ids[result_id]
                buffer.extend((action, result))
                loss_mask[group].extend([1, 0])
                loss_mask[1 - group].extend([0, 0])
                rollout_log_probs[group].extend([response["meta_info"]["output_token_logprobs"][0][0], 0])
                rollout_log_probs[1 - group].extend([0, 0])

            reward_list[group].append([reward])
            if len(reward_list[1 - group]) > 0:
                reward_list[1 - group][-1].append(-reward)
            logits_masks[group].append(logits_mask)

            if flags:
                f_ids = [flags_ids[i] for i in flags]
                buffer.extend(f_ids)

                n = len(f_ids)
                loss_mask[group].extend([0] * n)
                loss_mask[1 - group].extend([0] * n)
                rollout_log_probs[group].extend([0] * n)
                rollout_log_probs[1 - group].extend([0] * n)

            if dead_players:
                d_ids = [dead_ids[i] for i in dead_players]
                buffer.extend(d_ids)

                n = len(d_ids)
                loss_mask[group].extend([0] * n)
                loss_mask[1 - group].extend([0] * n)
                rollout_log_probs[group].extend([0] * n)
                rollout_log_probs[1 - group].extend([0] * n)

                dead_players_set.update(dead_players)
                if len(dead_players_set) == 3:
                    is_complete = True
                    break
                elif len(dead_players_set) == 2:
                    a, b = list(dead_players_set)
                    if abs(a - b) == 2:
                        is_complete = True
                        break

        await asyncio.gather(close_session(session_ids[0]), close_session(session_ids[1]))

    output = [
        Sample(
            tokens=prompt[i] + buffer,
            response_length=len(buffer),
            reward=reward_list[i],
            status=Sample.Status.COMPLETED if is_complete else Sample.Status.TRUNCATED,
            loss_mask=loss_mask[i],
            rollout_log_probs=rollout_log_probs[i],
            logits_masks=logits_masks[i],
        )
        for i in range(2)
    ]
    return output

async def generate_junqi1():
    state = GenerateState()
    async with state.semaphore:
        session_ids = await asyncio.gather(open_session(), open_session1())

        req_id = [None, None]
        buffer = []
        idx = [0, 0]

        board = Board()
        prompt = [generate_prompt(board.get_layout(0)), generate_prompt(board.get_layout(1))]

        loss_mask = [[], []]
        logits_masks = [[], []]
        reward_list = [[], []]
        rollout_log_probs = [[], []]
        dead_players_set = set()

        is_complete = False
        for i in range(400):
            player = i % 4
            group = i % 2
            if player in dead_players_set:
                buffer.append(157599)
                loss_mask[group].append(0)
                loss_mask[1 - group].append(0)
                rollout_log_probs[group].append(0)
                rollout_log_probs[1 - group].append(0)
                continue

            action_mask = board.get_action_mask(player)
            assert action_mask is not None
            logits_mask = (np.where(action_mask)[0] + 151665).tolist()
            buf_slice = buffer[idx[group]:]
            idx[group] = len(buffer)
            input_ids = prompt[group] + buf_slice if req_id[group] is None else buf_slice

            if group == 0:
                response = await generate({
                    'input_ids': input_ids,
                    'session_params': {'id': session_ids[group], 'rid': req_id[group], },
                    'sampling_params': {'max_new_tokens': 1, 'logits_mask': logits_mask, },
                    'return_logprob': True,
                })
            else:
                response = await generate1({
                    'input_ids': input_ids,
                    'session_params': {'id': session_ids[group], 'rid': req_id[group], },
                    'sampling_params': {'max_new_tokens': 1, 'logits_mask': logits_mask, },
                    'return_logprob': True,
                })

            req_id[group] = response['meta_info']['id']
            action = response['output_ids'][0]
            result_id, _, flags, dead_players, reward = board.update(player, action - 151665)

            if result_id == 4:
                assert action == 157599
                buffer.append(157599)
                loss_mask[group].append(1)
                loss_mask[1 - group].append(0)
                rollout_log_probs[group].append(response["meta_info"]["output_token_logprobs"][0][0])
                rollout_log_probs[1 - group].append(0)
            else:
                result = result_ids[result_id]
                buffer.extend((action, result))
                loss_mask[group].extend([1, 0])
                loss_mask[1 - group].extend([0, 0])
                rollout_log_probs[group].extend([response["meta_info"]["output_token_logprobs"][0][0], 0])
                rollout_log_probs[1 - group].extend([0, 0])

            reward_list[group].append([reward])
            if len(reward_list[1 - group]) > 0:
                reward_list[1 - group][-1].append(-reward)
            logits_masks[group].append(logits_mask)

            if flags:
                f_ids = [flags_ids[i] for i in flags]
                buffer.extend(f_ids)

                n = len(f_ids)
                loss_mask[group].extend([0] * n)
                loss_mask[1 - group].extend([0] * n)
                rollout_log_probs[group].extend([0] * n)
                rollout_log_probs[1 - group].extend([0] * n)

            if dead_players:
                d_ids = [dead_ids[i] for i in dead_players]
                buffer.extend(d_ids)

                n = len(d_ids)
                loss_mask[group].extend([0] * n)
                loss_mask[1 - group].extend([0] * n)
                rollout_log_probs[group].extend([0] * n)
                rollout_log_probs[1 - group].extend([0] * n)

                dead_players_set.update(dead_players)
                if len(dead_players_set) == 3:
                    is_complete = True
                    break
                elif len(dead_players_set) == 2:
                    a, b = list(dead_players_set)
                    if abs(a - b) == 2:
                        is_complete = True
                        break

        await asyncio.gather(close_session(session_ids[0]), close_session1(session_ids[1]))

    output = [
        Sample(
            tokens=prompt[i] + buffer,
            response_length=len(buffer),
            reward=reward_list[i],
            status=Sample.Status.COMPLETED if is_complete else Sample.Status.TRUNCATED,
            loss_mask=loss_mask[i],
            rollout_log_probs=rollout_log_probs[i],
            logits_masks=logits_masks[i],
        )
        for i in range(2)
    ]
    temp = 0.0
    for i in reward_list[0]:
        for j in i:
            temp +=j
    temp1 = 0.0
    for i in reward_list[1]:
        for j in i:
            temp1 +=j
    return temp > temp1

async def generate_rollout_async(args):
    state = GenerateState(args)

    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(generate_junqi())
        for _ in range(args.rollout_batch_size)
    ]
    results = await asyncio.gather(*tasks)
    return results

async def generate_rollout_async1(args):

    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(generate_junqi1())
        for _ in range(8)
    ]
    results = await asyncio.gather(*tasks)
    return results

def generate_rollout(
        args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    return run(generate_rollout_async(args))


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    result = run(generate_rollout_async1(None))
