from collections.abc import Callable
from typing import Any

import numpy as np
import pybase64
import sglang_router
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
from board.board import Board
from collections import defaultdict

client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=512),
    timeout=httpx.Timeout(None),
    trust_env=False,
)
url = "http://172.17.0.43:15000/open_session"
url1 = "http://172.17.0.43:15000/generate"


async def make_group():
    response = await client.post(url, json={"capacity_of_str_len": 1024}, headers=None)
    return response.json()


async def generate(payload):
    response = await client.post(url1, json=payload, headers=None)
    return response.json()


flags_ids = [8, 8, 8, 8, 8, 8, 8, 8]
dead_ids = [4, 4, 4, 4]


class BoardNode:
    def __init__(self, session_ids, board: Board, layout):
        self.session_ids = session_ids
        self.board = board
        self.layout = layout

        self.buffer = []
        self.idx = [0, 0]
        self.loss_mask=[[], []]
        self.req_id = [None, None]

        self.player = 0
        self.finish = False
        self.dead_players = []

        self.futures = [[], []]
        self.rewards = [[], []]

    async def step(self):
        if self.finish:
            return
        if self.player in self.dead_players:
            self.player = (self.player + 1) % 4
            return

        action_mask = self.board.get_action_mask(self.player)
        assert action_mask is not None
        group = self.player % 2
        input_ids = self.layout[group] + self.buffer[self.idx[group]:] \
            if self.req_id[group] is None else self.buffer[self.idx[group]:]

        response = await generate({
            'input_ids': input_ids,
            'session_params': {'id': self.session_ids[group], 'rid': self.req_id[group], },
            'sampling_params': {'max_new_tokens': 1, 'logits_mask': np.where(action_mask)[0].tolist(), },
            'return_logprob': True,
        })

        self.req_id[group] = response['meta_info']['rid']
        action = response['output_ids'][0]
        result, _, flags, dead_players = self.board.update(self.player, 0)

        self.idx[group] = len(self.buffer)
        self.buffer.append(action)
        self.buffer.append(result)
        self.loss_mask[group].extend([1,0])
        if flags:
            self.buffer.extend(flags_ids[flags])
            self.loss_mask[group].append(0)
        if dead_players:
            self.buffer.extend(dead_ids[dead_players])
            self.loss_mask[group].extend([0 for _ in range(len(dead_players))])
            self.dead_players.extend(dead_players)
            if len(self.dead_players) == 3 or (
                    len(self.dead_players) == 2 and abs(self.dead_players[0] - self.dead_players[1]) == 2):
                self.finish = True

        self.player = (self.player + 1) % 4

    async def split_step(self):
        action_mask = self.board.get_action_mask(self.player)
        if action_mask is None:
            self.player = (self.player + 1) % 4
            return []

        group = self.player % 2
        input_ids = self.layout[group] + self.buffer[self.idx[group]:] \
            if self.req_id[group] is None else self.buffer[self.idx[group]:]

        response = await generate({
            'input_ids': input_ids,
            'session_params': {'id': self.session_ids[group], 'rid': self.req_id[group], },
            'sampling_params': {'max_new_tokens': 1, 'logits_mask': action_mask.tolist(), },
            'return_logprob': True,
            'top_logprobs_num': 4,
        })

        self.req_id[group] = response['meta_info']['rid']
        new_nodes = []
        for i, (_, action, _) in enumerate(response['meta_info']['output_top_logprobs'][0]):
            if i == 0:
                new_node = self
            else:
                new_node = copy.deepcopy(self)

            result, _, flags, dead_players = new_node.board.update(new_node.player, action)

            new_node.idx[group] = len(new_node.buffer)
            new_node.buffer.append(action)
            new_node.buffer.append(result)

            if flags:
                new_node.buffer.extend(flags_ids[flags])

            if dead_players:
                new_node.buffer.extend(dead_ids[dead_players])
                new_node.dead_players.extend(dead_players)
                if len(new_node.dead_players) == 3 or (
                        len(new_node.dead_players) == 2 and abs(
                    new_node.dead_players[0] - new_node.dead_players[1]) != 2):
                    new_node.finish = True

            new_node.player = (new_node.player + 1) % 4

            if i != 0:
                new_nodes.append(new_node)

        return new_nodes


async def compute_reward(futures, futures1):
    results = await asyncio.gather(*futures)

    rewards = [r * 2 for r in results]  # 举例

    for fut, reward in zip(futures1, rewards):
        fut.set_result(reward)

async def generate_junqi():
    session_ids = await asyncio.gather(
        make_group(),
        make_group()
    )
    board = Board()
    layout = [board.get_layout(0), board.get_layout(1)]
    running_nodes = [BoardNode(session_ids, board, layout)]
    running_node = running_nodes[0]

    loop = asyncio.get_event_loop()
    split_num = 4
    futures= [loop.create_future() for _ in range(split_num)]
    futures1 = [loop.create_future() for _ in range(split_num)]
    new_nodes = [running_node] + [running_node.copy() for _ in range(split_num - 1)]
    for node, future in zip(new_nodes, futures1):
        node.futures.append(future)
    for node, future in zip(new_nodes, futures):
        loop.create_task(aa(node, future))
    loop.create_task(compute_reward(futures, futures1))



    finish_nodes = []
    step_num = 0
    limit = 128

    while running_nodes and step_num < 16:
        finished = [n for n in running_nodes if n.finish]
        finish_nodes.extend(finished)

        running_nodes = [n for n in running_nodes if not n.finish]
        if not running_nodes:
            break

        if step_num % 4 == 0:
            results = await asyncio.gather(
                *(node.split_step() for node in running_nodes[:limit])
            )

            running_nodes.extend(n for sub in results for n in sub)
        else:
            await asyncio.gather(
                *(node.step() for node in running_nodes[:limit])
            )
        step_num += 1

    return finish_nodes


result = run(generate_junqi())
