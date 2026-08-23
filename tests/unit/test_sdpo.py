# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDPO (https://arxiv.org/abs/2601.20802): the advantage estimator and the
self-teacher reprompt builder in the experience maker."""

from types import SimpleNamespace

import torch

from molt.trainer.algorithm.advantage import AdvantageContext, get_advantage_estimator
from molt.trainer.algorithm.experience import Experience
from molt.trainer.rollout.experience_maker import SDPO_TEACHER_PREFIX, RemoteExperienceMaker


def test_sdpo_estimator_is_scaled_negated_kl_without_whitening():
    # The SDPO advantage is kl_coef * (log q_teacher - log pi_old) = -kl_coef * kl on action
    # tokens, with NO whitening: sign and scale of the distillation signal must survive.
    kl = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    ctx = AdvantageContext(
        sample_to_rollout=torch.tensor([0]),
        exp_len=[1],
        action_masks=[mask],
        kl_coef=2.0,
        gamma=1.0,
        lam=1.0,
        kls=[kl],
    )
    advantages, returns = get_advantage_estimator("sdpo")(torch.tensor([1.0]), [[0]], ctx)

    expected = torch.tensor([[-1.0, 2.0, -4.0, 0.0]])  # -2 * kl, masked
    torch.testing.assert_close(advantages[0], expected)
    torch.testing.assert_close(returns[0], expected)


class _FakeTokenizer:
    """Deterministic stand-in: decode joins ids as text, __call__ hashes text into two ids."""

    bos_token_id = 7

    def __init__(self):
        self.encoded_texts = []

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        assert not add_special_tokens and return_tensors == "pt"
        self.encoded_texts.append(text)
        ids = [1000 + sum(map(ord, text)) % 100, 1000 + len(text) % 100]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def _sdpo_maker(tokenizer, threshold=1.0):
    """Duck-typed RemoteExperienceMaker self with just what the teacher builder reads."""
    maker = SimpleNamespace(
        tokenizer=tokenizer,
        args=SimpleNamespace(algo=SimpleNamespace(sdpo=SimpleNamespace(success_threshold=threshold))),
    )
    maker._build_sdpo_teacher_inputs = RemoteExperienceMaker._build_sdpo_teacher_inputs.__get__(maker)
    return maker


def _sample(tokens, reward, group_id, n_actions):
    seq = torch.tensor([tokens], dtype=torch.long)
    action_mask = torch.zeros(1, seq.shape[1] - 1, dtype=torch.bool)
    action_mask[0, -n_actions:] = True
    return Experience(
        sequences=seq,
        attention_mask=torch.ones_like(seq),
        action_mask=action_mask,
        rewards=torch.tensor([float(reward)]),
        group_ids=[group_id],
        info={},
    )


def test_sdpo_teacher_builder_reprompts_from_group_solution():
    tokenizer = _FakeTokenizer()
    # g0: sample 0 succeeded, sample 1 failed -> both get a teacher (0 from itself, 1 from 0's
    # solution). g1: never succeeded -> stays teacherless (zero SDPO advantage downstream).
    exps = [
        _sample([1, 2, 3, 41, 42, 43], reward=1.0, group_id="g0", n_actions=3),
        _sample([1, 2, 3, 51, 52], reward=0.0, group_id="g0", n_actions=2),
        _sample([1, 2, 3, 61, 62], reward=0.0, group_id="g1", n_actions=2),
    ]
    _sdpo_maker(tokenizer)._build_sdpo_teacher_inputs(exps)

    # Both g0 reprompts embed the SUCCESSFUL sample's decoded action tokens ("41 42 43").
    solution_text = SDPO_TEACHER_PREFIX.format(solution="41 42 43")
    assert tokenizer.encoded_texts == [solution_text, solution_text]

    for exp in exps[:2]:
        prefix_len = exp.teacher_sequences.shape[1] - exp.sequences.shape[1]
        assert prefix_len == 2  # the fake tokenizer's two prefix ids, prepended (no BOS in seq)
        assert torch.equal(exp.teacher_sequences[:, prefix_len:], exp.sequences)
        assert torch.equal(exp.teacher_attention_mask, torch.ones_like(exp.teacher_sequences))
    assert exps[2].teacher_sequences is None and exps[2].teacher_attention_mask is None


def test_sdpo_teacher_builder_inserts_prefix_after_bos():
    tokenizer = _FakeTokenizer()
    exp = _sample([7, 2, 3, 41, 42], reward=1.0, group_id="g0", n_actions=2)  # 7 = BOS
    _sdpo_maker(tokenizer)._build_sdpo_teacher_inputs([exp])

    teacher = exp.teacher_sequences[0]
    assert teacher[0].item() == 7  # BOS stays first
    assert torch.equal(teacher[3:], exp.sequences[0, 1:])  # original tail intact after the prefix


def test_sdpo_teacher_builder_respects_success_threshold():
    tokenizer = _FakeTokenizer()
    exps = [
        _sample([1, 2, 3, 41], reward=0.9, group_id="g0", n_actions=1),
        _sample([1, 2, 3, 51], reward=0.0, group_id="g0", n_actions=1),
    ]
    _sdpo_maker(tokenizer, threshold=0.5)._build_sdpo_teacher_inputs(exps)
    assert exps[0].teacher_sequences is not None and exps[1].teacher_sequences is not None

    exps = [
        _sample([1, 2, 3, 41], reward=0.9, group_id="g0", n_actions=1),
        _sample([1, 2, 3, 51], reward=0.0, group_id="g0", n_actions=1),
    ]
    _sdpo_maker(tokenizer, threshold=1.0)._build_sdpo_teacher_inputs(exps)
    assert exps[0].teacher_sequences is None and exps[1].teacher_sequences is None
