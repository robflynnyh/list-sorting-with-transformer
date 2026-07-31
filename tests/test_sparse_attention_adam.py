from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from list_sorting_transformer.core.data import (
    make_pointer_next_batch,
    make_pointer_pair_batch,
)
from list_sorting_transformer.core.positions import sample_position_offsets
from list_sorting_transformer.length_generalisation.sparse_attention_adam import (
    AdaptiveEntmaxSelfAttention,
    PaperMatchedDecoder,
    PaperMatchedRMSNorm,
    SparseAttentionAdamConfig,
    SparseAttentionPointerTransformer,
    comparison_trace_targets,
    comparison_targets,
    entmax15,
    evaluation_batch_size,
    generate_comparison_trace,
    learning_rate_at_step,
    make_evaluation_data,
    make_position_offsets,
    make_task_batch,
    pointer_pair_trace_targets,
    pointer_targets,
    task_targets,
    teacher_forced_trace_logits,
)
from list_sorting_transformer.core.tokens import (
    VALUE_OFFSET,
    PointerCompareVocabulary,
)


def small_config(**overrides: object) -> SparseAttentionAdamConfig:
    values = {
        "run_name": "test",
        "steps": 10,
        "batch_size": 4,
        "train_max_length": 4,
        "eval_lengths": (2, 4),
        "final_eval_lengths": (8,),
        "eval_examples": 4,
        "final_eval_examples": 2,
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "alibi_heads": 2,
        "position_moduli": (3, 5, 7, 11),
        "position_offset_min": -100,
        "position_offset_max": 100,
        "warmup_steps": 1,
        "log_interval": 1,
        "eval_interval": 2,
        "checkpoint_interval": 2,
        "device": "cpu",
        "precision": "float32",
    }
    values.update(overrides)
    return SparseAttentionAdamConfig(**values)


def test_entmax15_is_normalized_sparse_and_differentiable() -> None:
    scores = torch.tensor(
        [[3.0, 1.0, -2.0, -10.0], [0.1, 0.2, 0.3, 0.4]],
        requires_grad=True,
    )
    probabilities = entmax15(scores)

    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2),
    )
    assert bool(probabilities[0, -1].eq(0))
    assert bool((probabilities >= 0).all())
    probabilities.square().sum().backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())


def test_entmax15_backward_is_finite_at_quantized_support_boundary() -> None:
    scores = torch.tensor(
        [
            [
                0.34375,
                0.1953125,
                -0.85546875,
                -1.3046875,
                -0.333984375,
                0.62890625,
                -0.6953125,
                -1.359375,
            ]
        ],
        requires_grad=True,
    )
    gradient = torch.arange(scores.shape[-1], dtype=scores.dtype)

    probabilities = entmax15(scores)
    (probabilities * gradient).sum().backward()

    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())


def test_nape_split_and_adaptive_scalers_receive_gradients() -> None:
    config = small_config()
    model = SparseAttentionPointerTransformer(config)
    attention = model.encoder.blocks[0].attention
    assert isinstance(attention, AdaptiveEntmaxSelfAttention)
    torch.testing.assert_close(
        attention.slopes,
        torch.tensor([1.0, 0.5, 0.0, 0.0]),
    )

    generator = torch.Generator().manual_seed(17)
    batch = make_pointer_next_batch(
        config.batch_size,
        4,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = sample_position_offsets(
        config.batch_size,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=generator,
        device=torch.device("cpu"),
    )
    logits = model(batch.prompt_ids, offsets=offsets)
    loss = F.cross_entropy(logits, pointer_targets(batch))
    loss.backward()

    assert attention.beta_projection.weight.grad is not None
    assert attention.gamma_projection.weight.grad is not None
    assert bool(torch.isfinite(attention.beta_projection.weight.grad).all())
    metrics = model.attention_metrics()
    assert 0 < metrics["attention/support_fraction"] < 1
    assert metrics["attention/support_size"] >= 1


def test_evaluation_batch_size_respects_attention_budget() -> None:
    config = small_config(
        eval_batch_size=32,
        eval_attention_element_budget=4 * 100 * 100,
    )
    assert evaluation_batch_size(config, prompt_length=10) == 32
    assert evaluation_batch_size(config, prompt_length=100) == 1


def test_long_recurring_evaluations_use_smaller_fixed_sets() -> None:
    config = small_config(
        eval_lengths=(2, 4, 8),
        final_eval_lengths=(12,),
        eval_examples=7,
        long_eval_examples=3,
        long_eval_min_length=8,
    )

    data = make_evaluation_data(
        config,
        lengths=config.eval_lengths,
        examples=config.eval_examples,
        long_examples=config.long_eval_examples,
        long_min_length=config.long_eval_min_length,
    )

    assert data[2][0].values.shape[0] == 7
    assert data[4][0].values.shape[0] == 7
    assert data[8][0].values.shape[0] == 3


def test_paper_matched_model_uses_nape_only_gemma_style_inputs() -> None:
    config = small_config(
        architecture="paper_gemma2",
        input_position_mode="nape_only",
        value_input_mode="embedding",
        minimum_lr_ratio=0.0,
        optimizer_name="adamw",
    )
    model = SparseAttentionPointerTransformer(config)

    assert isinstance(model.encoder, PaperMatchedDecoder)
    assert model.position_embedding is None
    assert sum(
        isinstance(module, PaperMatchedRMSNorm) for module in model.modules()
    ) == 4 * config.layers + 1
    generator = torch.Generator().manual_seed(19)
    batch = make_pointer_next_batch(
        config.batch_size,
        config.train_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = make_position_offsets(
        config,
        config.batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )
    logits = model(batch.prompt_ids, offsets=offsets)
    loss = F.cross_entropy(logits, pointer_targets(batch))
    loss.backward()

    assert torch.equal(offsets, torch.zeros_like(offsets))
    assert logits.shape == (config.batch_size, config.symbol_count)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_paper_cosine_schedule_reaches_zero() -> None:
    config = small_config(
        steps=20,
        warmup_steps=4,
        minimum_lr_ratio=0.0,
    )

    assert learning_rate_at_step(config, 4) == config.learning_rate
    assert learning_rate_at_step(config, 20) == 0.0


def test_short_ablation_can_remain_inside_long_warmup() -> None:
    config = small_config(
        steps=5_000,
        warmup_steps=20_000,
    )

    assert learning_rate_at_step(config, 5_000) == (
        config.learning_rate * 0.25
    )


def test_softmax_attention_is_dense_on_causally_allowed_positions() -> None:
    config = small_config(attention_normalizer="softmax")
    model = SparseAttentionPointerTransformer(config)
    generator = torch.Generator().manual_seed(23)
    batch = make_pointer_next_batch(
        config.batch_size,
        config.train_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = make_position_offsets(
        config,
        config.batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )

    loss = F.cross_entropy(
        model(batch.prompt_ids, offsets=offsets),
        pointer_targets(batch),
    )
    loss.backward()

    assert model.attention_metrics()["attention/support_fraction"] == 1.0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_chunked_eval_attention_matches_unchunked_attention() -> None:
    config = small_config(
        attention_normalizer="entmax15",
        eval_attention_query_chunk_size=2,
    )
    model = SparseAttentionPointerTransformer(config)
    generator = torch.Generator().manual_seed(27)
    batch = make_pointer_next_batch(
        config.batch_size,
        config.train_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = make_position_offsets(
        config,
        config.batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )

    model.eval()
    chunked = model(batch.prompt_ids, offsets=offsets)
    chunked_metrics = model.attention_metrics()
    for block in model.encoder.blocks:
        block.attention.eval_query_chunk_size = 0
    unchunked = model(batch.prompt_ids, offsets=offsets)
    unchunked_metrics = model.attention_metrics()

    torch.testing.assert_close(chunked, unchunked)
    assert chunked_metrics.keys() == unchunked_metrics.keys()
    for name, value in chunked_metrics.items():
        assert value == pytest.approx(unchunked_metrics[name])


def test_softmax_without_scaling_has_unit_scale() -> None:
    config = small_config(
        attention_normalizer="softmax",
        scaling_mode="none",
    )
    model = SparseAttentionPointerTransformer(config)
    generator = torch.Generator().manual_seed(29)
    batch = make_pointer_next_batch(
        config.batch_size,
        config.train_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = make_position_offsets(
        config,
        config.batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )

    model(batch.prompt_ids, offsets=offsets)
    metrics = model.attention_metrics()

    assert metrics["attention/scale_mean"] == 1.0
    assert all(
        block.attention.beta_projection is None
        and block.attention.gamma_projection is None
        for block in model.encoder.blocks
    )


def test_all_nope_heads_have_zero_alibi_slopes() -> None:
    config = small_config(
        alibi_heads=0,
        attention_normalizer="softmax",
        input_position_mode="nape_only",
        value_input_mode="embedding",
    )
    model = SparseAttentionPointerTransformer(config)
    generator = torch.Generator().manual_seed(31)
    batch = make_pointer_next_batch(
        config.batch_size,
        config.train_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = make_position_offsets(
        config,
        config.batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )

    model(batch.prompt_ids, offsets=offsets)

    assert all(
        torch.equal(
            block.attention.slopes,
            torch.zeros_like(block.attention.slopes),
        )
        for block in model.encoder.blocks
    )
    assert model.attention_metrics()["attention/alibi_support_size"] == 0.0


def test_all_alibi_heads_work_without_adaptive_scaling() -> None:
    config = small_config(
        alibi_heads=small_config().heads,
        attention_normalizer="softmax",
        scaling_mode="none",
        input_position_mode="nape_only",
        value_input_mode="embedding",
    )
    model = SparseAttentionPointerTransformer(config)
    generator = torch.Generator().manual_seed(37)
    batch = make_pointer_next_batch(
        config.batch_size,
        config.train_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = make_position_offsets(
        config,
        config.batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )

    model(batch.prompt_ids, offsets=offsets)

    assert all(
        bool(block.attention.slopes.gt(0).all())
        for block in model.encoder.blocks
    )
    assert model.attention_metrics()["attention/nope_support_size"] == 0.0


def test_pointer_compare_task_uses_action_vocabulary_and_targets() -> None:
    config = small_config(
        task="pointer_compare",
        attention_normalizer="softmax",
        scaling_mode="none",
        input_position_mode="nape_only",
        value_input_mode="embedding",
    )
    model = SparseAttentionPointerTransformer(config)
    batch = make_task_batch(
        config,
        3,
        3,
        generator=torch.Generator().manual_seed(41),
        vocabulary=model.vocabulary,
    )
    batch.values[:] = torch.tensor(
        [
            [1, 2, 9],
            [7, 7, 0],
            [8, 3, 4],
        ]
    )
    batch.pointers[:] = 0
    offsets = torch.zeros(3, dtype=torch.long)

    assert isinstance(model.vocabulary, PointerCompareVocabulary)
    assert torch.equal(comparison_targets(batch), torch.tensor([0, 0, 1]))
    assert torch.equal(task_targets(config, batch), torch.tensor([0, 0, 1]))
    assert model(batch.prompt_ids, offsets=offsets).shape == (3, 2)


def test_pointer_compare_evaluation_batches_hide_pair_outputs() -> None:
    config = small_config(task="pointer_compare")
    data = make_evaluation_data(
        config,
        lengths=(4,),
        examples=5,
    )
    batch, _ = data[4]
    direct = make_pointer_pair_batch(
        5,
        4,
        generator=torch.Generator().manual_seed(config.seed + 30_400),
        vocabulary=PointerCompareVocabulary("numbers", config.symbol_count),
    )

    assert torch.equal(batch.prompt_ids, direct.prompt_ids)
    assert batch.prompt_length == direct.prompt_length


def test_pointer_compare_trace_predicts_values_then_action() -> None:
    config = small_config(
        task="pointer_compare_trace",
        attention_normalizer="softmax",
        scaling_mode="none",
        input_position_mode="nape_only",
        value_input_mode="embedding",
    )
    model = SparseAttentionPointerTransformer(config)
    batch = make_task_batch(
        config,
        3,
        3,
        generator=torch.Generator().manual_seed(43),
        vocabulary=model.vocabulary,
    )
    batch.values[:] = torch.tensor(
        [
            [1, 2, 9],
            [7, 7, 0],
            [8, 3, 4],
        ]
    )
    batch.pointers[:] = 0
    offsets = torch.zeros(3, dtype=torch.long)

    assert isinstance(model.vocabulary, PointerCompareVocabulary)
    expected = torch.tensor(
        [
            [
                VALUE_OFFSET + 1,
                VALUE_OFFSET + 2,
                model.vocabulary.action_token("KEEP"),
            ],
            [
                VALUE_OFFSET + 7,
                VALUE_OFFSET + 7,
                model.vocabulary.action_token("KEEP"),
            ],
            [
                VALUE_OFFSET + 8,
                VALUE_OFFSET + 3,
                model.vocabulary.action_token("SWAP"),
            ],
        ]
    )
    targets = comparison_trace_targets(batch, model.vocabulary)
    logits, teacher_targets = teacher_forced_trace_logits(
        model,
        batch,
        offsets,
    )

    assert torch.equal(targets, expected)
    assert torch.equal(task_targets(config, batch, model.vocabulary), expected)
    assert torch.equal(teacher_targets, expected)
    assert logits.shape == (3, 3, model.vocabulary.size)
    F.cross_entropy(logits.flatten(0, 1), teacher_targets.flatten()).backward()
    assert model.output.weight.grad is not None


def test_pointer_pair_trace_uses_the_same_autoregressive_value_targets() -> None:
    config = small_config(
        task="pointer_pair_trace",
        attention_normalizer="softmax",
        scaling_mode="none",
        input_position_mode="nape_only",
        value_input_mode="embedding",
    )
    model = SparseAttentionPointerTransformer(config)
    batch = make_task_batch(
        config,
        3,
        3,
        generator=torch.Generator().manual_seed(47),
        vocabulary=model.vocabulary,
    )
    offsets = torch.zeros(3, dtype=torch.long)
    expected = pointer_pair_trace_targets(batch)

    logits, targets = teacher_forced_trace_logits(model, batch, offsets)

    assert torch.equal(targets, expected)
    assert torch.equal(task_targets(config, batch, model.vocabulary), expected)
    assert logits.shape == (3, 2, model.vocabulary.size)


def test_pointer_compare_trace_decoding_is_autoregressive() -> None:
    class ScriptedModel:
        def __init__(self) -> None:
            self.input_lengths: list[int] = []

        def __call__(
            self,
            token_ids: torch.Tensor,
            *,
            offsets: torch.Tensor,
        ) -> torch.Tensor:
            del offsets
            self.input_lengths.append(token_ids.shape[1])
            logits = torch.zeros(token_ids.shape[0], 20)
            logits[:, 5 + len(self.input_lengths)] = 1
            return logits

    model = ScriptedModel()
    prompt = torch.tensor([[1, 5, 2], [1, 6, 2]])
    generated = generate_comparison_trace(
        model,  # type: ignore[arg-type]
        prompt,
        offsets=torch.zeros(2, dtype=torch.long),
    )

    assert model.input_lengths == [3, 4, 5]
    assert torch.equal(
        generated,
        torch.tensor([[6, 7, 8], [6, 7, 8]]),
    )
