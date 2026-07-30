"""Parameter-matched recurrent baseline for the sorting task."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tokens import EOS, PAD, VALUE_OFFSET


@dataclass(frozen=True)
class LSTMConfig:
    vocab_size: int
    symbol_count: int = 10
    representation: str = "numbers"
    d_model: int = 128
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.representation not in {"alphabet", "numbers"}:
            raise ValueError("representation must be 'alphabet' or 'numbers'")
        if self.symbol_count < 2:
            raise ValueError("symbol_count must be at least two")
        if self.vocab_size < VALUE_OFFSET + self.symbol_count:
            raise ValueError("vocab_size is too small for symbol_count")
        if self.d_model < 1 or self.hidden_size < 1 or self.n_layers < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class LSTMSorter(nn.Module):
    architecture = "lstm"

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.number_projection = (
            nn.Linear(1, config.d_model, bias=False)
            if config.representation == "numbers"
            else None
        )
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.d_model,
            bias=False,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self._initialize()

    def _initialize(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.number_projection is not None:
            nn.init.normal_(self.number_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                if "bias_ih" in name:
                    parameter.data[
                        self.config.hidden_size : 2 * self.config.hidden_size
                    ].fill_(1.0)

    def embed(self, token_ids: Tensor) -> Tensor:
        hidden = self.token_embedding(token_ids)
        if self.number_projection is None:
            return hidden
        is_value = (token_ids >= VALUE_OFFSET) & (
            token_ids < VALUE_OFFSET + self.config.symbol_count
        )
        values = token_ids.to(dtype=hidden.dtype) - VALUE_OFFSET
        values = 2.0 * values / (self.config.symbol_count - 1) - 1.0
        values = torch.where(is_value, values, torch.zeros_like(values))
        return hidden + self.number_projection(values.unsqueeze(-1))

    def decode_hidden(self, recurrent_hidden: Tensor) -> Tensor:
        hidden = self.final_norm(self.output_projection(recurrent_hidden))
        return F.linear(hidden, self.token_embedding.weight)

    def forward_with_state(
        self,
        token_ids: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        recurrent_hidden, new_state = self.lstm(self.embed(token_ids), state)
        return self.decode_hidden(recurrent_hidden), new_state

    def forward(self, token_ids: Tensor) -> Tensor:
        logits, _ = self.forward_with_state(token_ids)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        stop_token: int = EOS,
    ) -> Tensor:
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, time]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        prompt_logits, state = self.forward_with_state(prompt_ids)
        next_logits = prompt_logits[:, -1:]
        generated = []
        finished = torch.zeros(
            prompt_ids.shape[0],
            dtype=torch.bool,
            device=prompt_ids.device,
        )
        for _ in range(max_new_tokens):
            next_token = next_logits[:, -1].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PAD),
                next_token,
            )
            generated.append(next_token)
            finished = finished | next_token.eq(stop_token)
            if bool(finished.all()):
                break
            next_logits, state = self.forward_with_state(
                next_token[:, None],
                state,
            )
        return torch.stack(generated, dim=1)
