"""deeprelay as a Hermes model provider: pick deeprelay in ``hermes model`` or pass ``--provider deeprelay``.

deeprelay is a serverless inference relay for open-weight models (DeepSeek, Moonshot Kimi, Z.AI GLM,
Alibaba Qwen, MiniMax, Meta Muse, OpenAI gpt-oss) behind one OpenAI-compatible API at
https://api.deeprelay.ai/v1. What this plugin adds over pointing a custom endpoint there:

* The picker offers the chat models below. deeprelay's model list also carries image, video and
  embedding models; they stay out. A new chat model is added here in a reviewed release, not because
  it appears in the list.
* Context window, tool support, reasoning and image input are declared for each model, so Hermes does
  not have to guess them. Model ids keep the ``deeprelay/`` prefix the API requires.
* Reasoning effort goes out as the top-level ``reasoning_effort`` field, clamped to the levels each
  model accepts (deeprelay answers 400 to a level a model does not have), and ``hermes model`` offers
  only those levels. ``none`` turns thinking off. With no effort configured nothing is sent and the
  model's own default applies.
* The cost of each answer is the amount deeprelay charged for it (``usage.cost`` in the response), so
  Hermes shows the real charge instead of an estimate from list prices. Requests covered by a
  deeprelay plan are charged $0.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal, InvalidOperation
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

API_KEY_ENV = "DEEPRELAY_API_KEY"
BASE_URL_ENV = "DEEPRELAY_BASE_URL"
BASE_URL = "https://api.deeprelay.ai/v1"

# Hermes' reasoning-effort levels, weakest first.
EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

_ALL = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_TO_XHIGH = ("none", "minimal", "low", "medium", "high", "xhigh")

# Models offered in the picker: id -> (context window in tokens, accepts image input, effort levels).
# From deeprelay's authenticated GET /v1/models (context_length, the `vision` parameter and
# reasoning_efforts) on 2026-11-25. Order sets the picker order and the fallback default: deeprelay's
# coding-agent ranking first, then alphabetical.
MODELS: dict[str, tuple[int, bool, tuple[str, ...]]] = {
    "deeprelay/deepseek-v4.1-flash": (1_048_576, True, _ALL),
    "deeprelay/deepseek-v4-pro": (1_048_576, False, _ALL),
    "deeprelay/glm-5.3": (1_048_576, False, ("minimal", "low", "medium", "high", "xhigh")),
    "deeprelay/kimi-k3.1-pro": (1_048_576, True, ("none", "low", "medium", "high")),
    "deeprelay/glm-5.2": (1_048_576, False, _TO_XHIGH),
    "deeprelay/glm-5.3-flash": (1_048_576, True, _TO_XHIGH),
    "deeprelay/gpt-oss-120b": (131_072, False, _TO_XHIGH),
    "deeprelay/kimi-k3": (1_048_576, True, _TO_XHIGH),
    "deeprelay/minimax-m3.1-flash": (1_048_576, True, _TO_XHIGH),
    "deeprelay/muse-glimmer-30b": (131_072, False, _TO_XHIGH),
    "deeprelay/muse-spark-open-816b-a42b": (1_048_576, False, ("none", "low", "medium", "high", "xhigh")),
    "deeprelay/qwen3.5-9b": (262_144, True, ("low", "medium", "high")),
    "deeprelay/qwen3.6-plus": (1_000_000, False, _TO_XHIGH),
    "deeprelay/qwen3.7-max": (1_000_000, False, _TO_XHIGH),
    "deeprelay/qwen3.7-plus": (1_000_000, False, _TO_XHIGH),
    "deeprelay/qwen3.8-flash": (1_000_000, False, _TO_XHIGH),
    "deeprelay/qwen3.8-max": (1_000_000, True, _TO_XHIGH),
    "deeprelay/qwen4-max": (1_000_000, False, _ALL),
}


def is_chat_model(model_id: Any) -> bool:
    """Whether *model_id* is one of the listed chat models (exact, case-sensitive)."""
    return isinstance(model_id, str) and model_id in MODELS


def reasoning_effort_for(model: Any, reasoning_config: Any) -> str | None:
    """The ``reasoning_effort`` to send for *model*, or None to leave the field out.

    Unset sends nothing. Turning reasoning off sends ``none``, or the weakest level of a model that
    cannot turn thinking off. With reasoning on, a level the model does not take becomes the nearest
    weaker one it does (the weakest when none is weaker), and never ``none``: the same rule as
    Hermes's own clamp. A name that is not a Hermes level is never sent.
    """
    if not isinstance(reasoning_config, dict):
        return None
    requested = str(reasoning_config.get("effort") or "").strip().lower()
    if reasoning_config.get("enabled") is False:
        requested = "none"
    if requested not in EFFORT_LADDER:
        return None
    levels = MODELS[model][2] if is_chat_model(model) else _ALL
    if requested == "none":
        return "none" if "none" in levels else levels[0]
    enabled = [level for level in levels if level != "none"]
    rank = EFFORT_LADDER.index(requested)
    weaker = [level for level in enabled if EFFORT_LADDER.index(level) <= rank]
    return weaker[-1] if weaker else enabled[0]


def reasoning_levels(model: Any) -> tuple[str, ...] | None:
    """Levels *model* accepts (shown by ``hermes model``), or None for a model not listed here."""
    return MODELS[model][2] if is_chat_model(model) else None


def charged_cost(usage: Any) -> Decimal | None:
    """deeprelay's own charge for the answer (``usage.cost``, USD), or None when the response has none."""
    raw = getattr(usage, "raw_usage", None)
    if not isinstance(raw, dict):
        return None
    value = raw.get("cost")
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def cost_label(amount: Decimal) -> str:
    if amount == 0:
        return "$0.00"
    if amount < Decimal("0.01"):
        text = f"${amount:.4f}"
        return text if text != "$0.0000" else "$<0.0001"
    return f"${amount:.2f}"


class DeeprelayProfile(ProviderProfile):
    """deeprelay's OpenAI-compatible endpoint: listed chat models, per-model reasoning levels, charged cost."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = reasoning_effort_for(context.get("model"), reasoning_config)
        return {}, ({"reasoning_effort": effort} if effort else {})

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        return reasoning_levels(model)

    def get_model_context_length(self, model: str) -> int | None:
        return MODELS[model][0] if is_chat_model(model) else None

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        amount = charged_cost(usage)
        if amount is None:
            return None
        from agent.usage_pricing import CostResult

        return CostResult(
            amount_usd=amount, status="actual", source="provider_cost_api", label=cost_label(amount),
            notes=("Amount deeprelay charged for this response (usage.cost).",),
        )

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if models is None:
            return None
        # deeprelay's list also holds image, video and embedding models: keep only the chat models
        # listed here, in picker order, each once. A model deeprelay retires drops out of the live
        # list; Hermes merges the result with fallback_models when the fetch fails.
        available = {model for model in models if is_chat_model(model)}
        return [model for model in MODELS if model in available]


deeprelay = DeeprelayProfile(
    name="deeprelay",
    aliases=("deep-relay",),
    display_name="deeprelay",
    description="deeprelay - one OpenAI-compatible key for DeepSeek, Kimi, GLM, Qwen, MiniMax and more",
    signup_url="https://deeprelay.ai/sign-up",
    env_vars=(API_KEY_ENV, BASE_URL_ENV),
    base_url=BASE_URL,
    auth_type="api_key",
    default_aux_model="deeprelay/deepseek-v4.1-flash",
    fallback_models=tuple(MODELS),
    model_capabilities={
        model_id: {
            "context_window": context_window,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_vision": vision,
        }
        for model_id, (context_window, vision, _levels) in MODELS.items()
    },
)

# The per-model effort step of `hermes model` (newer Hermes; the field is absent from 0.22.x).
if "setup_reasoning_efforts" in {f.name for f in dataclasses.fields(ProviderProfile)}:
    deeprelay.setup_reasoning_efforts = reasoning_levels

register_provider(deeprelay)
