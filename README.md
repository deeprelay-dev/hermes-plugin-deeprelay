# deeprelay for Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin for
[deeprelay](https://deeprelay.ai): one OpenAI-compatible key for open-weight chat models from
DeepSeek, Moonshot (Kimi), Z.AI (GLM), Alibaba (Qwen), MiniMax, Meta (Muse) and OpenAI (gpt-oss).

## Install

```bash
hermes plugins install deeprelay-provider
export DEEPRELAY_API_KEY=...   # create a key at https://deeprelay.ai after signing up
hermes model                   # pick "deeprelay", then a model
```

Or per run: `hermes chat --provider deeprelay --model deeprelay/kimi-k3.1-pro`.

## What it does

- **Models.** The picker lists deeprelay's chat models (image, video and embedding models are left
  out). Ids keep the `deeprelay/` prefix the API requires, e.g. `deeprelay/deepseek-v4.1-flash`.
- **Capabilities.** Context window, tool support, reasoning and image input are declared per model.
- **Reasoning.** Hermes' effort setting is sent as the top-level `reasoning_effort`, clamped to the
  levels each model accepts (nearest weaker level). Turning reasoning off sends `none`, or the
  lowest level on models that always think (GLM 5.3, Qwen3.5 9B).
- **Cost.** Hermes records the amount deeprelay charged for each answer (`usage.cost`). Requests
  covered by a deeprelay plan show $0.

| Model | Context | Image input |
| --- | --- | --- |
| `deeprelay/deepseek-v4.1-flash` | 1,048,576 | yes |
| `deeprelay/deepseek-v4-pro` | 1,048,576 | no |
| `deeprelay/glm-5.3` | 1,048,576 | no |
| `deeprelay/kimi-k3.1-pro` | 1,048,576 | yes |
| `deeprelay/glm-5.2` | 1,048,576 | no |
| `deeprelay/glm-5.3-flash` | 1,048,576 | yes |
| `deeprelay/gpt-oss-120b` | 131,072 | no |
| `deeprelay/kimi-k3` | 1,048,576 | yes |
| `deeprelay/minimax-m3.1-flash` | 1,048,576 | yes |
| `deeprelay/muse-glimmer-30b` | 131,072 | no |
| `deeprelay/muse-spark-open-816b-a42b` | 1,048,576 | no |
| `deeprelay/qwen3.5-9b` | 262,144 | yes |
| `deeprelay/qwen3.6-plus` | 1,000,000 | no |
| `deeprelay/qwen3.7-max` | 1,000,000 | no |
| `deeprelay/qwen3.7-plus` | 1,000,000 | no |
| `deeprelay/qwen3.8-flash` | 1,000,000 | no |
| `deeprelay/qwen3.8-max` | 1,000,000 | yes |
| `deeprelay/qwen4-max` | 1,000,000 | no |

## Disclosure

The plugin registers one model provider at import and nothing else: no tools, hooks, middleware,
commands, background work or file access. Chat requests (prompts, tool schemas and results, attached
images) and the model list go to `https://api.deeprelay.ai/v1` (or `DEEPRELAY_BASE_URL`) through
Hermes' own client with `DEEPRELAY_API_KEY`. Usage is billed to your deeprelay account. The plugin
itself opens no connections.

## Environment

| Variable | Purpose |
| --- | --- |
| `DEEPRELAY_API_KEY` | API key (required) |
| `DEEPRELAY_BASE_URL` | Override the API base, e.g. `https://api.demo.deeprelay.ai/v1` |

Docs: https://deeprelay.ai/docs · License: MIT
