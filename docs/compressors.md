# Compression backends

P2 includes runnable deterministic `truncate_tail` and seeded `random_drop`
adapters. `llmlingua2` and `longllmlingua` are black-box adapters: deployment
code must inject a verified installed implementation and target tokenizer.

CPC is recorded as an unavailable stub because no usable released CPC package
is installed in this environment. It is not silently substituted with another
algorithm.
Real LLMLingua-2 and LLMLingua adapters use the installed `PromptCompressor`
signature and measure realised rate with the injected target tokenizer. Install
`uv sync --extra real` and provide model weights locally; the default offline
tests use deterministic truncate/random backends and never download weights.
