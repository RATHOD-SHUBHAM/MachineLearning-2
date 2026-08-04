# Decoder LLMs (Next-Token Prediction)

## Say this (60–90 sec)
Decoder-only LLMs like GPT are autoregressive language models. They predict the next token given all previous tokens — left to right, causal attention masks future positions. Training: take a text sequence, feed tokens 1 through t, predict token t+1, accumulate cross-entropy loss over all positions. At inference, generate one token at a time: sample or greedy pick from softmax over vocabulary, append to context, repeat. Context window is the max sequence length the model can attend to — 4k, 32k, 128k depending on model. Everything — chat, code, summarization — is framed as completing text. Chat templates wrap user/assistant turns into one token sequence the model continues. Scaling laws: bigger model plus more data plus more compute predictably improves loss and downstream ability. Decoder LLMs are general pattern completers — quality depends on pretrain data and alignment.

## Why it matters
The default GenAI architecture. Must explain autoregressive training, inference loop, and why "completion" unifies tasks.

## How it works
- **Architecture**: stacked decoder blocks with masked self-attention only.
- **Training loss**: `-log P(x_t | x_<t)` averaged over tokens and batch.
- **Inference**: autoregressive loop; KV cache stores past keys/values for speed.
- **Sampling**: temperature, top-k, top-p control randomness vs determinism.
- **Chat format**: system/user/assistant tokens — model learns to continue as assistant.

## Tradeoffs
- Use when: open-ended generation, chat, code, any task expressible as text continuation.
- Avoid when: need bidirectional context for encoding only — encoder models (BERT) or encoder-decoder (T5) may be better/cheaper.

## If they dig deeper
- Teacher forcing in training vs free-running generation — exposure bias.
- Speculative decoding — draft model proposes tokens, target model verifies — faster inference.
- Emergent abilities — debated; often correlate with scale and eval setup.
