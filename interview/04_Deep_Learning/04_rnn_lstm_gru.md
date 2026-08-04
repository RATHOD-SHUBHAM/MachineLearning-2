# RNN, LSTM, GRU

## Say this (60–90 sec)
Recurrent networks process sequences one step at a time, maintaining a hidden state that carries information forward. At each timestep: new input plus previous hidden state produce output and updated hidden state. Plain RNNs share weights across time but suffer from vanishing gradients — they forget long-range dependencies. LSTM fixes this with a cell state and gates: forget gate drops old info, input gate adds new info, output gate decides what to expose. GRU is a lighter variant — reset and update gates, no separate cell state. Both learn what to remember and forget. RNNs are naturally suited for variable-length sequences: time series, text before transformers, speech. Downsides: hard to parallelize over time, still struggle on very long context compared to attention. Bidirectional RNNs read forward and backward for encoding; generation usually uses unidirectional.

## Why it matters
Historical backbone for sequences; gate mechanics foreshadow transformers. Interviewers still ask LSTM vs GRU and why attention replaced them for NLP.

## How it works
- **RNN**: `h_t = tanh(W_h h_{t-1} + W_x x_t + b)`. Same W at every step.
- **Vanishing**: repeated `W_h` multiplication shrinks/explodes signal over long T.
- **LSTM gates**: forget `f`, input `i`, output `o`; cell `c_t = f ⊙ c_{t-1} + i ⊙ candidate`.
- **GRU**: update gate blends old/new hidden; reset gate controls how much past to use.
- **Many-to-one**: whole sequence → one label (sentiment). **One-to-many**: one input → sequence (captioning).

## Tradeoffs
- Use when: short–medium sequences, streaming/online inference, small models on device, time series baselines.
- Avoid when: very long context, need full parallel training at scale — use transformer.

## If they dig deeper
- Teacher forcing during training — feed ground-truth previous token; exposure bias at inference.
- Seq2seq + attention — bridge between RNNs and full transformer.
- Stacked/bidirectional RNNs — deeper and context from both directions for encoding.
