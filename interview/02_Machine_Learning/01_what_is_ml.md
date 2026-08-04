# What Is Machine Learning

## Say this (60–90 sec)
Machine learning is a way to build systems that improve from data instead of being hand-coded for every rule. You show the model examples — inputs and, usually, desired outputs — and it learns patterns that generalize to new cases. The core loop is: collect data, choose a model family, train it to minimize error on that data, then evaluate on held-out examples you did not train on. Unlike traditional software where you write explicit if-then logic, ML discovers a function that maps inputs to outputs. That makes it powerful for problems where rules are hard to write — vision, language, fraud, recommendations — but it also means performance depends on data quality, representativeness, and how well you measure success. I treat ML as applied statistics plus engineering: learn from data, but validate rigorously before deploying.

## Why it matters
Interviewers use this to see whether you understand ML as a process, not a buzzword. It sets up everything else — features, training splits, metrics, and why models fail in production.

## How it works
- **Traditional programming**: rules + data → output.
- **Machine learning**: data + desired outputs → rules (the learned model).
- **Training**: adjust internal parameters so predictions match labels on training examples.
- **Generalization**: the goal is low error on new, unseen data — not memorizing the training set.
- **Pipeline**: problem framing → data → features → model → train → evaluate → deploy → monitor.

## Tradeoffs
- Use when: patterns exist in data, labels or structure are available, and hand-crafted rules are brittle or expensive.
- Avoid when: you need guaranteed correctness, data is tiny or non-representative, or a simple rule or lookup table suffices.

## If they dig deeper
- Difference from AI vs DL vs ML — ML is the umbrella; DL is a subset using neural networks.
- Why data drift matters — the world changes; a model trained on last year’s data may degrade.
- ML is not magic — garbage in, garbage out; no amount of tuning fixes bad labels or leakage.
