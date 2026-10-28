# interpretability

The code behind "where the llama keeps its grammar", the sequel to "the
llama who chases the dogs". It trains a one block transformer on a
generated grammar, then looks inside it with a sparse autoencoder and
scores what it finds against the grammar's own labels.

```
mise exec -- mix deps.get
mise exec -- mix test
mise exec -- mix interp.toy      # toy model of superposition
mise exec -- mix interp.train    # train the llama model
mise exec -- mix interp.neurons  # how clean are the raw neurons?
mise exec -- mix interp.sae      # train and score a sparse autoencoder
```

See `CLAUDE.md` for the constraints and `docs/` for decisions and findings.
