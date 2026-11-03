# The llama grammar explorer

A static page that runs the llama model in the browser. Type or pick a
sentence, tap a word to see what the model expects next and which SAE
features (or raw neurons) are firing, then tap a unit to see its best
matching grammar labels and the evaluation words it fired on hardest.

## Run it locally

    python3 -m http.server 8765 -d explorer

then open <http://localhost:8765/>. It needs a server, not a `file://`
URL, because it fetches its data.

## Where the data comes from

`mix interp.export` (run from `interpretability/`) writes `data/` from
the committed checkpoints `priv/checkpoints/model.etf` and
`priv/checkpoints/sae_512.etf`, and writes `test/fixtures.json`, the Nx
outputs the parity test checks against. Rerun it whenever a checkpoint
changes, then rerun the tests.

## Tests

    node --test 'explorer/test/*.test.js'

The parity test checks the JavaScript forward pass against Nx to within
1e-4 (the largest difference measured is 2.4e-6), and the SAE's active
features against Nx's.

## Size

About 735 KB gzipped in total, almost all of it `data/sae.bin` (the 512
feature SAE's weights, 480 KB gzipped). A forward pass takes about half
a millisecond.

## Files

- `js/model.js`: the forward pass, step for step as `Interpretability.Model`
- `js/sae.js`: the SAE encoder, as `Interpretability.SAE.encode`
- `js/sentence.js`: typed text to word ids, with messages for unknown words
- `js/tensors.js`: the exported binary back into named tensors
- `js/main.js`: fetching and drawing; no math
