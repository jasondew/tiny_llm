# Talk outline: the llama who chases the dogs

Attention from scratch, in Elixir. A 60 minute slot: about 46 minutes of
talking, the rest for questions and slack.

This document is the design for the slide deck. It fixes the arc, the
timing, what each section must land, which math idea is introduced where,
and the three moments the Livebook is used. Slide-by-slide drafting happens
against this outline; the outline does not change without changing this
file.

## Goal

An attendee leaves able to explain, at a high level, how a transformer
works, with attention as the part they can actually describe: each
position looks back at earlier positions, scores them, and pulls in what
it needs. Code appears as evidence that nothing is hidden, not as the
lesson. Training gets one concept slide.

## The frame that carries the whole talk

State it by minute two and never let go of it:

> A language model is one function. Given the words so far, it gives a
> probability to every one of the 32 words that could come next.

Every model in the arc is that same function with more context. The
bigram sees one word. The neural bigram sees one word, learned. Attention
sees all of them. The transformer is attention plus a place to think. If
the audience holds this frame, attention stops being magic and becomes
"how the function gets to see further back."

## The primer, and where it goes

There is no primer section. Four math ideas appear at the moment a model
needs them, each one as the answer to a question the previous model just
raised. Nothing else is taught.

| idea | one line | appears in |
| --- | --- | --- |
| matrix as table | rows are things, columns are things, cells are numbers about the pair | bigram heatmap |
| vector as point | a list of floats is a location; near means similar | embeddings, PCA scatter |
| softmax | turn any list of scores into probabilities that sum to one | neural bigram output |
| dot product as similarity | multiply pairwise, add up; big when two vectors point the same way | attention scores |

Do not teach the word "tensor." The repo has none; matrices are lists of
lists of floats, and that is a feature of the talk. Do not teach matrix
multiplication as an operation. Teach it once as "every output is a
weighted sum of the inputs, and the weights are what gets learned," at the
neural bigram, and then stop saying it.

## The arc, with timing

| # | section | min | must land |
| --- | --- | --- | --- |
| 0 | cold open | 3 | the sentence, and the question it asks |
| 1 | the setup | 4 | 32 words, a grammar we own, the one function frame |
| 2 | bigram | 6 | counting works, until the answer is more than one word back |
| 3 | neural bigram | 7 | learning does not help if the context is the problem |
| 4 | attention | 12 | each position looks back and pulls in what it needs |
| 5 | the rest of the block | 6 | residual, MLP, and why depth would help |
| 6 | generating | 5 | the loop, and temperature |
| 7 | close | 3 | what is not here, and what is |

46 minutes. Sections 4 and 5 are the ones to protect. If the talk runs
long, section 6 shrinks to the slider alone and section 3 loses the PCA
scatter.

## Section by section

### 0. Cold open (3 min)

Slide: the sentence, alone, lowercase, with a blank.

    the llama who chases the dogs ____

Ask the room: flees or flee? Everyone knows. Nobody can say how they know
in fewer than a paragraph. The word that decides it is five back, and
there is a plural noun sitting right next to the blank pointing the wrong
way.

Slide: the same sentence with `llama` and `dogs` highlighted and a bracket
from the blank back to `llama`. That bracket is the talk. We are going to
build, from nothing, a program that draws it.

Slide: what "from nothing" means. Pure Elixir standard library. No Nx, no
Axon, no hex packages, `mix.exs` deps are empty. 15,104 parameters. Trains
in under 30 seconds on a laptop. Every gradient written out by hand and
checked. Repo link, which then stays in the footer of every slide.

### 1. The setup (4 min)

Slide: the vocabulary. All 32 words, grouped by part of speech, singular
and plural side by side. One word is one token is one integer; there is no
tokenizer. Point at `<start>` and `.`: sequences begin with one and end
with the other.

Slide: the grammar. Not the full production rules; three or four example
sentences with the structure visible, and the two rules that matter:
subjects agree with verbs, and a relative clause's verb agrees with the
head noun. Say why we own the training data: because then we know the
right answer to every question we ask the model, which no one does with a
real corpus.

Slide: the one function. Input: the words so far. Output: 32 probabilities.
Drawn as a box with a bar chart coming out of it. This slide comes back
three more times with a different box each time.

Slide: how we score it. Held-out loss is "how surprised the model is by
sentences it has not seen," lower is better. Two horizontal lines that
will appear on every loss chart: ln(32) = 3.466 is knowing nothing; 1.9021
is the best possible score for anything that sees only the previous word,
computed from the grammar. Do not explain entropy. Say "surprise" and move
on.

### 2. Bigram (6 min)

Primer beat: matrix as table.

Slide: count every pair of adjacent words in 2,000 sentences. Show the
counts for one row, say `chases`, as a list. Then the 32 by 32 heatmap
from `Bigram`, and read three cells out loud: `chases` is followed by
`the` or `a`, nothing else. `.` is followed by nothing. `llama` is
followed by a singular verb, `llamas` by a plural one.

Slide: the box again, with "count table" inside it. To predict, look up
the row for the last word and pick from it.

Slide: where it wins. It gets determiners, it gets the period, it gets
agreement when the noun is right there. Ten generated sentences, most of
them fine.

Slide: where it cannot. The probe. The bigram's context is `dogs` and
nothing else. Its row for `dogs` says `flee`. On the distractor case it
scores 55.3%, which is the base rate of the plural forms, so it is
guessing. Show the row.

Slide: the floor. 1.9021 is computed from the grammar itself: it is the
best score anything can reach seeing only the previous word, and a count
table with enough data sits on it. Nothing that sees one word can beat
this. That sentence is the setup for the next section's punchline.

### 3. Neural bigram (7 min)

Primer beats: vector as point, softmax, and the one sentence about matrix
multiplication.

Slide: the question. If counting is optimal, what does learning buy?
Answer up front: for this model, nothing. We build it anyway because it
introduces every part that survives into the transformer.

Slide: embeddings. Replace the word's integer with a row of 32 floats,
looked up from a table. The table starts random. A vector is a point;
words that behave the same should end up near each other, and the model
will move them there on its own.

Slide: the linear layer and softmax. The 32 floats go through one weighted
sum per output word ("every output is a weighted sum of the inputs, and
the weights are what gets learned") to give 32 scores, and softmax turns
scores into probabilities. The box again, with "lookup, weighted sums,
softmax" inside it. Show `Embedder.forward` if it fits on a slide; it
should.

Slide: training, the one concept slide. Loss is surprise at the right
answer. Every parameter has a slope: nudge it a little and the loss goes
up or down. Move every parameter a small step downhill. Repeat a few
hundred times. The slope is computed by hand in this repo, the derivation
is in `docs/backprop.md`, and a finite-difference test checks every one of
them. That is the whole of backprop for this talk. No chain rule on
screen.

Demo moment 1: train the Embedder live in the Livebook and watch the loss
chart. It starts at 3.466 and falls within seconds to about 1.944, just
above the 1.9021 line. It does not cross the line. It cannot.

Slide: PCA scatter of the learned embeddings. Nouns cluster tightly at one
end; the first axis is a noun detector. Adjectives and determiners cluster.
Verbs do not cluster at all, and that is honest: verb number lives in the
output table, not the input one, because nothing after a verb depends on
its number. Say that in one sentence, or cut the slide if time is short.

Slide: the punchline. On the distractor case the neural bigram scores
55.3%. Identical to the count table, to the decimal, because it sees the
same one word and gives the same answer. Learning was never the problem.
Context is the problem.

### 4. Attention (12 min)

Primer beat: dot product as similarity.

Slide: what we want. The blank needs to reach back to `llama`, past
`dogs`, past `the`, past `chases`. A fixed window would not do it; the
subject can be anywhere. What we want is for the blank to look at every
earlier word and decide for itself which ones matter.

Slide: dot product. Two vectors, multiply pairwise, add. Big when they
point the same way, near zero when unrelated. That is the whole primer
beat: a dot product is a similarity score, and we can learn what
"similar" should mean.

Slide: query, key, value, in words before symbols. Each position produces
three vectors from its embedding. A query is what this position is
looking for. A key is what this position is advertising. A value is what
this position hands over if chosen. Three weighted sums, three learned
weight tables, `Wq`, `Wk`, `Wv`.

Slide: scores. Take the blank's query, dot it against every earlier key.
That gives one number per earlier word. Divide by the square root of the
width so the numbers stay tame. Softmax across them: now it is a
probability of where to look. Show the row for the blank.

Slide: pulling in. Multiply each earlier position's value by its weight
and add them up. The blank now holds a blend of the words it chose to
look at. One more weighted sum, `Wo`, and the result goes on to the same
"weighted sums, softmax" output as before.

Slide: the causal mask. During training every position predicts the next
word at once, so each position must only see what came before it. Future
positions get a score of minus a billion before the softmax, which rounds
to zero attention. Show the triangle.

Slide: positions. Attention as described is a bag: it does not know that
`llama` came before `dogs`. So each position also has a learned vector
added to its word embedding. 16 positions, 16 vectors. Now the model can
tell "the noun near the start" from "the noun near the end."

Slide: the whole head as code. `Attention.forward`, or the six lines of
it that matter. This is where the "nothing is hidden" claim is cashed.

Demo moment 2: the attention heatmap for the probe, from the Livebook.
Rows are positions predicting, columns are positions looked at. The lower
triangle is filled, the upper is empty.

Slide: reading the heatmap honestly. The blank position attends 0.55 to
`who`, 0.17 to `dogs`, 0.13 to `llama`. It is not looking at `llama`. It
does not need to: on the head-only model it read the subject's number off
`chases`, which already agrees with the head noun, and this model has
found its own route. The point is not that it draws the bracket we
imagined; the point is that it is visibly structured, not flat, and it
gets the answer.

Slide: the number. On the distractor case, 77.4%, against 55.3% for both
bigrams. The one-word models could not do better than chance when a noun
intervened; this one can. Do not quote the 91.4% aggregate; a third of the
probes are free for every model.

Slide: the sink. The first verb position, with nothing useful behind it,
puts 0.96 of its attention on `<start>`. Production transformers do
exactly this and it has a name, attention sink. 15,104 parameters
reproduced it unprompted. One slide, one laugh, move on.

### 5. The rest of the block (6 min)

Slide: the architecture, as a diagram. Embedding plus position, then
attention, then MLP, then the output. Two arrows that go around: the
residuals. Two small boxes: the norms. This is the box from section 1 with
its lid off.

Slide: residual connections. Instead of replacing the position's vector
with what attention returned, add it. Keep what you had, add what you
learned. Without this, information at the input has to survive every
layer to reach the output; with it, the default is to pass through.

Slide: the MLP. Attention gathers; it cannot compute much about what it
gathered. The MLP is two weighted sums with a ReLU between, applied to
each position on its own. 32 wide in, 128 in the middle, 32 out. Half the
parameters in the model are here. Call it "a place to think about what
you gathered" and do not go further.

Slide: RMSNorm, one line. Rescale each position's vector to a fixed size
before attention and before the MLP so nothing blows up. Learned gain.
Skip the formula.

Demo moment 2, revisited: the `who` row of the same heatmap. `who`
attends 0.57 to `llama`. On the mirror probe, `the dogs who chase the
llama`, it attends 0.65 to `dogs`. The `who` position has gathered the
head noun into itself. The blank attends to `who`. So the model has built
half of a two-hop route, head noun into `who`, `who` into the blank, and
with one block it cannot use the second hop, because both hops happen at
once. A second block would read `who` after it had already gathered the
subject. That is what depth buys, and the model drew the argument for it
by itself.

Slide: scale. Same shape, bigger numbers. One head becomes many, one block
becomes dozens, 32 words becomes a hundred thousand tokens and a
tokenizer, 16 positions becomes a hundred thousand. Nothing on this list
is a new idea. Do not put a parameter count for a frontier model on the
slide; put the ratio in the speaker notes and say it out loud.

### 6. Generating (5 min)

Slide: the loop. Start with `<start>`. Ask the function. Pick a word.
Append it. Ask again. Stop at `.`. In Elixir this is `Stream.unfold`, and
it fits on the slide. Say the uncomfortable part: no state carries between
steps, the whole prefix is re-read every time, and that is why a model
cannot take back something it has already said.

Slide: temperature. Divide the scores by a number before the softmax.
Below 1 sharpens toward the top choice, above 1 flattens toward uniform,
zero is argmax.

Demo moment 3: the slider in the Livebook, range 0 to 3. At 0 it says one
sentence forever, `the fast dogs and the llamas are fast .`, which is not
in the training set. At 1 it is 89.5% grammatical and 90.5% distinct. At 3
it is word salad. Show the failure modes coming apart in order:
`a llama is dog` loses meaning first, `sleepy fast goose dogs flee` loses
the determiner last. Structure goes before content.

Slide: the trade-off in one chart, grammaticality against distinctness as
temperature rises. Temperature is the dial between correct-and-boring and
varied-and-wrong. This is the slide people photograph.

### 7. Close (3 min)

Slide: what is not here. No autodiff, no tokenizer, no GPU, no
multi-head, no depth, no dependencies. And what is here: embeddings,
learned positions, scaled dot-product attention, causal mask, residuals,
RMSNorm, an MLP, cross-entropy, hand-written backprop with a
finite-difference check, temperature sampling. Every one of these is the
same thing a frontier model does.

Slide: the sentence again, with the bracket drawn, and the probability
the model puts on `flees`. Repo link, large.

## The three demo moments

The Livebook is scripted to exactly three cells. It is not opened and
scrolled.

1. Section 3: train the Embedder live and watch the loss fall to the
   floor and stop. About ten seconds.
2. Section 4 and 5: the attention heatmap for the probe, already
   rendered from a checkpoint. The same cell is read twice, once for the
   blank's row and once for the `who` row.
3. Section 6: the temperature slider.

Everything else is a pre-rendered image from a fixed seed, with the seed
in the speaker notes. The transformer is trained before the talk from a
checkpoint in `priv/checkpoints/`, not live: 27 seconds of a loss curve
is a beat once, in section 3, and dead air the second time.

If the Livebook fails, each demo moment has a static slide behind it with
the same picture. The talk does not depend on the notebook.

## Numbers the talk quotes, and where they come from

Every number states its provenance in the speaker notes. From
`docs/build-brief.md` unless noted.

| number | what | source |
| --- | --- | --- |
| 32, 16, 32, 1, 1 | vocab, context, width, heads, blocks | fixed design |
| 15,104 | parameters | counted from `Transformer.init/1` |
| 3.466 | ln(32), knowing nothing | arithmetic |
| 1.9021 | bigram floor, H(next given previous) | computed from the grammar |
| 1.9436 / 1.5842 | held-out loss, Embedder / Transformer | stage 6 |
| 55.3% / 55.3% / 77.4% | distractor row agreement, bigram / Embedder / Transformer | stage 6, 420 probes |
| 0.55 / 0.17 / 0.13 | blank attends to `who` / `dogs` / `llama` | stage 7, stage 5 model |
| 0.57 / 0.65 | `who` attends to head noun, probe / mirror | stage 7 |
| 0.96 | first verb position on `<start>` | stage 7 |
| 100 / 89.5 / 29.5% | grammatical at T = 0 / 1 / 3 | stage 7 |
| 0.5 / 90.5% | distinct at T = 0 / 1 | stage 7 |
| 64.0 / 90.6 / 54.6% | generated: unseen / grammatical / both | stage 6, 1,000 sentences |

The transformer numbers come from batch 8, 500 steps, cosine decay from
0.5, the Livebook's config. Attention weights are from one seed and should
be re-read from the checkpoint the talk ships with, since they move by a
few hundredths between seeds.

## Deliberately left out

- Backprop beyond one slide. `docs/backprop.md` is the backup if asked.
- Cross-entropy as a formula. "Surprise" is enough.
- Multi-head attention. Mentioned once on the scale slide.
- Why RMSNorm rather than LayerNorm, why untied output table, learning
  rate schedules, batch size. All in the brief for anyone who asks.
- The training harness, `Train` and `Config`. One sentence in section 3.
- The grammar's production rules. Examples instead.

## Open decisions

- Slide tool. Not chosen. Requirements: code blocks with syntax
  highlighting, image slides from PNGs the Livebook exports, a footer
  with the repo link on every slide, speaker notes.
- Whether section 3's PCA scatter survives. It is the first cut if the
  talk runs long in rehearsal.
- Which checkpoint ships in `priv/checkpoints/`, and re-reading the
  attention numbers above from it once chosen.
