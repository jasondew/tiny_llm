# Hard questions

Honest answers for the Q&A, filled in as results arrive. An answer marked
"pending a result" has no evidence behind it yet.

## Aren't you just finding the concepts you put into the grammar?

Starting position to defend: the grammar supplies labels to check
against, but the model never saw them. It saw only words, and it had to
discover these concepts for itself in order to predict the next word.

Pending a result.

## What about dead features, or features that fire on everything?

32 of the 512 features never fire on the evaluation sentences (6%), and
none fires on more than half of all words. Dead features are wasted
capacity, not wrong answers; they are left out of every score.

## How do we know a feature belongs to the model and is not an invention of the SAE?

Splice test: replace the model's 128 neurons with the SAE's rebuilt
version and see how much worse it predicts. Loss goes from 1.9635 to
1.9657, while switching the neurons off entirely gives 2.7664, so the SAE
keeps 99.7% of what the neurons contribute. If its features were
inventions, the model would notice. Steering is the second test: turning
a feature up should change what the model writes in the way its label
says.

## Does this work on real models?

Pending a result.

## Isn't superposition contested?

Yes. Goodfire's "Neural Networks Are Already Clean" (Cammarata, 2026)
argues most neurons in real models have one job and that messiness usually
means a weak rubric or a missing label. This model tested both halves.
A missing label was real: adding "the sentence could end here" made it
the best label for 23 neurons. But no neuron became clean (0 of 128),
while the SAE found 33 clean features. The honest summary: on single
words and grammar words the SAE clearly beats the neurons; on abstract
grammar concepts the best neuron usually beats the best feature, because
the SAE splits each concept into many precise slices.

## Your SAE uses TopK. Towards Monosemanticity used an L1 penalty. Why?

TopK keeps each word's 8 strongest features, so the sparsity is a number
you can state, with nothing to tune. It is a later variant of the same
idea. It changes how sparsity is enforced, not what the SAE is for.
