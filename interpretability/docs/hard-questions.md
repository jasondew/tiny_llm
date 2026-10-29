# Hard questions

Honest answers for the Q&A, filled in as results arrive. An answer marked
"pending a result" has no evidence behind it yet.

## Aren't you just finding the concepts you put into the grammar?

Starting position to defend: the grammar supplies labels to check
against, but the model never saw them. It saw only words, and it had to
discover these concepts for itself in order to predict the next word.

Pending a result.

## What about dead features, or features that fire on everything?

31 of the 512 features never fire on the evaluation sentences (6%), and
none fires on more than half of all words. Dead features are wasted
capacity, not wrong answers. They still count in the 512: "23 of 512
features are clean" includes them in the denominator.

## How do we know a feature belongs to the model and is not an invention of the SAE?

Splice test: replace the model's 128 neurons with the SAE's rebuilt
version and see how much worse it predicts. On held out sentences the
loss is 2.0096 with the real neurons and 2.0095 with the SAE's version,
while switching the neurons off gives 2.7052, so the SAE keeps all of
what the neurons contribute. (Switching them off is the harsher baseline
of the two usual ones: replacing them with their average gives 4.55.) If its features were
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
while the SAE found 23 clean features covering 22 labels. The honest
summary: on single words and grammar words the SAE clearly beats the
neurons (67 of 67 words); on abstract grammar concepts the best neuron
beats the best feature on 12 of 25 labels and loses on 7, because the
SAE splits each concept into many precise slices.

## Your SAE uses TopK. Towards Monosemanticity used an L1 penalty. Why?

TopK keeps each word's 8 strongest features, so the sparsity is a number
you can state, with nothing to tune. It is a later variant of the same
idea. It changes how sparsity is enforced, not what the SAE is for.

## Did the model just memorise the cold open?

No. Every probe sentence is checked against the 60,000 training
sentences and dropped if it appeared there. The cold open case scores
0.9911 on 448 sentences the model never saw.

## Isn't it unfair to compare the best of 512 features with the best of 128 neurons?

Partly. Four times as many candidates gives the features more chances,
which helps most on single words. It cannot explain the grammar labels,
where the neurons still win more often with fewer candidates.
