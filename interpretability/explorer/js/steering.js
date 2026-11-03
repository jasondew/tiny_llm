// Steering: push the neurons along one direction and see whether the
// model changes its mind about agreement.
//
// The push is added to all 128 neurons at every word, before the rest of
// the model sees them, exactly as `Interpretability.Steering` does. The
// *margin* is the best plural present verb's score minus the best
// singular one's, after the last word: below zero the model would say
// "flees", above zero "flee".

import { forward } from "./model.js";

export function margin(model, ids, direction, strength, { plural_verbs: plural, singular_verbs: singular }) {
  const push = direction.map((value) => value * strength);
  const { logits } = forward(model, ids, push);
  const vocabulary = model.manifest.vocabulary;
  const last = logits.subarray((ids.length - 1) * vocabulary.length, ids.length * vocabulary.length);
  const best = (words) => Math.max(...words.map((word) => last[vocabulary.indexOf(word)]));
  return best(plural) - best(singular);
}

// The probability the model puts on plural and on singular present tense
// verbs after the last word. The margin can flip just because singular
// verbs collapse while the model moves to words that carry no number
// ("who", past tense verbs), so the page shows these shares too.
export function shares(model, ids, direction, strength, { plural_verbs: plural, singular_verbs: singular }) {
  const push = direction.map((value) => value * strength);
  const { logits } = forward(model, ids, push);
  const vocabulary = model.manifest.vocabulary;
  const last = Array.from(logits.subarray((ids.length - 1) * vocabulary.length, ids.length * vocabulary.length));
  const largest = Math.max(...last);
  const exponentials = last.map((value) => Math.exp(value - largest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const share = (words) => words.reduce((sum, word) => sum + exponentials[vocabulary.indexOf(word)] / total, 0);
  return { plural: share(plural), singular: share(singular) };
}
