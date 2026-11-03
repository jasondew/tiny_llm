// Turning what someone typed into the model's word ids.
//
// The model knows exactly 70 words, one id each, and reads at most 20
// ids, the first of which is always the start token. So typed text is
// lowercased, punctuation is dropped, and every word must be one the
// model knows. Anything else gets a message that says what went wrong.

export function parse(text, vocabulary, startToken, contextLength) {
  const words = text.toLowerCase().replace(/[.,!?;:]/g, " ").split(/\s+/).filter(Boolean);
  const room = contextLength - 1;

  if (words.length === 0) return { ok: false, message: "Type a sentence, or pick one below." };

  const unknown = [...new Set(words.filter((word) => word === startToken || !vocabulary.includes(word)))];
  if (unknown.length > 0) {
    return { ok: false, message: `The model doesn't know: ${unknown.join(", ")}` };
  }

  if (words.length > room) {
    return { ok: false, message: `At most ${room} words fit; this has ${words.length}.` };
  }

  return { ok: true, words, ids: [startToken, ...words].map((word) => vocabulary.indexOf(word)) };
}
