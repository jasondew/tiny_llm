import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parse } from "../js/sentence.js";

const model = JSON.parse(readFileSync(new URL("../data/model.json", import.meta.url)));
const { vocabulary, start_token: startToken, context_length: contextLength } = model;
const read = (text) => parse(text, vocabulary, startToken, contextLength);

test("a typed sentence becomes words and ids after the start token", () => {
  const result = read("The llama who chases the dogs.");
  assert.equal(result.ok, true);
  assert.deepEqual(result.words, ["the", "llama", "who", "chases", "the", "dogs"]);
  assert.deepEqual(result.ids, [vocabulary.indexOf(startToken), ...result.words.map((word) => vocabulary.indexOf(word))]);
});

test("extra spaces and punctuation are ignored", () => {
  const result = read("  the   llama, sleeps!?  ");
  assert.deepEqual(result.words, ["the", "llama", "sleeps"]);
});

test("unknown words are named", () => {
  assert.equal(vocabulary.includes("horses"), false);
  const result = read("the llama who chases the horses");
  assert.equal(result.ok, false);
  assert.match(result.message, /horses/);
});

test("the start token cannot be typed as a word", () => {
  const result = read("<start> the llama");
  assert.equal(result.ok, false);
  assert.match(result.message, /<start>/);
});

test("at most 19 words fit", () => {
  const result = read(Array(20).fill("llama").join(" "));
  assert.equal(result.ok, false);
  assert.match(result.message, /19 words/);
  assert.equal(read(Array(19).fill("llama").join(" ")).ok, true);
});

test("an empty sentence asks for one", () => {
  assert.deepEqual(read("   "), { ok: false, message: "Type a sentence, or pick one below." });
});
