import { test } from "node:test";
import assert from "node:assert/strict";
import { describeUnit } from "../js/units.js";

const labelled = {
  firing_rate: 0.05,
  top: [{ label: "agreement number: plural", precision: 0.99, recall: 0.23 }],
  examples: [{ activation: 11.6, words: ["<start>", "the", "owls"], position: 2 }],
};

test("a unit with examples shows its best label and its peak", () => {
  assert.deepEqual(describeUnit(labelled), {
    label: "agreement number: plural",
    scores: "precision 0.99, recall 0.23",
    peak: 11.6,
  });
});

test("a unit that never fired on the evaluation sentences gets no label and no scale", () => {
  const unseen = { firing_rate: 0, top: [{ label: "after main verb: false", precision: 0, recall: 0 }], examples: [] };
  assert.deepEqual(describeUnit(unseen), {
    label: "No label: it never fired on the evaluation sentences",
    scores: "",
    peak: null,
  });
});
