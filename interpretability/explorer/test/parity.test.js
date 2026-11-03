// The page must run the same model the slides describe. These tests run
// the JavaScript forward pass on the exported weights and compare it with
// what Nx computed for the same sentences (fixtures.json, written by
// `mix interp.export`). The tolerance is 1e-4, absolute.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadTensors } from "../js/tensors.js";
import { forward } from "../js/model.js";
import { encode } from "../js/sae.js";
import { margin } from "../js/steering.js";

const tolerance = 1e-4;
const here = new URL(".", import.meta.url);

function load(name) {
  const manifest = JSON.parse(readFileSync(new URL(`../data/${name}.json`, here)));
  const bytes = readFileSync(new URL(`../data/${name}.bin`, here));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return { manifest, tensors: loadTensors(manifest, buffer) };
}

const model = load("model");
const sae = load("sae");
const activeCount = sae.manifest.active_count;
const fixtureFile = JSON.parse(readFileSync(new URL("fixtures.json", here)));
const fixtures = fixtureFile.sentences;
const steering = JSON.parse(readFileSync(new URL("../data/steering.json", here)));

function largestDifference(computed, expectedRows) {
  const width = expectedRows[0].length;
  let largest = 0;
  expectedRows.forEach((row, position) => {
    row.forEach((expected, column) => {
      largest = Math.max(largest, Math.abs(computed[position * width + column] - expected));
    });
  });
  return largest;
}

test("logits match Nx within 1e-4", () => {
  let largest = 0;
  for (const sentence of fixtures) {
    const { logits } = forward(model, sentence.ids);
    largest = Math.max(largest, largestDifference(logits, sentence.logits));
  }
  console.log(`largest logit difference: ${largest}`);
  assert.ok(largest <= tolerance, `largest logit difference ${largest}`);
});

test("neurons match Nx within 1e-4", () => {
  let largest = 0;
  for (const sentence of fixtures) {
    const { hidden } = forward(model, sentence.ids);
    largest = Math.max(largest, largestDifference(hidden, sentence.hidden));
  }
  assert.ok(largest <= tolerance, `largest neuron difference ${largest}`);
});

test("active features match Nx", () => {
  const width = 128;
  for (const sentence of fixtures) {
    const { hidden } = forward(model, sentence.ids);
    sentence.features.forEach((expectedPairs, position) => {
      const row = hidden.subarray(position * width, (position + 1) * width);
      const features = encode(sae.tensors, row, activeCount);
      const expected = new Map(expectedPairs);
      const active = new Map();
      features.forEach((value, index) => { if (value !== 0) active.set(index, value); });
      const cutoff = Math.min(...expected.values());

      for (const [index, value] of expected) {
        if (active.has(index)) {
          assert.ok(Math.abs(active.get(index) - value) <= tolerance, `feature ${index}`);
        } else {
          assert.ok(Math.abs(value - cutoff) <= tolerance, `feature ${index} missing`);
        }
      }
      for (const [index, value] of active) {
        if (!expected.has(index)) {
          assert.ok(Math.abs(value - cutoff) <= tolerance, `feature ${index} extra`);
        }
      }
    });
  }
});

test("keeps ties at the cutoff, like Nx", () => {
  // An SAE whose encoder is the identity: features equal the input, so
  // a row with its 8th and 9th largest values equal has 9 active.
  const width = 16;
  const identity = new Float32Array(width * width);
  for (let index = 0; index < width; index++) identity[index * width + index] = 1;
  const zeros = new Float32Array(width);
  const tiny = {
    encoder: { shape: [width, width], data: identity },
    encoder_bias: { shape: [width], data: zeros },
    decoder: { shape: [width, width], data: identity },
    decoder_bias: { shape: [width], data: zeros },
  };
  const row = Float32Array.from([9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 0.5, 0, 0, 0, 0, 0]);
  const active = encode(tiny, row, 8).filter((value) => value !== 0).length;
  assert.equal(active, 9);
});

test("a forward pass takes under 50 ms", () => {
  const vocabulary = model.manifest.vocabulary;
  const start = vocabulary.indexOf(model.manifest.start_token);
  const ids = [start, ...Array.from({ length: 19 }, (_, index) => (index * 7) % vocabulary.length)];
  forward(model, ids);
  const began = performance.now();
  for (let run = 0; run < 100; run++) forward(model, ids);
  const average = (performance.now() - began) / 100;
  console.log(`average forward pass: ${average.toFixed(2)} ms`);
  assert.ok(average < 50, `average ${average} ms`);
});

test("steering margins match Nx within 1e-3", () => {
  const coldOpen = fixtures[0].ids;
  for (const expected of fixtureFile.margins) {
    const direction = steering.directions.find(({ name }) => name === expected.direction);
    const strength = expected.multiple * steering.word_length;
    const computed = margin(model, coldOpen, Float32Array.from(direction.vector), strength, steering);
    assert.ok(
      Math.abs(computed - expected.margin) <= 1e-3,
      `${expected.direction} at ${expected.multiple}: ${computed} against ${expected.margin}`,
    );
  }
  const unpushed = fixtureFile.margins.find(({ multiple }) => multiple === 0);
  assert.equal(unpushed.margin.toFixed(3), "-5.023");
});

test("verb shares: an unpushed cold open favours singular present verbs", async () => {
  const { shares } = await import("../js/steering.js");
  const direction = Float32Array.from(steering.directions[0].vector);
  const unpushed = shares(model, fixtures[0].ids, direction, 0, steering);
  assert.ok(unpushed.singular > 0.5 && unpushed.plural < 0.05, JSON.stringify(unpushed));
  const pushed = shares(model, fixtures[0].ids, direction, steering.word_length, steering);
  assert.ok(pushed.plural > 0.5 && pushed.singular < 0.05, JSON.stringify(pushed));
});
