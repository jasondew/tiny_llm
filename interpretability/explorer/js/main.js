// The page: loads the exported model, runs it on a sentence, and draws
// what it finds. All the math lives in model.js, sae.js and sentence.js;
// this file only fetches data and draws.

import { loadTensors } from "./tensors.js";
import { forward } from "./model.js";
import { encode } from "./sae.js";
import { parse, typeableWords } from "./sentence.js";
import { describeUnit } from "./units.js";
import { margin, shares } from "./steering.js";

const presets = [
  "the llama who chases the dogs",
  "the dogs near the llama chase the cat",
  "the geese that the fox chased were hungry",
];
const neuronListLength = 12;
const predictionCount = 5;

const state = { kind: "features", run: null, position: 0, unit: null, direction: 1, push: 0 };
let data;

async function fetchJson(name) {
  const response = await fetch(`data/${name}`);
  if (!response.ok) throw new Error(`Could not load ${name}`);
  return response.json();
}

async function fetchTensors(name) {
  const [manifest, buffer] = await Promise.all([
    fetchJson(`${name}.json`),
    fetch(`data/${name}.bin`).then((response) => response.arrayBuffer()),
  ]);
  return { manifest, tensors: loadTensors(manifest, buffer) };
}

function softmax(scores) {
  const largest = Math.max(...scores);
  const exponentials = scores.map((score) => Math.exp(score - largest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "text") node.textContent = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function meter(className, fraction) {
  const fill = element("span");
  fill.style.width = `${Math.min(fraction, 1) * 100}%`;
  return element("span", { class: className }, [fill]);
}

function units() {
  return state.kind === "features" ? data.features : data.neurons;
}

function unitName(index) {
  return state.kind === "features" ? `Feature ${index}` : `Neuron ${index}`;
}

// The unit's activation at every word, scaled by the most it ever reached
// on the evaluation sentences, so bars mean the same thing in every
// sentence.
function unitValues(position) {
  return state.kind === "features" ? state.run.features[position] : state.run.neurons[position];
}

// A unit that never fired on the evaluation sentences has no peak, so its
// bars are scaled by the most it reaches in this sentence instead.
function peak(unit) {
  const { peak: evaluationPeak } = describeUnit(units()[unit]);
  if (evaluationPeak !== null) return evaluationPeak;
  const sentencePeak = Math.max(...state.run.words.map((_word, position) => unitValues(position)[unit]));
  return sentencePeak > 0 ? sentencePeak : 1;
}

function run(words, ids) {
  const { logits, hidden } = forward(data.model, ids);
  const vocabularySize = data.model.manifest.vocabulary.length;
  const width = 128;
  const neurons = [];
  const features = [];
  const predictions = [];

  // Position 0 is the start token; word i sits at position i + 1.
  for (let position = 1; position < ids.length; position++) {
    const row = hidden.slice(position * width, (position + 1) * width);
    neurons.push(row);
    features.push(encode(data.sae.tensors, row, data.sae.manifest.active_count));
    predictions.push(softmax(Array.from(logits.subarray(position * vocabularySize, (position + 1) * vocabularySize))));
  }
  return { words, ids, neurons, features, predictions };
}

function strongestUnit(position) {
  const values = unitValues(position);
  let best = 0;
  values.forEach((value, index) => { if (value > values[best]) best = index; });
  return values[best] > 0 ? best : null;
}

function read(text) {
  const { vocabulary, start_token: startToken, context_length: contextLength } = data.model.manifest;
  const result = parse(text, vocabulary, startToken, contextLength);
  document.querySelector("#sentence-message").textContent = result.ok ? "" : result.message;
  if (!result.ok) return;

  state.run = run(result.words, result.ids);
  state.position = result.words.length - 1;
  state.unit = strongestUnit(state.position);
  draw();
}

function drawSentence() {
  const container = document.querySelector("#sentence");
  container.replaceChildren(
    ...state.run.words.map((word, position) => {
      const fill = element("span");
      const value = state.unit === null ? 0 : unitValues(position)[state.unit];
      fill.style.height = `${Math.min(value / peak(state.unit ?? 0), 1) * 100}%`;
      return element(
        "button",
        {
          type: "button",
          class: "word",
          "aria-pressed": String(position === state.position),
          "aria-label": `${word}, ${state.unit === null ? "" : `${unitName(state.unit)} at ${value.toFixed(2)}`}`,
          onclick: () => { state.position = position; draw(); },
        },
        [element("span", { text: word }), element("span", { class: "bar" }, [fill])],
      );
    }),
  );
}

function predictionItems(probabilities) {
  const vocabulary = data.model.manifest.vocabulary;
  return probabilities
    .map((probability, id) => ({ word: vocabulary[id], probability }))
    .sort((first, second) => second.probability - first.probability)
    .slice(0, predictionCount)
    .map(({ word, probability }) =>
      element("li", {}, [
        element("span", { text: word }),
        meter("meter", probability),
        element("span", { class: "percent", text: `${(probability * 100).toFixed(1)}%` }),
      ]),
    );
}

function drawPredictions() {
  document.querySelector("#prediction-heading").textContent = `After “${state.run.words[state.position]}”, the model expects`;
  document.querySelector("#predictions").replaceChildren(...predictionItems(state.run.predictions[state.position]));
}

function drawSteering() {
  const { directions, word_length: wordLength } = data.steering;
  const direction = Float32Array.from(directions[state.direction].vector);
  const strength = state.push * wordLength;
  const ids = state.run.ids;
  const lastWord = state.run.words[state.run.words.length - 1];

  document.querySelector("#steering-directions").replaceChildren(
    ...directions.map(({ name, description }, index) =>
      element("button", {
        type: "button",
        text: name,
        title: description,
        "aria-pressed": String(index === state.direction),
        onclick: () => { state.direction = index; drawSteering(); },
      }),
    ),
  );
  document.querySelector("#steering-push-value").textContent = state.push.toFixed(2);

  const difference = margin(data.model, ids, direction, strength, data.steering);
  const { plural, singular } = shares(data.model, ids, direction, strength, data.steering);
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  document.querySelector("#steering-verdict").textContent =
    `After “${lastWord}”: plural present verbs ${percent(plural)}, singular ${percent(singular)}.`;
  document.querySelector("#steering-margin").textContent =
    `Between the best present tense verbs, plural ${difference > 0 ? "leads" : "trails"} by ${Math.abs(difference).toFixed(2)} in score. Watch the rest of the list: a push can drown out singular verbs without making plural ones likely.`;

  const { logits } = forward(data.model, ids, direction.map((value) => value * strength));
  const vocabularySize = data.model.manifest.vocabulary.length;
  const last = Array.from(logits.subarray((ids.length - 1) * vocabularySize, ids.length * vocabularySize));
  document.querySelector("#steering-predictions").replaceChildren(...predictionItems(softmax(last)));
}

function drawUnits() {
  const values = unitValues(state.position);
  let active = Array.from(values)
    .map((value, index) => ({ index, value }))
    .filter(({ value }) => value > 0)
    .sort((first, second) => second.value - first.value);
  if (state.kind === "neurons") active = active.slice(0, neuronListLength);

  document.querySelectorAll(".toggle button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.kind === state.kind));
  });

  const list = document.querySelector("#units");
  if (active.length === 0) {
    list.replaceChildren(element("li", { class: "empty", text: "Nothing fires at this word." }));
    return;
  }

  list.replaceChildren(
    ...active.map(({ index, value }) => {
      const description = describeUnit(units()[index]);
      return element("li", {}, [
        element(
          "button",
          {
            type: "button",
            class: "unit",
            "aria-pressed": String(index === state.unit),
            onclick: () => {
              state.unit = index;
              draw();
              // On a phone the detail panel sits below the list; bring it up.
              if (window.matchMedia("(max-width: 52rem)").matches) {
                document.querySelector("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
              }
            },
          },
          [
            element("span", { class: "name", text: unitName(index) }),
            description.peak === null ? element("span") : meter("strength", value / description.peak),
            element("span", { class: "label", text: description.label }),
            element("span", { class: "scores", text: description.scores }),
          ],
        ),
      ]);
    }),
  );
}

function drawExample(example) {
  const words = example.words.slice(1);
  const position = example.position - 1;
  return element("li", {}, [
    words.slice(0, position).join(" ") + (position > 0 ? " " : ""),
    element("strong", { text: words[position] }),
    element("span", { class: "after", text: words.length > position + 1 ? ` ${words.slice(position + 1).join(" ")}` : "" }),
  ]);
}

function drawDetail() {
  const detail = document.querySelector("#detail");
  if (state.unit === null) {
    detail.replaceChildren(element("p", { class: "empty", text: "Tap a unit to see what it is about." }));
    return;
  }

  const unit = units()[state.unit];
  const children = [element("h2", { text: unitName(state.unit) })];

  if (unit.examples.length === 0) {
    children.push(element("p", { text: "This unit never fires on the evaluation sentences." }));
    detail.replaceChildren(...children);
    return;
  }

  children.push(
    element("p", { class: "scores", text: `Fires on ${(unit.firing_rate * 100).toFixed(1)}% of words in the evaluation sentences. Its best matching grammar labels:` }),
    element("ol", { class: "labels" }, unit.top.map((grade) =>
      element("li", { text: `${grade.label}: precision ${grade.precision.toFixed(2)}, recall ${grade.recall.toFixed(2)}` }),
    )),
    element("h2", { text: "Where it fired hardest" }),
    element("p", { class: "scores", text: "Each line is an evaluation sentence; the bold word is where it fired, and the grey words came next." }),
    element("ul", { class: "examples" }, unit.examples.map(drawExample)),
  );
  detail.replaceChildren(...children);
}

function draw() {
  drawSentence();
  drawPredictions();
  drawUnits();
  drawDetail();
  drawSteering();
}

// Until the model has loaded, the form must not submit itself, which
// would reload the page and start the download again.
document.querySelector("#sentence-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (data) read(document.querySelector("#sentence-input").value);
});

async function start() {
  const [model, sae, neurons, features, steering] = await Promise.all([
    fetchTensors("model"),
    fetchTensors("sae"),
    fetchJson("neurons.json"),
    fetchJson("features.json"),
    fetchJson("steering.json"),
  ]);
  data = { model, sae, neurons, features, steering };

  document.querySelector("#steering-push").addEventListener("input", (event) => {
    state.push = Number(event.target.value);
    drawSteering();
  });

  const input = document.querySelector("#sentence-input");
  document.querySelector("#sentence-form button[type=submit]").disabled = false;
  document.querySelector("#sentence-message").classList.remove("loading");
  document.querySelector("#vocabulary-list").textContent =
    typeableWords(model.manifest.vocabulary, model.manifest.start_token).join(", ");

  document.querySelector("#presets").replaceChildren(
    ...presets.map((sentence) =>
      element("button", { type: "button", text: sentence, onclick: () => { input.value = sentence; read(sentence); } }),
    ),
  );

  document.querySelectorAll(".toggle button").forEach((button) => {
    button.addEventListener("click", () => {
      state.kind = button.dataset.kind;
      state.unit = strongestUnit(state.position);
      draw();
    });
  });

  input.value = presets[0];
  read(presets[0]);
}

start().catch((error) => {
  document.querySelector("#sentence-message").textContent = `The model did not load: ${error.message}. Reload the page to try again.`;
});
