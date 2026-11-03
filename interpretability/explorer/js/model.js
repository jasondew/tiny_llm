// The llama model's forward pass, in plain JavaScript.
//
// This is `Interpretability.Model`'s `run`, step for step, so the page
// shows the same model the slides describe (the parity test checks it to
// within 1e-4 of Nx):
//
//     x       = embeddings[word] + positions[place]
//     r       = x + attention(rmsnorm(x, gain1))       one causal head
//     hidden  = relu(rmsnorm(r, gain2) · weight1 + bias1)   the 128 neurons
//     y       = r + hidden · weight2 + bias2
//     logits  = rmsnorm(y, gain3) · projection
//
// Everything works on one sentence at a time, one position at a time.
// The weights are float32, but the sums run in ordinary JavaScript
// numbers (float64), which is why the results agree with Nx closely but
// not bit for bit.

// `vector` (length rows) times `matrix` (rows × columns), as a new array.
function times(vector, matrix) {
  const [rows, columns] = matrix.shape;
  const result = new Float64Array(columns);
  for (let row = 0; row < rows; row++) {
    const value = vector[row];
    if (value === 0) continue;
    const start = row * columns;
    for (let column = 0; column < columns; column++) {
      result[column] += value * matrix.data[start + column];
    }
  }
  return result;
}

function plus(vector, other) {
  return vector.map((value, index) => value + other[index]);
}

function rmsnorm(vector, gain, epsilon) {
  let squares = 0;
  for (const value of vector) squares += value * value;
  const rootMeanSquare = Math.sqrt(squares / vector.length + epsilon);
  return vector.map((value, index) => (value / rootMeanSquare) * gain.data[index]);
}

function row(matrix, index) {
  const columns = matrix.shape[1];
  return matrix.data.subarray(index * columns, (index + 1) * columns);
}

// Each position mixes in the values of itself and the positions before
// it, weighted by how well its query matches their keys.
function attention(tensors, normalized) {
  const queries = normalized.map((vector) => times(vector, tensors.query_weight));
  const keys = normalized.map((vector) => times(vector, tensors.key_weight));
  const values = normalized.map((vector) => times(vector, tensors.value_weight));
  const scale = Math.sqrt(queries[0].length);

  return queries.map((query, position) => {
    const scores = [];
    for (let earlier = 0; earlier <= position; earlier++) {
      let dot = 0;
      for (let index = 0; index < query.length; index++) dot += query[index] * keys[earlier][index];
      scores.push(dot / scale);
    }
    const largest = Math.max(...scores);
    const exponentials = scores.map((score) => Math.exp(score - largest));
    const total = exponentials.reduce((sum, value) => sum + value, 0);

    const attended = new Float64Array(query.length);
    exponentials.forEach((weight, earlier) => {
      for (let index = 0; index < attended.length; index++) {
        attended[index] += (weight / total) * values[earlier][index];
      }
    });
    return times(attended, tensors.output_weight);
  });
}

// Next word scores and neuron values for every position of `ids`, which
// starts with the start token. `push`, if given, is added to the neurons
// at every position before the rest of the model sees them: steering.
//
// Returns flat Float32Arrays: `logits` is length × vocabulary, and
// `hidden` is length × 128.
export function forward({ manifest, tensors }, ids, push = null) {
  const epsilon = manifest.rmsnorm_epsilon;
  const vocabularySize = tensors.projection.shape[1];
  const hiddenWidth = tensors.weight1.shape[1];

  const embedded = ids.map((id, position) =>
    plus(Float64Array.from(row(tensors.embeddings, id)), row(tensors.positions, position)),
  );
  const mixed = attention(tensors, embedded.map((vector) => rmsnorm(vector, tensors.gain1, epsilon)));
  const residual = embedded.map((vector, position) => plus(vector, mixed[position]));

  const logits = new Float32Array(ids.length * vocabularySize);
  const hidden = new Float32Array(ids.length * hiddenWidth);

  residual.forEach((vector, position) => {
    let neurons = plus(times(rmsnorm(vector, tensors.gain2, epsilon), tensors.weight1), tensors.bias1.data)
      .map((value) => Math.max(value, 0));
    if (push) neurons = plus(neurons, push);
    hidden.set(neurons, position * hiddenWidth);

    const output = plus(plus(vector, times(neurons, tensors.weight2)), tensors.bias2.data);
    logits.set(times(rmsnorm(output, tensors.gain3, epsilon), tensors.projection), position * vocabularySize);
  });

  return { logits, hidden };
}
