// The sparse autoencoder's encoder, as `Interpretability.SAE.encode`:
//
//     features = keep_top_k(relu((neurons − decoder_bias) · encoder + encoder_bias))
//
// "Keep the top k" keeps every feature at least as large as the k-th
// largest, so a tie at the cutoff keeps one extra, exactly as Nx does.

export function encode(sae, neurons, activeCount) {
  const [inputs, width] = sae.encoder.shape;
  const features = new Float32Array(width);

  for (let feature = 0; feature < width; feature++) {
    let sum = sae.encoder_bias.data[feature];
    for (let input = 0; input < inputs; input++) {
      sum += (neurons[input] - sae.decoder_bias.data[input]) * sae.encoder.data[input * width + feature];
    }
    features[feature] = Math.max(sum, 0);
  }

  const cutoff = Array.from(features).sort((first, second) => second - first)[activeCount - 1];
  return features.map((value) => (value >= cutoff ? value : 0));
}
