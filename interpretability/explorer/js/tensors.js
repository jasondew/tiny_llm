// Turning the exported weights back into named tensors.
//
// `mix interp.export` writes every tensor's raw 32 bit floats into one
// binary file, and a manifest saying, for each name, its shape and where
// it starts (counted in floats). A tensor here is just `{shape, data}`,
// with `data` a flat Float32Array laid out row by row, the same order Nx
// uses.

export function loadTensors(manifest, arrayBuffer) {
  const floats = new Float32Array(arrayBuffer);
  const tensors = {};
  for (const { name, shape, offset } of manifest.tensors) {
    const size = shape.reduce((product, dimension) => product * dimension, 1);
    tensors[name] = { shape, data: floats.subarray(offset, offset + size) };
  }
  return tensors;
}
