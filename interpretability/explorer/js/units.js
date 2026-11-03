// What the page says about one unit (a neuron or an SAE feature).
//
// A unit's label is its best matching grammar label from the evaluation
// sentences. A unit that never fired there has no label worth showing,
// even though typed sentences can still switch it on, and no peak to
// scale its bars by.

export function describeUnit(unit) {
  if (unit.examples.length === 0) {
    return { label: "No label: it never fired on the evaluation sentences", scores: "", peak: null };
  }
  const [best] = unit.top;
  return {
    label: best.label,
    scores: `precision ${best.precision.toFixed(2)}, recall ${best.recall.toFixed(2)}`,
    peak: unit.examples[0].activation,
  };
}
