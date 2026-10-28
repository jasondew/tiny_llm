# Builder decisions

Each decision the spec left to the builder, with the reason.

## Nx.Defn and Polaris instead of Axon

Steering and SAE splicing both need to replace the MLP's hidden
activations partway through the forward pass. With a hand written
`Nx.Defn` forward pass that is one function argument. In Axon it is a
hook on a layer inside a compiled graph. Polaris supplies the Adam
optimizer that Axon would otherwise have provided.
