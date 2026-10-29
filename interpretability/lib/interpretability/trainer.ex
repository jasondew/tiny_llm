defmodule Interpretability.Trainer do
  @moduledoc """
  One training loop, shared by the llama model and the sparse autoencoder.

  Training is the same three moves, over and over:

  1. Measure how wrong the model is on a batch: the *loss*.
  2. Ask, for every parameter, which direction would make the loss
     smaller: the *gradient*. The first talk derived every gradient by
     hand. Here `Nx.Defn` derives them automatically from the loss
     function, which is called *autodiff*.
  3. Nudge every parameter a little that way. Adam, the optimizer used
     here, scales each nudge by how noisy that parameter's gradient has
     been, which makes training far less sensitive to the learning rate
     than the plain gradient descent of the first talk.

  The loop knows nothing about language or autoencoders. A model is just
  a map of tensors and a loss function over a batch.
  """

  alias Interpretability.Compiler

  @typedoc "Named parameter tensors."
  @type params :: %{atom() => Nx.Tensor.t()}

  @typedoc "One logged point of the loss curve."
  @type history_point :: %{step: pos_integer(), loss: float()}

  @doc """
  Trains `params` for `steps` batches taken from `batches`.

  Options:

  - `:steps` (required): how many batches to train on.
  - `:learning_rate` (required): Adam's step size.
  - `:weight_decay`: how strongly to shrink every parameter toward zero
    each step (AdamW). Defaults to 0, plain Adam.
  - `:log_every`: how often to record the loss. Defaults to 100.
  - `:after_step`: a function applied to the parameters after every
    update, for constraints such as unit length decoder rows. Defaults to
    leaving them alone.

  Returns the trained parameters and the logged loss history.
  """
  @spec run(params(), Enumerable.t(), (params(), map() -> Nx.Tensor.t()), keyword()) ::
          {params(), [history_point()]}
  def run(params, batches, loss_function, options) do
    steps = Keyword.fetch!(options, :steps)
    log_every = Keyword.get(options, :log_every, 100)
    after_step = Keyword.get(options, :after_step, & &1)

    {init_optimizer, update_optimizer} =
      optimizer(Keyword.fetch!(options, :learning_rate), Keyword.get(options, :weight_decay, 0.0))

    step =
      Nx.Defn.jit(
        fn params, optimizer_state, batch ->
          {loss, gradients} = Nx.Defn.value_and_grad(params, &loss_function.(&1, batch))
          {updates, optimizer_state} = update_optimizer.(gradients, optimizer_state, params)
          params = after_step.(Polaris.Updates.apply_updates(params, updates))
          {params, optimizer_state, loss}
        end,
        Compiler.options()
      )

    {params, _optimizer_state, history} =
      batches
      |> Stream.take(steps)
      |> Stream.with_index(1)
      |> Enum.reduce({params, init_optimizer.(params), []}, fn {batch, index},
                                                               {params, optimizer_state, history} ->
        {params, optimizer_state, loss} = step.(params, optimizer_state, batch)

        history =
          if rem(index, log_every) == 0,
            do: [%{step: index, loss: Nx.to_number(loss)} | history],
            else: history

        {params, optimizer_state, history}
      end)

    {params, Enum.reverse(history)}
  end

  ## PRIVATE FUNCTIONS

  defp optimizer(learning_rate, weight_decay) when weight_decay == 0,
    do: Polaris.Optimizers.adam(learning_rate: learning_rate)

  defp optimizer(learning_rate, weight_decay),
    do: Polaris.Optimizers.adamw(learning_rate: learning_rate, decay: weight_decay)
end
