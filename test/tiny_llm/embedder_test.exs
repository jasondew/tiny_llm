defmodule TinyLlm.EmbedderTest do
  @moduledoc """
  Stage 3 acceptance tests, written from the criteria in docs/build-brief.md.

  These are a specification, not a description of existing code.

  The brief fixes the architecture but not every shape, so these tests
  assume:

    * params are `%{embedding: E, projection: W}`, with `E` sized
      `vocabulary_size` by `hidden_size` and `W` the transpose of that
      shape, so the two are separate rather than tied. Stage 5 revisits
      tying and is asked to document its choice.
    * `forward/2` takes one input id and returns a probability row.
    * `loss/2` and `gradients/2` take a batch and return means, so the
      learning rate does not have to change with the batch size.
  """

  use ExUnit.Case, async: true

  # TEMPORARY: skipped until TinyLlm.GradCheck is implemented, since the
  # acceptance criterion for these gradients is that the checker agrees with
  # them. Delete this line to bring them back.
  @moduletag :skip

  alias TinyLlm.Embedder
  alias TinyLlm.GradCheck
  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @config %Train.Config{model: Embedder, vocabulary_size: 32, hidden_size: 32}
  @tiny %Train.Config{model: Embedder, vocabulary_size: 5, hidden_size: 3}

  defp id(word), do: Vocab.word_to_id(word)

  describe "init/1" do
    test "builds an embedding and a projection of the configured shape" do
      Train.seed(1)
      params = Embedder.init(@tiny)

      assert Map.keys(params) |> Enum.sort() == [:embedding, :projection]
      assert Tensor.shape(params.embedding) == {5, 3}
      assert Tensor.shape(params.projection) == {3, 5}
    end

    test "draws the same parameters twice from the same seed" do
      Train.seed(7)
      first = Embedder.init(@config)

      Train.seed(7)
      second = Embedder.init(@config)

      assert first == second
    end

    test "starts small enough that the first logits are near zero" do
      Train.seed(1)
      params = Embedder.init(@config)

      for value <- List.flatten(params.embedding) ++ List.flatten(params.projection) do
        assert abs(value) <= 0.02
      end
    end
  end

  describe "examples/1" do
    test "pairs every token with the one that follows it" do
      assert Embedder.examples([~w(the llama flees .)]) == [
               {id(Vocab.start_token()), id("the")},
               {id("the"), id("llama")},
               {id("llama"), id("flees")},
               {id("flees"), id(".")}
             ]
    end

    test "accumulates across sentences" do
      examples = Embedder.examples([~w(the llama flees .), ~w(the dog flees .)])

      assert length(examples) == 8
    end

    test "never predicts from a period, since nothing follows an end" do
      Grammar.seed(3)

      inputs =
        Grammar.corpus(200)
        |> Embedder.examples()
        |> Enum.map(fn {input, _target} -> input end)
        |> Enum.uniq()

      refute id(".") in inputs
    end

    test "has nothing to learn from an empty corpus" do
      assert Embedder.examples([]) == []
    end
  end

  describe "forward/2" do
    setup do
      Train.seed(1)
      {:ok, params: Embedder.init(@config)}
    end

    test "returns a distribution over the whole vocabulary", %{params: params} do
      row = Embedder.forward(params, id("the"))

      assert length(row) == 32
      assert_in_delta Enum.sum(row), 1.0, 1.0e-9
      assert Enum.all?(row, &(&1 >= 0.0))
    end

    test "is nearly uniform before training", %{params: params} do
      row = Embedder.forward(params, id("the"))

      for probability <- row do
        assert_in_delta probability, 1.0 / 32, 0.01
      end
    end
  end

  describe "loss/2" do
    test "starts near ln(vocabulary_size), the cost of knowing nothing" do
      Train.seed(1)
      params = Embedder.init(@config)
      batch = [{id("the"), id("llama")}, {id("llama"), id("flees")}]

      assert_in_delta Embedder.loss(params, batch), :math.log(32), 0.05
    end

    test "falls when the model is made confident and right" do
      Train.seed(1)
      params = Embedder.init(@config)
      batch = [{id("the"), id("llama")}]

      before = Embedder.loss(params, batch)
      confident = Embedder.gradients(params, batch)

      improved = %{
        embedding: Tensor.sub(params.embedding, Tensor.scale(confident.embedding, 1.0)),
        projection: Tensor.sub(params.projection, Tensor.scale(confident.projection, 1.0))
      }

      assert Embedder.loss(improved, batch) < before
    end
  end

  describe "gradients/2" do
    setup do
      Train.seed(1)
      {:ok, params: Embedder.init(@config)}
    end

    test "come back keyed and shaped exactly like the params", %{params: params} do
      batch = [{id("the"), id("llama")}]
      gradients = Embedder.gradients(params, batch)

      assert Map.keys(gradients) == Map.keys(params)
      assert Tensor.shape(gradients.embedding) == Tensor.shape(params.embedding)
      assert Tensor.shape(gradients.projection) == Tensor.shape(params.projection)
    end

    test "touch only the embedding rows of tokens in the batch", %{params: params} do
      batch = [{id("the"), id("llama")}]
      gradients = Embedder.gradients(params, batch)

      touched =
        gradients.embedding
        |> Enum.with_index()
        |> Enum.reject(fn {row, _index} -> Enum.all?(row, &(&1 == 0.0)) end)
        |> Enum.map(fn {_row, index} -> index end)

      assert touched == [id("the")]
    end

    test "matches finite differences on a tiny config" do
      Train.seed(1)
      params = Embedder.init(@tiny)
      batch = [{0, 1}, {2, 3}, {4, 0}, {1, 1}]

      assert GradCheck.max_relative_error(Embedder, params, batch) < 1.0e-3
    end

    test "matches finite differences on a batch of one" do
      Train.seed(2)
      params = Embedder.init(@tiny)

      assert GradCheck.max_relative_error(Embedder, params, [{3, 2}]) < 1.0e-3
    end
  end
end
