defmodule TinyLlm.BigramTest do
  @moduledoc """
  Stage 2 acceptance tests, written from the criteria in docs/build-brief.md.

  These are a specification, not a description of existing code.

  The brief fixes the counts shape but not the rest, so these tests assume:

    * each sentence is counted with `Vocab.start_token/0` prepended, so a
      sentence's first word is an observed successor of `"<start>"`.
    * `matrix/1` leaves a row of zeros for a token never seen as a
      predecessor, rather than smoothing it to uniform. `"."` is always
      such a token, since nothing follows the end of a sentence.
    * `sentence/1` takes the normalized matrix, not the raw counts, and
      returns words without the leading start token but with the
      trailing period.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Bigram
  alias TinyLlm.Grammar
  alias TinyLlm.Test.Vocabulary
  alias TinyLlm.Vocab

  defp id(word), do: Vocab.word_to_id(word)

  defp trained_matrix do
    Grammar.seed(20_250_807)
    Grammar.corpus(2_000) |> Bigram.counts() |> Bigram.matrix()
  end

  describe "counts/1" do
    test "counts each adjacent pair in a sentence exactly once" do
      counts = Bigram.counts([~w(the llama flees .)])

      assert counts == %{
               {id(Vocab.start_token()), id("the")} => 1,
               {id("the"), id("llama")} => 1,
               {id("llama"), id("flees")} => 1,
               {id("flees"), id(".")} => 1
             }
    end

    test "records the first word as following the start token" do
      counts = Bigram.counts([~w(a dog flees .)])

      assert counts[{id(Vocab.start_token()), id("a")}] == 1
    end

    test "never records anything as following a period" do
      counts = Bigram.counts([~w(the llama flees .), ~w(the dog flees .)])

      following_period =
        Enum.filter(counts, fn {{previous, _next}, _count} -> previous == id(".") end)

      assert following_period == []
    end

    test "accumulates counts across sentences" do
      counts = Bigram.counts([~w(the llama flees .), ~w(the dog flees .)])

      assert counts[{id(Vocab.start_token()), id("the")}] == 2
      assert counts[{id("flees"), id(".")}] == 2
      assert counts[{id("the"), id("llama")}] == 1
      assert counts[{id("the"), id("dog")}] == 1
    end

    test "counts nothing for an empty corpus" do
      assert Bigram.counts([]) == %{}
    end
  end

  describe "matrix/1" do
    test "is 32 by 32" do
      matrix = trained_matrix()

      assert Enum.count(matrix) == 32

      for row <- matrix do
        assert Enum.count(row) == 32
      end
    end

    test "normalizes an observed row to sum to 1" do
      matrix = [~w(the llama flees .), ~w(the dog flees .)] |> Bigram.counts() |> Bigram.matrix()
      row = Enum.at(matrix, id("the"))

      assert_in_delta Enum.at(row, id("llama")), 0.5, 1.0e-12
      assert_in_delta Enum.at(row, id("dog")), 0.5, 1.0e-12
      assert_in_delta Enum.sum(row), 1.0, 1.0e-12
    end

    test "leaves every row either empty or summing to 1" do
      for row <- trained_matrix() do
        sum = Enum.sum(row)

        assert sum == 0.0 or abs(sum - 1.0) < 1.0e-9
      end
    end

    test "leaves the row of an unobserved token all zeros" do
      matrix = [~w(the llama flees .)] |> Bigram.counts() |> Bigram.matrix()

      assert Enum.at(matrix, id("goose")) == List.duplicate(0.0, 32)
    end

    test "leaves the period with an empty row, since nothing follows an end" do
      assert Enum.at(trained_matrix(), id(".")) == List.duplicate(0.0, 32)
    end

    test "leaves the start token with an empty column, since nothing precedes a start" do
      column = Enum.map(trained_matrix(), &Enum.at(&1, id(Vocab.start_token())))

      assert column == List.duplicate(0.0, 32)
    end

    test "leaves the period as its only empty row" do
      empty_rows = Enum.count(trained_matrix(), fn row -> Enum.sum(row) == 0.0 end)

      assert empty_rows == 1
    end

    test "never assigns a negative probability" do
      for row <- trained_matrix(), probability <- row do
        assert probability >= 0.0
      end
    end
  end

  describe "sentence/1" do
    setup do
      # Build the matrix first. `trained_matrix/0` seeds for corpus
      # generation, and both seed functions share one process `:rand`
      # state, so seeding for sampling has to come second.
      matrix = trained_matrix()
      Bigram.seed(1234)

      {:ok, matrix: matrix}
    end

    test "ends every sample with a period", %{matrix: matrix} do
      for _draw <- 1..500 do
        assert List.last(Bigram.sentence(matrix)) == "."
      end
    end

    test "draws only vocabulary words", %{matrix: matrix} do
      unknown =
        1..500
        |> Enum.flat_map(fn _draw -> Bigram.sentence(matrix) end)
        |> Enum.reject(&(&1 in Vocabulary.words()))
        |> Enum.uniq()

      assert unknown == []
    end

    test "never exceeds the context length", %{matrix: matrix} do
      longest =
        1..500
        |> Enum.map(fn _draw -> length(Bigram.sentence(matrix)) end)
        |> Enum.max()

      assert longest <= Bigram.max_tokens()
    end

    test "does not emit the start token it began from", %{matrix: matrix} do
      for _draw <- 1..500 do
        refute Vocab.start_token() in Bigram.sentence(matrix)
      end
    end

    test "draws the same sample twice from the same seed", %{matrix: matrix} do
      Bigram.seed(99)
      first = Enum.map(1..20, fn _draw -> Bigram.sentence(matrix) end)

      Bigram.seed(99)
      second = Enum.map(1..20, fn _draw -> Bigram.sentence(matrix) end)

      assert first == second
    end
  end
end
