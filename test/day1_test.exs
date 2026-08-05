defmodule Day1Test do
  @moduledoc """
  Stage 1 acceptance tests, written from the criteria in docs/build-brief.md.

  These are a specification, not a description of existing code. They are
  expected to fail until Vocab, Grammar, and Tensor are written.

  The brief fixes the function names but not every shape. Where it was
  silent, these tests assume:

    * `Grammar.sentence/0` returns a list of word strings, not token ids.
    * `Tensor.one_hot/2` takes `(index, size)` and returns one row, so a
      plain list of floats rather than a 1-row matrix.
    * `Tensor.dot/2` takes two rows and returns a float.
    * `Tensor.argmax/1` takes one row and returns an integer index.
    * `Tensor.random/3` takes `(rows, cols, scale)` and draws uniformly
      from the closed interval `-scale..scale`.

  Any of those can be flipped; they are assumptions, not requirements.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Test.Structure
  alias TinyLlm.Vocab

  @vocabulary ~w(
    the a
    cat cats dog dogs bird birds fox foxes mouse mice
    sees see chases chase eats eat sleeps sleep
    is are
    big small hungry happy fast old
    and then who
    .
  )

  describe "Vocab" do
    test "holds exactly 32 words" do
      assert Vocab.size() == 32
    end

    test "lists the 32 words in the order the brief fixes" do
      assert Vocab.words() == @vocabulary
    end

    test "assigns every word a distinct id covering 0 through 31" do
      ids = Enum.map(@vocabulary, &Vocab.word_to_id/1)

      assert Enum.sort(ids) == Enum.to_list(0..31)
    end

    test "round-trips every word through its id" do
      for word <- @vocabulary do
        assert word |> Vocab.word_to_id() |> Vocab.id_to_word() == word
      end
    end

    test "anchors the first and last ids to the fixed ordering" do
      assert Vocab.word_to_id("the") == 0
      assert Vocab.word_to_id(".") == 31
    end

    test "round-trips a sentence through encode and decode" do
      sentence = ~w(the big cat sees a mouse .)

      assert sentence |> Vocab.encode() |> Vocab.decode() == sentence
    end
  end

  describe "Grammar" do
    setup do
      Grammar.seed(1234)
      :ok
    end

    test "draws the same corpus twice from the same seed" do
      Grammar.seed(99)
      first = Grammar.corpus(50)

      Grammar.seed(99)
      second = Grammar.corpus(50)

      assert first == second
    end

    test "builds every sentence out of vocabulary words" do
      unknown =
        10_000
        |> Grammar.corpus()
        |> Enum.flat_map(fn sentence -> Enum.reject(sentence, &(&1 in @vocabulary)) end)
        |> Enum.uniq()

      assert unknown == []
    end

    test "ends every sentence with a period" do
      refute Enum.any?(Grammar.corpus(10_000), &(List.last(&1) != "."))
    end

    test "keeps every sentence within the 16 token context length" do
      longest = 10_000 |> Grammar.corpus() |> Enum.map(&length/1) |> Enum.max()

      assert longest <= 16
    end

    test "agrees subject and verb in number in simple clauses" do
      disagreeing =
        10_000
        |> Grammar.corpus()
        |> Enum.filter(&Structure.simple?/1)
        |> Enum.reject(&Structure.agreement_ok?/1)

      assert disagreeing == []
    end

    test "never places the determiner a before a plural noun" do
      offending =
        10_000
        |> Grammar.corpus()
        |> Enum.reject(&Structure.determiners_ok?/1)

      assert offending == []
    end

    test "produces relative clauses and compound subjects" do
      words = 10_000 |> Grammar.corpus() |> List.flatten() |> MapSet.new()

      assert "who" in words
      assert "and" in words
    end

    test "generates mostly distinct sentences rather than memorizable few" do
      corpus = Grammar.corpus(5_000)
      unique_fraction = corpus |> Enum.uniq() |> length() |> Kernel./(5_000)

      assert unique_fraction > 0.6
    end
  end

  describe "Tensor" do
    test "builds a zero matrix of the requested shape" do
      assert Tensor.zeros(2, 3) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    end

    test "reports the shape as rows and columns" do
      assert Tensor.shape([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == {2, 3}
    end

    test "draws random entries within plus or minus the scale" do
      entries = 8 |> Tensor.random(8, 0.02) |> List.flatten()

      assert length(entries) == 64
      assert Enum.all?(entries, &(&1 >= -0.02 and &1 <= 0.02))
    end

    test "builds a one hot row" do
      assert Tensor.one_hot(2, 5) == [0.0, 0.0, 1.0, 0.0, 0.0]
    end

    test "transposes rows into columns" do
      assert Tensor.transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) ==
               [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    end

    test "adds matrices entry by entry" do
      assert Tensor.add([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]) ==
               [[11.0, 22.0], [33.0, 44.0]]
    end

    test "subtracts matrices entry by entry" do
      assert Tensor.sub([[10.0, 20.0], [30.0, 40.0]], [[1.0, 2.0], [3.0, 4.0]]) ==
               [[9.0, 18.0], [27.0, 36.0]]
    end

    test "multiplies matrices entry by entry" do
      assert Tensor.hadamard([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]) ==
               [[5.0, 12.0], [21.0, 32.0]]
    end

    test "scales every entry by a constant" do
      assert Tensor.scale([[1.0, 2.0], [3.0, 4.0]], 3.0) == [[3.0, 6.0], [9.0, 12.0]]
    end

    test "maps a function over every entry" do
      assert Tensor.map([[1.0, -2.0], [-3.0, 4.0]], &abs/1) == [[1.0, 2.0], [3.0, 4.0]]
    end

    test "takes the dot product of two rows" do
      assert Tensor.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    end

    test "multiplies a 2x3 by a 3x2 into a 2x2" do
      left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      right = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

      assert Tensor.matmul(left, right) == [[58.0, 64.0], [139.0, 154.0]]
    end

    test "multiplies non-square shapes in the other order" do
      left = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
      right = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

      assert Tensor.matmul(left, right) ==
               [[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]
    end

    test "softmaxes each row to sum to one" do
      for row <- Tensor.softmax([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]) do
        assert_in_delta Enum.sum(row), 1.0, 1.0e-12
      end
    end

    test "softmaxes a uniform row into a uniform distribution" do
      assert Tensor.softmax([[0.0, 0.0, 0.0, 0.0]]) == [[0.25, 0.25, 0.25, 0.25]]
    end

    test "softmaxes ordering-preservingly" do
      [row] = Tensor.softmax([[1.0, 3.0, 2.0]])

      assert Enum.at(row, 1) > Enum.at(row, 2)
      assert Enum.at(row, 2) > Enum.at(row, 0)
    end

    test "survives logits of 1000 without overflowing" do
      [row] = Tensor.softmax([[1000.0, 1000.0, 1000.0]])

      assert Enum.all?(row, &is_float/1)
      assert_in_delta Enum.sum(row), 1.0, 1.0e-12

      for probability <- row do
        assert_in_delta probability, 1.0 / 3.0, 1.0e-12
      end
    end

    test "collapses to near certainty on a dominant logit" do
      [row] = Tensor.softmax([[1000.0, 0.0]])

      assert_in_delta Enum.at(row, 0), 1.0, 1.0e-12
      assert_in_delta Enum.at(row, 1), 0.0, 1.0e-12
    end

    test "finds the index of the largest entry in a row" do
      assert Tensor.argmax([0.1, 0.7, 0.2]) == 1
    end

    test "breaks argmax ties toward the first index" do
      assert Tensor.argmax([0.5, 0.5, 0.1]) == 0
    end
  end
end
