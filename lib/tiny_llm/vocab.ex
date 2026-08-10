defmodule TinyLlm.Vocab do
  @moduledoc """
  The model's entire universe of words: 32 of them, in a fixed order.
  """

  @typedoc "One of the 32 words the model may ever see."
  @type word :: String.t()

  @typedoc "A word's position in the fixed ordering, which is also its token."
  @type id :: 0..31

  @typedoc "Whether a word is in its singular or plural form."
  @type grammatical_number :: :singular | :plural

  # Every inflecting class is stored as its singular and plural forms, then
  # interleaved back into vocabulary order. The interleaving is what fixes
  # the ids, so these lists must stay parallel and equal in length.
  @determiners ~w(the a)

  @singular_nouns ~w(llama dog goose fox mouse)
  @plural_nouns ~w(llamas dogs geese foxes mice)
  @nouns Enum.flat_map(Enum.zip(@singular_nouns, @plural_nouns), &Tuple.to_list/1)

  @singular_transitive_verbs ~w(sees chases ignores)
  @plural_transitive_verbs ~w(see chase ignore)
  @transitive_verbs Enum.flat_map(
                      Enum.zip(@singular_transitive_verbs, @plural_transitive_verbs),
                      &Tuple.to_list/1
                    )

  @singular_intransitive_verbs ~w(flees)
  @plural_intransitive_verbs ~w(flee)
  @intransitive_verbs Enum.flat_map(
                        Enum.zip(@singular_intransitive_verbs, @plural_intransitive_verbs),
                        &Tuple.to_list/1
                      )

  @verbs @transitive_verbs ++ @intransitive_verbs

  @singular_copula ~w(is)
  @plural_copula ~w(are)
  @copula @singular_copula ++ @plural_copula

  @adjectives ~w(big small hungry grumpy fast sleepy)
  @connectives ~w(and who)

  # Boundary markers rather than words. Nothing precedes the start and
  # nothing follows the end, so on a transition heatmap the start has an
  # empty column and the end has an empty row.
  @start_token "<start>"
  @end_token "."
  @boundaries [@start_token, @end_token]

  @words Enum.concat([
           @determiners,
           @nouns,
           @verbs,
           @copula,
           @adjectives,
           @connectives,
           @boundaries
         ])

  @spec determiners() :: [word()]
  def determiners, do: @determiners

  @spec nouns() :: [word()]
  def nouns, do: @nouns

  @doc """
  Nouns in one number only, for building a noun phrase that agrees.
  """
  @spec nouns(grammatical_number()) :: [word()]
  def nouns(:singular), do: @singular_nouns
  def nouns(:plural), do: @plural_nouns

  @doc """
  Verbs that always take a direct object, so never end a clause.
  """
  @spec transitive_verbs() :: [word()]
  def transitive_verbs, do: @transitive_verbs

  @spec transitive_verbs(grammatical_number()) :: [word()]
  def transitive_verbs(:singular), do: @singular_transitive_verbs
  def transitive_verbs(:plural), do: @plural_transitive_verbs

  @doc """
  Verbs that never take a direct object, so always end a clause.
  """
  @spec intransitive_verbs() :: [word()]
  def intransitive_verbs, do: @intransitive_verbs

  @spec intransitive_verbs(grammatical_number()) :: [word()]
  def intransitive_verbs(:singular), do: @singular_intransitive_verbs
  def intransitive_verbs(:plural), do: @plural_intransitive_verbs

  @spec copula() :: [word()]
  def copula, do: @copula

  @spec copula(grammatical_number()) :: [word()]
  def copula(:singular), do: @singular_copula
  def copula(:plural), do: @plural_copula

  @spec adjectives() :: [word()]
  def adjectives, do: @adjectives

  @spec connectives() :: [word()]
  def connectives, do: @connectives

  @doc """
  The marker every sequence begins with. Never emitted by the grammar.
  """
  @spec start_token() :: word()
  def start_token, do: @start_token

  @doc """
  The marker every sentence ends with, which is also an ordinary period.
  """
  @spec end_token() :: word()
  def end_token, do: @end_token

  @doc """
  Every word in the vocabulary, in id order.
  """
  @spec words() :: [word()]
  def words, do: @words

  @doc """
  How many words the vocabulary holds. Always 32.
  """
  @spec size() :: pos_integer()
  def size, do: Enum.count(@words)

  @doc """
  The id of a word. Raises for a word outside the vocabulary.
  """
  @spec word_to_id(word()) :: id()
  def word_to_id(word),
    do: Enum.find_index(@words, &(&1 == word)) || raise("Unknown word: #{word}")

  @doc """
  The word an id stands for. Raises for an id outside 0..31.
  """
  @spec id_to_word(id()) :: word()
  def id_to_word(id), do: Enum.at(@words, id) || raise("Unknown id: #{id}")

  @doc """
  A list of words as a list of ids.
  """
  @spec encode([word()]) :: [id()]
  def encode(words), do: Enum.map(words, &word_to_id/1)

  @doc """
  A list of ids back as a list of words.
  """
  @spec decode([id()]) :: [word()]
  def decode(ids), do: Enum.map(ids, &id_to_word/1)
end
