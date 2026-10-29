defmodule Interpretability.Vocab do
  @moduledoc """
  The extended model's whole universe of words: 70 of them, in a fixed order.

  The first talk's vocabulary had 32 words. This one grew because the
  grammar grew, not for its own sake. Each new group of words exists to
  carry a new concept the model has to track:

  - **animacy**: llamas chase things, rocks do not. Inanimate nouns get
    their own verb (`falls`) and their own relative pronoun (`that`
    instead of `who`).
  - **tense**: past tense verbs (`chased`) drop the singular and plural
    distinction that present tense verbs (`chases`, `chase`) carry, except
    the copula (`was`, `were`).
  - **pronouns**: `he`, `she`, `it` and `they` carry number and animacy
    in a single word.
  - **prepositions**: `near` and `behind` add a second kind of distractor
    noun between a subject and its verb.

  More concepts than the model has room for is the pressure that can push
  it into superposition. One word is still one token, with no tokenizer, so
  any feature we find fires on whole words a person can read.
  """

  @typedoc "One of the 70 words the model may ever see."
  @type word :: String.t()

  @typedoc "A word's position in the fixed ordering, which is also its token."
  @type id :: 0..69

  @type grammatical_number :: :singular | :plural
  @type animacy :: :animate | :inanimate
  @type tense :: :present | :past

  # Inflecting classes are stored as parallel singular and plural lists,
  # then interleaved into vocabulary order, so the pairs sit side by side.
  @determiners ~w(the a)

  @singular_animate_nouns ~w(llama dog goose fox mouse cat owl wolf)
  @plural_animate_nouns ~w(llamas dogs geese foxes mice cats owls wolves)
  @singular_inanimate_nouns ~w(rock tree river hill)
  @plural_inanimate_nouns ~w(rocks trees rivers hills)

  @pronouns ~w(he she it they)

  @singular_transitive_verbs ~w(sees chases ignores follows)
  @plural_transitive_verbs ~w(see chase ignore follow)
  @past_transitive_verbs ~w(saw chased ignored followed)

  @singular_animate_intransitive_verbs ~w(flees sleeps)
  @plural_animate_intransitive_verbs ~w(flee sleep)
  @past_animate_intransitive_verbs ~w(fled slept)

  @singular_inanimate_intransitive_verbs ~w(falls)
  @plural_inanimate_intransitive_verbs ~w(fall)
  @past_inanimate_intransitive_verbs ~w(fell)

  @adjectives ~w(big small hungry grumpy fast sleepy old wet)
  @relative_pronouns ~w(who that)
  @conjunctions ~w(and)
  @prepositions ~w(near behind)

  @start_token "<start>"
  @end_token "."

  @words Enum.concat([
           @determiners,
           Enum.flat_map(
             Enum.zip(@singular_animate_nouns, @plural_animate_nouns),
             &Tuple.to_list/1
           ),
           Enum.flat_map(
             Enum.zip(@singular_inanimate_nouns, @plural_inanimate_nouns),
             &Tuple.to_list/1
           ),
           @pronouns,
           Enum.flat_map(
             Enum.zip(@singular_transitive_verbs, @plural_transitive_verbs),
             &Tuple.to_list/1
           ),
           @past_transitive_verbs,
           Enum.flat_map(
             Enum.zip(@singular_animate_intransitive_verbs, @plural_animate_intransitive_verbs),
             &Tuple.to_list/1
           ),
           @past_animate_intransitive_verbs,
           @singular_inanimate_intransitive_verbs,
           @plural_inanimate_intransitive_verbs,
           @past_inanimate_intransitive_verbs,
           ~w(is are was were),
           @adjectives,
           @relative_pronouns,
           @conjunctions,
           @prepositions,
           [@start_token, @end_token]
         ])

  @spec determiners() :: [word()]
  def determiners, do: @determiners

  @doc """
  Nouns of one animacy and one number, for building a noun phrase.
  """
  @spec nouns(animacy(), grammatical_number()) :: [word()]
  def nouns(:animate, :singular), do: @singular_animate_nouns
  def nouns(:animate, :plural), do: @plural_animate_nouns
  def nouns(:inanimate, :singular), do: @singular_inanimate_nouns
  def nouns(:inanimate, :plural), do: @plural_inanimate_nouns

  @spec pronouns() :: [word()]
  def pronouns, do: @pronouns

  @doc """
  Verbs that always take an object. Past tense forms ignore number.
  """
  @spec transitive_verbs(tense(), grammatical_number()) :: [word()]
  def transitive_verbs(:present, :singular), do: @singular_transitive_verbs
  def transitive_verbs(:present, :plural), do: @plural_transitive_verbs
  def transitive_verbs(:past, _number), do: @past_transitive_verbs

  @doc """
  Verbs that never take an object. Animals flee and sleep; rocks fall.
  """
  @spec intransitive_verbs(animacy(), tense(), grammatical_number()) :: [word()]
  def intransitive_verbs(:animate, :present, :singular), do: @singular_animate_intransitive_verbs
  def intransitive_verbs(:animate, :present, :plural), do: @plural_animate_intransitive_verbs
  def intransitive_verbs(:animate, :past, _number), do: @past_animate_intransitive_verbs

  def intransitive_verbs(:inanimate, :present, :singular),
    do: @singular_inanimate_intransitive_verbs

  def intransitive_verbs(:inanimate, :present, :plural), do: @plural_inanimate_intransitive_verbs
  def intransitive_verbs(:inanimate, :past, _number), do: @past_inanimate_intransitive_verbs

  @doc """
  The one verb that keeps number in the past tense: `was` and `were`.
  """
  @spec copula(tense(), grammatical_number()) :: [word()]
  def copula(:present, :singular), do: ["is"]
  def copula(:present, :plural), do: ["are"]
  def copula(:past, :singular), do: ["was"]
  def copula(:past, :plural), do: ["were"]

  @spec adjectives() :: [word()]
  def adjectives, do: @adjectives

  @doc """
  `who` introduces a relative clause on an animate noun, `that` on an
  inanimate one.
  """
  @spec relative_pronouns() :: [word()]
  def relative_pronouns, do: @relative_pronouns

  @spec prepositions() :: [word()]
  def prepositions, do: @prepositions

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
  How many words the vocabulary holds. Always 70.
  """
  @spec size() :: pos_integer()
  def size, do: length(@words)

  @doc """
  The id of a word. Raises for a word outside the vocabulary.
  """
  @spec word_to_id(word()) :: id()
  def word_to_id(word),
    do: Enum.find_index(@words, &(&1 == word)) || raise("Unknown word: #{word}")

  @doc """
  The word an id stands for. Raises for an id outside 0..69.
  """
  @spec id_to_word(id()) :: word()
  def id_to_word(id) when id in 0..69, do: Enum.at(@words, id)
  def id_to_word(id), do: raise("Unknown id: #{id}")

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
