defmodule TinyLlmTest do
  use ExUnit.Case
  doctest TinyLlm

  test "greets the world" do
    assert TinyLlm.hello() == :world
  end
end
