defmodule Interpretability.MixProject do
  use Mix.Project

  def project do
    [
      app: :interpretability,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Nx and Polaris are the math. EXLA only compiles it faster for training
  # runs, so it stays out of :prod, which is how a path dependency (the
  # deck) sees this project.
  defp deps do
    [
      {:nx, "~> 1.0"},
      {:polaris, "~> 0.2.0"},
      {:exla, "~> 1.0", only: [:dev, :test]}
    ]
  end
end
