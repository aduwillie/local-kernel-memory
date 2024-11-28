#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using SmartComponents.LocalEmbeddings;

var embedder = new LocalEmbedder();

var cat = embedder.Embed("Cats can be blue.");
var dog = embedder.Embed("Dogs can be red.");
var random = embedder.Embed("This is something very random.");

// Comparing things that are similar
var kitten = embedder.Embed("Kittens!!!");
Console.WriteLine(cat.Similarity(kitten));

// Comparing non-similar things
Console.WriteLine(cat.Similarity(random));
Console.WriteLine(dog.Similarity(random));

// Semantic Kernel setup
var kernel = Kernel.CreateBuilder()
  .AddLocalTextEmbeddingGeneration()
  .Build();

var embeddingsGenerator = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
var inputEmbedding = await embeddingsGenerator
  .GenerateEmbeddingAsync("Williams is a Semantic Kernel developer.");

Console.WriteLine($"Input embedding length: {inputEmbedding.Length}");

var info = new[]
{
  "Williams is married to Abigail",
  "Williams works at Microsoft",
  "Another nickname of Williams is Bill"
};
var infoEmbeddings = info.Select(i => new Info
{
  Text = i,
  Embedding = embedder.Embed(i),
});

var question = "Where does Williams work?";
var questionEmbedding = embedder.Embed(question);

var answers = LocalEmbedder.FindClosest(
  target: questionEmbedding,
  candidates: infoEmbeddings.Select(i => (i, i.Embedding)),
  maxResults: 1);

Console.WriteLine();
Console.WriteLine($"Question: {question}");
Console.WriteLine($"Answer: {answers.FirstOrDefault()?.Text ?? "No answer for you."}");

Console.ReadLine();

class Info
{
  public string Text { get; init; } = string.Empty;
  public EmbeddingF32 Embedding { get; init; }
}