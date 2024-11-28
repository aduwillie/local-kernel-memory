#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

using feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.TextGeneration;
using SmartComponents.LocalEmbeddings;
using System.Reflection;
using System.Runtime.CompilerServices;

var skConfig = new Microsoft.KernelMemory.SemanticKernel.SemanticKernelConfig();

var memory = new KernelMemoryBuilder()
  .WithSemanticKernelTextEmbeddingGenerationService(
    service: new CustomTextEmbeddingService(),
    config: skConfig)
  .WithSemanticKernelTextGenerationService(
    service: new CustomTextGenerationService(),
    config: skConfig)
  .WithDefaultWebScraper()
  .Build();

//var aliceInWonderLandUri = "https://www.gutenberg.org/files/11/11-h/11-h.htm";
//await memory.ImportWebPageAsync(aliceInWonderLandUri);
await memory.ImportTextAsync("""
At the beginning of the book, Alice was jumping. She followed the rabbit because she was hungry.
""");

var question1 = "What was Alice doing with her sister at the beginning of the book?";
Console.WriteLine($"Question: {question1}");
Console.WriteLine($"Answer: {await memory.AskAsync(question1)}");

var question2 = "Why did Alice follow the White Rabbit?";
Console.WriteLine($"Question: {question2}");
Console.WriteLine($"Answer: {await memory.AskAsync(question2)}");

Console.ReadLine();

public class CustomTextEmbeddingService() : ITextEmbeddingGenerationService
{
  private readonly Dictionary<string, object?> _internalAttributes = [];
  private readonly LocalEmbedder embedder = new LocalEmbedder();

  public IReadOnlyDictionary<string, object?> Attributes => _internalAttributes;

  public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(IList<string> data, Kernel? kernel = null, CancellationToken cancellationToken = default)
  {
    IList<ReadOnlyMemory<float>> embeddings = data
      .Select(d => embedder.Embed(d).Values).ToList() ?? new();

    return Task.FromResult(embeddings);
  }
}

class CustomTextGenerationService : ITextGenerationService
{
  private string modelPath;
  private Kernel baseKernel;

  public IReadOnlyDictionary<string, object?> Attributes => new Dictionary<string, object?>();

  public CustomTextGenerationService()
  {
    var path = Path.Combine(
      Directory.GetParent(Assembly.GetExecutingAssembly().Location)!.FullName,
      "Models",
      "Phi-3-medium-4k-instruct-onnx-cpu",
      "cpu-int4-rtn-block-32-acc-level-4");
    if (!Directory.Exists(path))
    {
      throw new DirectoryNotFoundException($"Model file not found at {path}");
    }

    modelPath = path;
    baseKernel = Kernel.CreateBuilder()
      .AddLocalTextEmbeddingGeneration()
      .AddOnnxRuntimeGenAIChatCompletion(
        modelPath: modelPath)
      .Build();

    var answer = baseKernel.InvokePromptAsync("Hello", new KernelArguments
    (
      new OnnxRuntimeGenAIPromptExecutionSettings { MaxLength = 2048 }
    )).Result;
    Console.WriteLine(answer.ToString());
  }

  public async IAsyncEnumerable<StreamingTextContent> GetStreamingTextContentsAsync(string prompt, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
  {
    await foreach(var textContent in RunPromptWithStreaming(prompt, null, cancellationToken))
    {
      yield return textContent;
    }
  }

  public Task<IReadOnlyList<TextContent>> GetTextContentsAsync(string prompt, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
  {
    return Task.FromResult<IReadOnlyList<TextContent>>([]);
  }

  private async Task<string> RunPromptWithoutStreaming(string prompt, KernelArguments? args, CancellationToken cancellationToken)
  {
    var result = await baseKernel.InvokePromptAsync(prompt, args, cancellationToken: cancellationToken);
    return result.ToString();
  }

  private async IAsyncEnumerable<StreamingTextContent> RunPromptWithStreaming(string prompt, KernelArguments? args, [EnumeratorCancellation] CancellationToken cancellationToken)
  {
    var streamResults = baseKernel.InvokePromptStreamingAsync(prompt, args, cancellationToken: cancellationToken);
    await foreach (var content in streamResults)
    {
      var textContent = content.ToString();
      yield return new StreamingTextContent(textContent);
    }
  }
}
