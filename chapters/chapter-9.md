# Model Parallelism: Slicing Large Models Across Devices

As models grow past the memory limits of a single GPU or accelerator, “just use a bigger batch” and classic data parallelism stop being enough. Model parallelism tackles this by splitting the model itself across devices so that layers, blocks, or even individual tensors live on different pieces of hardware.

In this chapter, you’ll learn how to reason about these splits, what kinds of parallelism exist in practice (tensor, pipeline, expert/sharded), and how modern libraries hide much of the complexity—while still leaving you responsible for debugging and performance tuning. The goal is not to memorize every framework’s API, but to recognize the common patterns so you can scale a single-node script to multi-GPU, multi-accelerator setups without rewriting your whole codebase. Think of this as a companion to the data-parallel techniques in [chapter 8](../chapters/chapter-8.md) and the profiling workflows from [chapter 3](../chapters/chapter-3.md).

Learning objectives:
- Explain when model parallelism is needed vs when data parallelism is enough.
- Describe common model parallel strategies (layer-wise, tensor, pipeline, expert).
- Understand how model state, activations, and communication patterns affect memory and throughput.
- Identify the main trade-offs between different model parallel approaches in real workloads.

Sections:
- When and why model parallelism is necessary
- Forms of model parallelism: layer, tensor, pipeline, and experts
- Memory anatomy: parameters, activations, optimizer state, and checkpoints
- Communication patterns and performance pitfalls
- Framework support: PyTorch, DeepSpeed, Megatron, and ecosystem tools
- Debugging and monitoring distributed model execution
- Exercises: simple model splits and profiling communication vs compute

## When and why model parallelism is necessary

Data parallelism is the default scaling tool: you copy the model to multiple devices, split the batch, run forward and backward passes independently, then average gradients. As long as the model fits in the memory of a single device, this works well and is often the simplest, fastest option. If you haven’t yet implemented basic data parallelism and input sharding, review the patterns in [chapter 8](../chapters/chapter-8.md) first; they’re usually easier to reason about and debug.

The trouble starts when the *model itself* no longer fits. Large transformer stacks, giant embedding tables, diffusion models, or multi-task systems with many heads can exceed the memory of a single GPU even with a batch size of one. You might first try gradient checkpointing, activation recomputation, or aggressive mixed precision, but at some point you hit a wall: there just isn’t enough device memory to hold all parameters, activations, and optimizer state at once. Techniques like mixed precision from [chapter 7](../chapters/chapter-7.md) and the memory strategies in [chapter 16](../chapters/chapter-16.md) are still worth applying before you commit to a complex parallelism scheme.

Model parallelism becomes necessary once that wall is your main bottleneck rather than throughput. Instead of replicating the entire model on each device, you partition it: different layers, blocks, or tensor shards live on different accelerators. During the forward and backward pass, tensors flow across device boundaries, allowing you to “assemble” a logical model that is much larger than any single GPU.

There are also cases where model parallelism is attractive even before you run out of memory. Certain architectures have natural boundaries (e.g., encoder–decoder splits or expert layers in Mixture-of-Experts models) that map well onto separate devices. Spreading them out can improve utilization, reduce hot spots (like a single giant embedding layer), or let you reuse existing hardware layouts (e.g., one GPU per stage of a pipeline tied to I/O or preprocessing components).

The trade-off is complexity: once the model is spread across devices, you must think about communication costs, load balancing, and failure modes across multiple processes or nodes. In this chapter, we’ll treat model parallelism as a tool you reach for *after* you’ve exhausted simpler options—profiling, better batching, and data parallelism—and we’ll focus on the situations where its added complexity clearly pays off. Later chapters on multi-node setups ([chapter 10](../chapters/chapter-10.md)) and orchestration ([chapter 12](../chapters/chapter-12.md)) build on these ideas when you scale past a single machine.

## Forms of model parallelism: layer, tensor, pipeline, and experts

“Model parallelism” is an umbrella term for several concrete patterns. In practice, most large-scale systems combine more than one. It helps to name them explicitly so you can recognize which one a framework or paper is talking about, and what trade-offs it implies.

At the simplest end is **layer-wise (or block-wise) model parallelism**. Here you divide the model along its depth: early layers on one device, later layers on another. A common toy example is “encoder on GPU 0, decoder on GPU 1,” or splitting a tall transformer into two halves. During the forward pass, activations flow from one device to the next. This is usually the first strategy people try because it maps directly onto how they think about the model, but it can leave devices underutilized if one half does much more work than the other.

**Tensor (or intra-layer) model parallelism** goes a level deeper. Instead of assigning whole layers to a device, you split the heavy tensor operations inside a layer across multiple devices. For example, a giant linear layer’s weight matrix can be sharded by columns across two GPUs; each GPU computes a partial result, and then the results are combined. Libraries like Megatron-LM popularized this style for transformers. Tensor parallelism gives finer-grained control and better load balancing for very large layers, but it introduces more frequent all-reduce or all-gather communications.

**Pipeline parallelism** focuses on overlapping work across microbatches. You still assign groups of layers to different stages (devices), but instead of waiting for a full batch to traverse the pipeline, you stream multiple smaller microbatches through. While stage 0 processes microbatch `k+1`, stage 1 can process microbatch `k`, and so on. When tuned well, this keeps all devices busy and amortizes communication overhead. When tuned poorly, it suffers from “pipeline bubbles” where some stages sit idle.

Finally, **expert or sharded models** (often seen in Mixture-of-Experts architectures) use a conditional or sparse form of model parallelism. Different “experts” (sub-networks) live on different devices, and a routing mechanism decides which experts to call for each input. Only a subset of the model’s parameters are active per example, so you can scale total parameter count far beyond what a single device could store, while keeping per-example compute roughly constant. The flip side is more complicated routing logic, load balancing, and failure behavior.

These patterns are not mutually exclusive. A modern large language model might use tensor parallelism within layers, pipeline parallelism across groups of layers, and data parallelism across whole replicas. Throughout the rest of this chapter, we’ll keep referring back to these four forms—layer-wise, tensor, pipeline, and experts—and show how different frameworks implement them under friendlier APIs.

## Memory anatomy: parameters, activations, optimizer state, and checkpoints

To understand *why* model parallelism helps, you need a clear picture of where memory actually goes during training. “The model doesn’t fit on one GPU” usually means “some combination of parameters, activations, optimizer state, and temporary buffers exceeds available device memory.” Model parallelism gives you more total memory by spreading those pieces across devices, but it doesn’t change the basic accounting.

**Parameters** are the weights and biases you define in your model. For large transformers or recommendation systems, these alone can reach tens or hundreds of gigabytes. In a simple data-parallel setup, a full copy of all parameters lives on each device. In a model-parallel setup, each device holds only a *slice* of the parameters—specific layers, tensor shards, or experts—reducing per-device parameter memory at the cost of more cross-device communication.

**Activations** are the intermediate tensors produced during the forward pass (e.g., the output of each layer). They are needed again during backpropagation to compute gradients. For deep networks, activation memory can rival or exceed parameter memory, especially with large batch sizes or long sequences. Techniques like activation checkpointing reduce this cost by recomputing some activations instead of storing them, but model parallelism can also help: if different layers live on different devices, their activations are naturally spread out as well.

**Optimizer state** includes whatever extra tensors your optimizer tracks beyond the raw parameters—momentum buffers, Adam’s running means and variances, or more exotic statistics. A common rule of thumb is that Adam-like optimizers use about 2–3× as much memory as the parameters they update. In pure data parallelism, every replica stores a full copy of this state. In many model-parallel schemes, the optimizer state is sharded alongside parameters so that each device is responsible only for the subset of state associated with its own parameter shards.

On top of these, there are **temporary buffers and workspace allocations** used by kernels (for example, cuDNN scratch space) and **checkpoint files** stored on disk. Checkpoints don’t live in GPU memory, but they influence how you design sharding strategies: if parameters are partitioned across devices, you need matching logic to save and restore that partitioning. Frameworks that support distributed or sharded checkpoints try to hide this complexity, but it’s still useful to know that the structure of your model (and its parallelism scheme) leaks into how checkpoints are written. The checkpointing and fault-tolerance patterns in [chapter 16](../chapters/chapter-16.md) build directly on these decisions.

When you plan a model-parallel design, think in terms of this memory anatomy:

- How much parameter and optimizer state will each device own?
- How large are the activations that will flow between devices at each boundary?
- How will checkpoints be written and restored in a way that matches the sharding?

Later sections will connect these questions to concrete strategies in PyTorch, DeepSpeed, Megatron, and friends, but the mental model stays the same: you’re rearranging where parameters, activations, and optimizer state live so that no single device has to carry everything at once.

## Communication patterns and performance pitfalls

Once you start slicing a model across devices, communication becomes a first-class character in your performance story. It’s not enough to know *what* is being sent (activations, gradients, parameters); you also need to understand *how* it moves and how often. Most model-parallel systems lean on a small set of collective patterns:

### Common collective patterns

- **Point-to-point sends/receives** between adjacent stages or layers (common in pipeline or layer-wise parallelism).
- **All-reduce** operations, which combine tensors from all ranks and broadcast the result back (e.g., summing gradients or partial outputs).
- **All-gather** operations, which concatenate or assemble shards from all ranks into a full tensor.
- **Reduce-scatter**, a combination that reduces and splits results in one go (often used in optimizer and tensor-parallel schemes).

Each of these has a cost that depends on tensor size, topology (intra-node vs cross-node), and overlap with compute. A naïve model-parallel implementation can easily spend more time *waiting for data* than doing math.

### Common communication anti-patterns

Some of the most common pitfalls:

- **Too many small messages.** If every layer sends tiny activation tensors or gradients across devices, per-call overhead dominates. It’s usually better to batch communications: send fewer, larger tensors, or fuse multiple small tensors into a single collective.

- **Imbalanced stages.** In pipeline or layer-wise parallelism, if one device has much more work than its neighbors, you get idle time (“bubbles”) before and after that stage. Profiling FLOPs per stage and redistributing layers can make a large difference without changing the overall architecture.

- **Blocking communication.** If your code waits synchronously after each send/receive, you throw away opportunities to overlap communication with compute. Many frameworks provide asynchronous collectives and streams so you can start a transfer, then do useful work while it’s in flight, synchronizing only when the data is actually needed.

- **Cross-node bandwidth assumptions.** In multi-node setups, the network (InfiniBand, Ethernet, etc.) is usually slower and higher latency than intra-node links (NVLink, PCIe, shared memory). A parallelism scheme that works well inside one box may fall over when you stretch it across nodes, simply because intermediate activations or gradients are too large or too frequent.

- **Redundant or unnecessary synchronization.** Extra barriers, poorly placed `torch.cuda.synchronize()` calls, or frequent host/device round-trips can serialize otherwise parallel work. When in doubt, profile with and without those synchronizations and check GPU utilization.

### A basic workflow for communication tuning

The practical workflow is:

1. **Choose a model-parallel strategy** that fits your memory needs (layer, tensor, pipeline, experts).
2. **Sketch the communication pattern** it implies: which tensors move, when, and between which devices.
3. **Profile end-to-end execution** using framework profilers and NCCL/communication traces to see where time is actually going. For a deeper dive into profiling tools and trace viewers, refer back to [chapter 3](../chapters/chapter-3.md).
4. **Iterate on the slow edges**: rebalance stages, adjust microbatch sizes, fuse collectives, or change sharding to reduce the heaviest communication links or better overlap them with compute.

The rest of the chapter will show concrete examples of these patterns in PyTorch, DeepSpeed, and Megatron-style setups, but the underlying goal stays simple: maximize useful compute per unit of communication, and hide as much of that communication as you can behind ongoing work.

## Framework support: PyTorch, DeepSpeed, Megatron, and ecosystem tools

The good news is that you rarely have to wire every send/receive and all-reduce by hand. Modern libraries package common model-parallel patterns into higher-level APIs, so you can focus on *which* strategy you want rather than *how* to implement every communication primitive. The less-good news is that each library comes with its own terminology, defaults, and assumptions about how your code is structured.

### Core PyTorch building blocks

In **PyTorch**, the core building blocks live in `torch.distributed`. You get process groups, collective operations (like all-reduce and all-gather), and higher-level wrappers such as `DistributedDataParallel` (DDP). A *process group* is just the set of ranks that participate in a given set of collectives—for example, “all GPUs in this node” for tensor parallelism, or “all GPUs across nodes” for data parallelism. Classic model parallelism in vanilla PyTorch is mostly manual: you move submodules to different devices, stitch them together in `forward`, and use `torch.distributed` calls where needed. This is flexible and great for learning, but it’s easy to make mistakes with synchronization, device placement, or checkpointing.

### Higher-level libraries on top of PyTorch

On top of PyTorch, libraries like **DeepSpeed** and **FairScale** provide batteries-included approaches. DeepSpeed, for example, combines several ideas:

- **Pipeline parallelism** via stage definitions and microbatching.
- **Tensor/sequence parallelism** for splitting large layers across GPUs.
- **Zero Redundancy Optimizer (ZeRO)** for sharding optimizer state and, optionally, gradients and parameters across ranks.

From a user’s perspective, you mainly configure a JSON file or a Python dict (world size, parallelism dimensions, ZeRO stage), wrap your model with `deepspeed.initialize`, and let it handle the details. The trade-off is that you agree to its runtime model (launch scripts, configuration style, and logging/monitoring stack).

**Megatron-LM** and “Megatron-style” frameworks specialize in scaling transformer models. They define explicit tensor-parallel groups and, often, pipeline-parallel stages, then ship carefully tuned kernels and layer implementations that know how to operate on sharded weights. When you see references to “TP size” or “PP size” in large language model training scripts, they usually come from this style of library. The benefit is strong performance and battle-tested recipes for large models; the cost is that you are working inside a specific architecture and codebase.

### Launchers and orchestration

Around these, there is a growing ecosystem of **launcher and orchestration tools**: PyTorch’s `torchrun`, DeepSpeed’s launcher, Ray Train, and cluster schedulers (Slurm, Kubernetes). They don’t implement model parallelism themselves, but they create and manage the processes, environments, and networking that model-parallel frameworks rely on. Getting comfortable with at least one of these launch paths will make multi-GPU experiments far less painful, and will connect directly to the orchestration patterns discussed in [chapter 12](../chapters/chapter-12.md) and the distributed compute tooling in [chapter 13](../chapters/chapter-13.md).

As you read documentation or examples, keep an eye out for:

- Which *dimensions* of parallelism they support (data, tensor, pipeline, experts).
- How they expect you to structure your model (specific layer types, inheritance, configuration objects).
- How they handle checkpoints and restarts under sharding.
- What profiling and debugging tools they integrate with (for example, PyTorch Profiler via `torch.profiler.profile(...)` or DeepSpeed’s JSON timelines and built-in profiler flags). The observability tools in [chapter 19](../chapters/chapter-19.md) complement these framework-specific options.

## Debugging and monitoring distributed model execution

Model parallelism doesn’t just multiply your available memory; it also multiplies the ways things can go wrong. Bugs that would be obvious in a single-process setup—shape mismatches, NaNs, deadlocks—become harder to reproduce and reason about once multiple processes, devices, and networks are involved. Having a basic debugging and monitoring toolkit is what makes model-parallel systems tolerable in practice.

### Early health checks and invariants

A good starting point is **basic health checks and assertions**. Before you worry about throughput, make sure each rank can run a tiny forward pass, exchange a small tensor with its neighbors, and exit cleanly. Simple assertions on tensor shapes and dtypes at device boundaries (e.g., between pipeline stages or tensor-parallel shards) catch many configuration mistakes early. Logging the current rank, device, and a few key tensor statistics (min/max, mean, number of NaNs/Infs) can quickly pinpoint which part of the model is misbehaving.

When things hang or crash, reach for **structured logging** and **per-rank logs**. Most distributed launchers let you capture stdout/stderr for each rank separately; use that to print concise, rank-tagged messages at key milestones (initialization, first batch, after each pipeline stage). If rank 3 never logs “finished forward”, you know where to start looking. Avoid dumping huge tensors; summaries and shapes are usually enough.

### Profiling and long-run monitoring

For performance issues, you’ll lean on **profilers and communication traces**. A concrete first step in PyTorch is to wrap a handful of training steps with `torch.profiler.profile` (optionally with `with_stack=True` and `record_shapes=True`) and view the resulting trace in TensorBoard or Chrome’s tracing viewer. With DeepSpeed, start by enabling its built-in profilers and JSON timeline logs so you can see step times, kernel breakdowns, and NCCL calls without changing much code. Visualizing these traces helps you see:

- Which ranks are idle vs busy.
- Where all-reduces or all-gathers dominate the timeline.
- Whether communication overlaps with compute or happens in big serialized blocks.

On longer-running jobs, **metrics and dashboards** become essential. Exporting per-rank GPU utilization, memory usage, communication bandwidth, and step times to Prometheus, Grafana, or similar systems lets you spot drifts (e.g., memory slowly climbing on one rank) before they turn into outages. Many frameworks integrate with these tools or provide JSON logs you can scrape. The logging and metrics patterns from [chapter 19](../chapters/chapter-19.md) provide one way to standardize this across projects.

### Reproducing and fixing distributed bugs

Finally, cultivate a **minimal reproducible setup** for distributed bugs. Instead of debugging on a huge cluster, try to reproduce issues with:

- Fewer GPUs (e.g., 2–4 instead of 64).
- A much smaller model and batch size.
- A synthetic dataset.

Most logic bugs and many performance pathologies show up even in these reduced settings, and they’re vastly easier to iterate on. Once your fix works there, scale back up and confirm with full-size runs and proper monitoring.

### A model-parallel training run checklist

To tie these ideas together, you can treat each model-parallel experiment as a short checklist:

1. **Plan the parallelism layout**
   - Choose the mix of data, layer, tensor, pipeline, and/or expert parallelism based on memory limits and architecture boundaries.
   - Sketch which ranks/devices own which layers or shards, and what tensors must cross each boundary.

2. **Do a minimal health check**
   - Launch with a tiny model, batch size of 1, and a few steps.
   - Verify all ranks initialize, run a forward/backward pass, exchange at least one tensor, and shut down cleanly.
   - Add assertions on tensor shapes/dtypes at device boundaries and basic NaN/Inf checks.

3. **Collect a short profiling trace**
   - Run 10–50 training steps with profilers enabled (e.g., `torch.profiler.profile`, DeepSpeed timelines, NCCL traces).
   - Confirm that kernels are running on all ranks and identify obvious idle periods or serialized communication blocks.

4. **Inspect communication and balance**
   - Look for hot collectives (large or frequent all-reduces/all-gathers) and imbalanced stages in pipeline or layer-wise setups.
   - Check whether communication is overlapping with compute or appearing as long, contiguous “gaps” between kernels.

5. **Iterate on the layout and configuration**
   - Adjust stage boundaries, tensor-parallel group sizes, microbatch sizes, and gradient accumulation steps.
   - Where possible, fuse small messages, reduce unnecessary synchronizations, and move non-critical work off the critical path.

6. **Set up long-run monitoring**
   - Once the short run looks healthy, start a longer job with metrics collection enabled (GPU utilization, memory, step time, error rates).
   - Watch for slow drifts (e.g., memory leaks, gradually increasing step time) and confirm that utilization stays high across ranks.

7. **Capture and reuse lessons**
   - Save the effective configuration (parallelism sizes, launch flags, profiler settings) alongside your experiment.
   - Use it as a template for future runs, adjusting only one or two variables at a time so you can attribute changes in behavior.

The rest of the chapter’s hands-on examples will assume you have some form of rank-aware logging, a basic PyTorch or DeepSpeed profiler trace, and minimal metrics in place. With those in your toolkit, model parallelism becomes less of a black box and more of a system you can observe, reason about, and steadily improve. Later case studies in [chapter 18](../chapters/chapter-18.md) revisit these debugging and monitoring ideas in the context of full end-to-end transformer training pipelines.

## Exercises: simple model splits and profiling communication vs compute

The goal of these exercises is to get hands-on experience with the concepts in this chapter: making simple model-parallel changes, then observing how they affect communication patterns and performance. You’ll work with provided code templates and small models so that you can focus on the parallelism aspects without getting lost in model details.

1. **Layer-wise parallelism:** Modify a given single-GPU model to use two GPUs with layer-wise parallelism. Identify a suitable splitting point (e.g., between encoder and decoder), move the layers to the appropriate devices, and adjust the forward pass. Measure the memory savings and verify that the model still trains correctly. Compare the experience to implementing plain data parallelism as in [chapter 8](../chapters/chapter-8.md).

2. **Tensor parallelism:** Take a model with a large linear layer and implement tensor parallelism by sharding the weight matrix across two GPUs. Update the forward and backward passes to handle the sharded tensor. Profile the communication overhead introduced by this parallelism.

3. **Pipeline parallelism:** Implement pipeline parallelism on a small transformer model. Choose appropriate microbatch sizes, and tune the pipeline stages to minimize idle time. Use profiling tools (see [chapter 3](../chapters/chapter-3.md) for tooling ideas) to visualize the compute and communication overlap.

4. **Expert/sharded models:** If you have access to a Mixture-of-Experts setup, experiment with placing different experts on separate GPUs. Analyze the impact on memory usage and training dynamics. If not, simulate a sharded model by partitioning a large embedding table across devices.

5. **Debugging and monitoring:** Intentionally introduce a bug (e.g., shape mismatch, missing all-reduce) in a model-parallel setup and practice using the debugging and monitoring techniques from this chapter to identify and fix the issue. If you’ve already set up centralized logging or tracing as suggested in [chapter 19](../chapters/chapter-19.md), reuse that stack here.

6. **Performance tuning:** Starting with a working model-parallel setup, experiment with different batch sizes, microbatch sizes, and parallelism configurations. Use the profilers and communication traces to guide your optimizations, aiming to reduce communication overhead and improve GPU utilization.

These exercises are designed to be incremental and build on each other. By the end, you should have a solid practical understanding of how to apply model parallelism and diagnose common issues, and you’ll be ready to combine these techniques with the multi-node patterns in [chapter 10](../chapters/chapter-10.md) and the large-scale case studies in [chapter 18](../chapters/chapter-18.md).
