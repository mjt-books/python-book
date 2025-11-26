# Model Parallelism: Slicing Large Models Across Devices

As models grow past the memory limits of a single GPU or accelerator, “just use a bigger batch” and classic data parallelism stop being enough. Model parallelism tackles this by splitting the model itself across devices so that layers, blocks, or even individual tensors live on different pieces of hardware.

In this chapter, you’ll learn how to reason about these splits, what kinds of parallelism exist in practice (tensor, pipeline, expert/sharded), and how modern libraries hide much of the complexity—while still leaving you responsible for debugging and performance tuning. The goal is not to memorize every framework’s API, but to recognize the common patterns so you can scale a single-node script to multi-GPU, multi-accelerator setups without rewriting your whole codebase.

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

Data parallelism is the default scaling tool: you copy the model to multiple devices, split the batch, run forward and backward passes independently, then average gradients. As long as the model fits in the memory of a single device, this works well and is often the simplest, fastest option.

The trouble starts when the *model itself* no longer fits. Large transformer stacks, giant embedding tables, diffusion models, or multi-task systems with many heads can exceed the memory of a single GPU even with a batch size of one. You might first try gradient checkpointing, activation recomputation, or aggressive mixed precision, but at some point you hit a wall: there just isn’t enough device memory to hold all parameters, activations, and optimizer state at once.

Model parallelism becomes necessary once that wall is your main bottleneck rather than throughput. Instead of replicating the entire model on each device, you partition it: different layers, blocks, or tensor shards live on different accelerators. During the forward and backward pass, tensors flow across device boundaries, allowing you to “assemble” a logical model that is much larger than any single GPU.

There are also cases where model parallelism is attractive even before you run out of memory. Certain architectures have natural boundaries (e.g., encoder–decoder splits or expert layers in Mixture-of-Experts models) that map well onto separate devices. Spreading them out can improve utilization, reduce hot spots (like a single giant embedding layer), or let you reuse existing hardware layouts (e.g., one GPU per stage of a pipeline tied to I/O or preprocessing components).

The trade-off is complexity: once the model is spread across devices, you must think about communication costs, load balancing, and failure modes across multiple processes or nodes. In this chapter, we’ll treat model parallelism as a tool you reach for *after* you’ve exhausted simpler options—profiling, better batching, and data parallelism—and we’ll focus on the situations where its added complexity clearly pays off.

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

On top of these, there are **temporary buffers and workspace allocations** used by kernels (for example, cuDNN scratch space) and **checkpoint files** stored on disk. Checkpoints don’t live in GPU memory, but they influence how you design sharding strategies: if parameters are partitioned across devices, you need matching logic to save and restore that partitioning. Frameworks that support distributed or sharded checkpoints try to hide this complexity, but it’s still useful to know that the structure of your model (and its parallelism scheme) leaks into how checkpoints are written.

When you plan a model-parallel design, think in terms of this memory anatomy:

- How much parameter and optimizer state will each device own?
- How large are the activations that will flow between devices at each boundary?
- How will checkpoints be written and restored in a way that matches the sharding?

Later sections will connect these questions to concrete strategies in PyTorch, DeepSpeed, Megatron, and friends, but the mental model stays the same: you’re rearranging where parameters, activations, and optimizer state live so that no single device has to carry everything at once.

## Communication patterns and performance pitfalls

Once you start slicing a model across devices, communication becomes a first-class character in your performance story. It’s not enough to know *what* is being sent (activations, gradients, parameters); you also need to understand *how* it moves and how often. Most model-parallel systems lean on a small set of collective patterns:

- **Point-to-point sends/receives** between adjacent stages or layers (common in pipeline or layer-wise parallelism).
- **All-reduce** operations, which combine tensors from all ranks and broadcast the result back (e.g., summing gradients or partial outputs).
- **All-gather** operations, which concatenate or assemble shards from all ranks into a full tensor.
- **Reduce-scatter**, a combination that reduces and splits results in one go (often used in optimizer and tensor-parallel schemes).

Each of these has a cost that depends on tensor size, topology (intra-node vs cross-node), and overlap with compute. A naïve model-parallel implementation can easily spend more time *waiting for data* than doing math.

Some of the most common pitfalls:

- **Too many small messages.** If every layer sends tiny activation tensors or gradients across devices, per-call overhead dominates. It’s usually better to batch communications: send fewer, larger tensors, or fuse multiple small tensors into a single collective.

- **Imbalanced stages.** In pipeline or layer-wise parallelism, if one device has much more work than its neighbors, you get idle time (“bubbles”) before and after that stage. Profiling FLOPs per stage and redistributing layers can make a large difference without changing the overall architecture.

- **Blocking communication.** If your code waits synchronously after each send/receive, you throw away opportunities to overlap communication with compute. Many frameworks provide asynchronous collectives and streams so you can start a transfer, then do useful work while it’s in flight, synchronizing only when the data is actually needed.

- **Cross-node bandwidth assumptions.** In multi-node setups, the network (InfiniBand, Ethernet, etc.) is usually slower and higher latency than intra-node links (NVLink, PCIe, shared memory). A parallelism scheme that works well inside one box may fall over when you stretch it across nodes, simply because intermediate activations or gradients are too large or too frequent.

- **Redundant or unnecessary synchronization.** Extra barriers, poorly placed `torch.cuda.synchronize()` calls, or frequent host/device round-trips can serialize otherwise parallel work. When in doubt, profile with and without those synchronizations and check GPU utilization.

The practical workflow is:

1. Choose a model-parallel strategy that fits your memory needs.
2. Sketch the communication pattern it implies (which tensors move, when, and between which devices).
3. Use profiling tools (framework profilers, NCCL/communication traces, GPU utilization metrics) to see where time is actually going.
4. Iterate: rebalance stages, adjust microbatch sizes, fuse collectives, or change sharding to reduce the heaviest communication edges.

The rest of the chapter will show concrete examples of these patterns in PyTorch, DeepSpeed, and Megatron-style setups, but the underlying goal stays simple: maximize useful compute per unit of communication, and hide as much of that communication as you can behind ongoing work.

## Framework support: PyTorch, DeepSpeed, Megatron, and ecosystem tools

The good news is that you rarely have to wire every send/receive and all-reduce by hand. Modern libraries package common model-parallel patterns into higher-level APIs, so you can focus on *which* strategy you want rather than *how* to implement every communication primitive. The less-good news is that each library comes with its own terminology, defaults, and assumptions about how your code is structured.

In **PyTorch**, the core building blocks live in `torch.distributed`. You get process groups, collective operations (like all-reduce and all-gather), and higher-level wrappers such as `DistributedDataParallel` (DDP). Classic model parallelism in vanilla PyTorch is mostly manual: you move submodules to different devices, stitch them together in `forward`, and use `torch.distributed` calls where needed. This is flexible and great for learning, but it’s easy to make mistakes with synchronization, device placement, or checkpointing.

On top of PyTorch, libraries like **DeepSpeed** and **FairScale** provide batteries-included approaches. DeepSpeed, for example, combines several ideas:

- **Pipeline parallelism** via stage definitions and microbatching.
- **Tensor/sequence parallelism** for splitting large layers across GPUs.
- **Zero Redundancy Optimizer (ZeRO)** for sharding optimizer state and, optionally, gradients and parameters across ranks.

From a user’s perspective, you mainly configure a JSON file or a Python dict (world size, parallelism dimensions, ZeRO stage), wrap your model with `deepspeed.initialize`, and let it handle the details. The trade-off is that you agree to its runtime model (launch scripts, configuration style, and logging/monitoring stack).

**Megatron-LM** and “Megatron-style” frameworks specialize in scaling transformer models. They define explicit tensor-parallel groups and, often, pipeline-parallel stages, then ship carefully tuned kernels and layer implementations that know how to operate on sharded weights. When you see references to “TP size” or “PP size” in large language model training scripts, they usually come from this style of library. The benefit is strong performance and battle-tested recipes for large models; the cost is that you are working inside a specific architecture and codebase.

Around these, there is a growing ecosystem of **launcher and orchestration tools**: PyTorch’s `torchrun`, DeepSpeed’s launcher, Ray Train, and cluster schedulers (Slurm, Kubernetes). They don’t implement model parallelism themselves, but they create and manage the processes, environments, and networking that model-parallel frameworks rely on. Getting comfortable with at least one of these launch paths will make multi-GPU experiments far less painful.

As you read documentation or examples, keep an eye out for:

- Which *dimensions* of parallelism they support (data, tensor, pipeline, experts).
- How they expect you to structure your model (specific layer types, inheritance, configuration objects).
- How they handle checkpoints and restarts under sharding.
- What profiling and debugging tools they integrate with.

## Debugging and monitoring distributed model execution

Model parallelism doesn’t just multiply your available memory; it also multiplies the ways things can go wrong. Bugs that would be obvious in a single-process setup—shape mismatches, NaNs, deadlocks—become harder to reproduce and reason about once multiple processes, devices, and networks are involved. Having a basic debugging and monitoring toolkit is what makes model-parallel systems tolerable in practice.

A good starting point is **basic health checks and assertions**. Before you worry about throughput, make sure each rank can run a tiny forward pass, exchange a small tensor with its neighbors, and exit cleanly. Simple assertions on tensor shapes and dtypes at device boundaries (e.g., between pipeline stages or tensor-parallel shards) catch many configuration mistakes early. Logging the current rank, device, and a few key tensor statistics (min/max, mean, number of NaNs/Infs) can quickly pinpoint which part of the model is misbehaving.

When things hang or crash, reach for **structured logging** and **per-rank logs**. Most distributed launchers let you capture stdout/stderr for each rank separately; use that to print concise, rank-tagged messages at key milestones (initialization, first batch, after each pipeline stage). If rank 3 never logs “finished forward”, you know where to start looking. Avoid dumping huge tensors; summaries and shapes are usually enough.

For performance issues, you’ll lean on **profilers and communication traces**. Frameworks like PyTorch and DeepSpeed can record timelines that include both compute kernels and NCCL or other communication calls. Visualizing these in tools such as TensorBoard or Chrome’s tracing viewer helps you see:

- Which ranks are idle vs busy.
- Where all-reduces or all-gathers dominate the timeline.
- Whether communication overlaps with compute or happens in big serialized blocks.

On longer-running jobs, **metrics and dashboards** become essential. Exporting per-rank GPU utilization, memory usage, communication bandwidth, and step times to Prometheus, Grafana, or similar systems lets you spot drifts (e.g., memory slowly climbing on one rank) before they turn into outages. Many frameworks integrate with these tools or provide JSON logs you can scrape.

Finally, cultivate a **minimal reproducible setup** for distributed bugs. Instead of debugging on a huge cluster, try to reproduce issues with:

- Fewer GPUs (e.g., 2–4 instead of 64).
- A much smaller model and batch size.
- A synthetic dataset.

Most logic bugs and many performance pathologies show up even in these reduced settings, and they’re vastly easier to iterate on. Once your fix works there, scale back up and confirm with full-size runs and proper monitoring.

The rest of the chapter’s hands-on examples will assume you have some form of rank-aware logging, profiling, and basic metrics in place. With those in your toolkit, model parallelism becomes less of a black box and more of a system you can observe, reason about, and steadily improve.

## Exercises: simple model splits and profiling communication vs compute

The goal of these exercises is to get hands-on experience with the concepts in this chapter: making simple model-parallel changes, then observing how they affect communication patterns and performance. You’ll work with provided code templates and small models so that you can focus on the parallelism aspects without getting lost in model details.

1. **Layer-wise parallelism:** Modify a given single-GPU model to use two GPUs with layer-wise parallelism. Identify a suitable splitting point (e.g., between encoder and decoder), move the layers to the appropriate devices, and adjust the forward pass. Measure the memory savings and verify that the model still trains correctly.

2. **Tensor parallelism:** Take a model with a large linear layer and implement tensor parallelism by sharding the weight matrix across two GPUs. Update the forward and backward passes to handle the sharded tensor. Profile the communication overhead introduced by this parallelism.

3. **Pipeline parallelism:** Implement pipeline parallelism on a small transformer model. Choose appropriate microbatch sizes, and tune the pipeline stages to minimize idle time. Use profiling tools to visualize the compute and communication overlap.

4. **Expert/sharded models:** If you have access to a Mixture-of-Experts setup, experiment with placing different experts on separate GPUs. Analyze the impact on memory usage and training dynamics. If not, simulate a sharded model by partitioning a large embedding table across devices.

5. **Debugging and monitoring:** Intentionally introduce a bug (e.g., shape mismatch, missing all-reduce) in a model-parallel setup and practice using the debugging and monitoring techniques from this chapter to identify and fix the issue.

6. **Performance tuning:** Starting with a working model-parallel setup, experiment with different batch sizes, microbatch sizes, and parallelism configurations. Use the profilers and communication traces to guide your optimizations, aiming to reduce communication overhead and improve GPU utilization.

These exercises are designed to be incremental and build on each other. By the end, you should have a solid practical understanding of how to apply model parallelism and diagnose common issues.
