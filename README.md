# **Hybrid SSM–Transformer Language Model (LLM)**

A high-performance, custom-built language model that integrates **State Space Models (SSMs)** and **optimized attention mechanisms** to overcome the quadratic bottleneck of traditional Transformers.
This project showcases advanced architectural design, GPU-level optimization, and practical deployment techniques for modern LLMs.


## **Overview**

This model uses an **interleaved hybrid architecture**, combining:

* **Selective State Space (Mamba-style) layers** for efficient long-range dependency modeling with **linear time complexity**, and
* **Sliding Window Attention** for precise local reasoning without the cost of full self-attention.

Alongside architectural innovation, the project includes extensive **low-level performance engineering**, including custom CUDA kernels and optimized inference paths.


## **Key Features**

### **1. Hybrid Architecture (Attention + SSM)**

#### **Selective State Space (Mamba) Layers**

* Efficiently capture long-range sequence dependencies.
* Linear complexity enables scaling to large context lengths.
* Supports parallelized recurrence via custom GPU kernels.

#### **Sliding Window Attention**

* Restricts attention to a fixed window size **w**, reducing cost from
  **O(N²)** → **O(N · w)**.
* Provides high-resolution local context understanding.


### **2. Low-Level Performance Optimization**

#### **Custom Triton Kernel – Parallel Scan**

A major bottleneck in SSMs is sequential recurrence.
To overcome this:

* Implemented a **CUDA kernel** for the **parallel scan (associative scan)** algorithm.
* Converts recurrence from sequential → parallel, maximizing GPU throughput.
* Significantly accelerates both training and inference for SSM layers.

#### **Attention Efficiency**

* **Hand-optimized sliding window mask** for minimal compute overhead.
* **Rotary Positional Embeddings (RoPE)** applied to Q/K vectors for scalable relative position encoding and length extrapolation.


### **3. Inference Acceleration**

#### **Key-Value (KV) Caching**

* Stores past keys/values across decoding steps.
* Reduces autoregressive decoding to **O(1)** per token.


##  **Technical Stack**

* **Core:** Python, PyTorch
* **GPU Optimization:** CUDA, NumPy
* **Training:** AdamW, Cross-Entropy Loss


##  **Purpose**

This project is designed as a demonstration of:

* Hybrid LLM architectural design
* GPU-level performance engineering
* State Space Models and efficient attention

    # def get_batch(self, split, block=cfg.block):
    #     encoding = self.train_split if split == "train" else self.val_split
    #
    #     indices = torch.randint(
    #         0, len(encoding) - block, (cfg.batch,))
    #     input = torch.tensor([encoding[i:i+block]
    #                          for i in indices], dtype=torch.long)
    #     output = torch.tensor(
    #         [encoding[i+1:i+1+block] for i in indices],  dtype=torch.long)
    #
    #     return input, output
