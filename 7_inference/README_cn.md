# 大模型推理

推理是指用经过训练的语言模型来进行预测或生成回答的过程。尽管推理看似简单直接，但要大规模高效部署模型，就需要仔细考量诸如性能、成本和可靠性等多种因素。由于大语言模型（LLMs）的规模和计算需求，模型推理存在着一些独特的挑战。

这里，我们借助 [`transformers`](https://huggingface.co/docs/transformers/index) 和 [`text-generation-inference`](https://github.com/huggingface/text-generation-inference) 来探索简单的推理方法以及适用于生产环境的推理方法。针对生产环境的模型部署，我们将重点关注 Text Generation Inference 这个工具包（简称 TGI），它提供了优化过的服务能力。

## 章节概览

大语言模型的推理可以分为两类：简单的使用 pipeline 进行推理，这适用于开发和测试阶段；优化过的服务级推理方案，者适用于生产环境的部署。我们将会讲解这两类方法，从简单的 pipeline 方法开始，再逐步深入到生产环境部署方案。

## 内容目录

### 1. [基本的 Pipeline 推理](./inference_pipeline_cn.md)

这一节你将了解使用 Hugging Face Transformers 的 pipeline 进行基本的推理方法。我们将讲解 pipeline 的设置、生成参数的配置，以及本地部署的最佳实践。该方法适用于原型设计和小规模应用。

### 2. [使用 TGI 进行生产环境部署](./text_generation_inference_cn.md)

这一节你讲学习如何使用 Text Generation Inference 这个工具包进行生产环境的模型部署。我们将探索服务端模型部署的优化技术、组 batch 推理的策略，以及如何监控。TGI 提供了很多适用于生产环境的功能，如健康监测、评估指标、Docker 部署等。

### 练习

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Pipeline Inference | Basic inference with transformers pipeline | 🐢 Set up a basic pipeline <br> 🐕 Configure generation parameters <br> 🦁 Create a simple web server | [Link](./notebooks/basic_pipeline_inference.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/basic_pipeline_inference.ipynb) |
| TGI Deployment | Production deployment with TGI | 🐢 Deploy a model with TGI <br> 🐕 Configure performance optimizations <br> 🦁 Set up monitoring and scaling | [Link](./notebooks/tgi_deployment.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/tgi_deployment.ipynb) |

## Resources

- [Hugging Face Pipeline Tutorial](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference/en/index)
- [Pipeline WebServer Guide](https://huggingface.co/docs/transformers/en/pipeline_tutorial#using-pipelines-for-a-webserver)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [Hugging Face Model Deployment Documentation](https://huggingface.co/docs/inference-endpoints/index)
- [vLLM: High-throughput LLM Serving](https://github.com/vllm-project/vllm)
- [Optimizing Transformer Inference](https://huggingface.co/blog/optimize-transformer-inference)
