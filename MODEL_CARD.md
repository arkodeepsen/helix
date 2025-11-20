# Helix-100M

- **Language**: English (+ code)
- **License**: Apache-2.0
- **Model type**: Llama-style decoder-only (RMSNorm, SwiGLU, RoPE)
- **Parameters**: ~100M
- **Context length**: 4096 tokens

## Intended Use

- Educational purposes and research
- Small-device experimentation
- Code generation demos
- Tool-calling and agent workflows
- Prototyping language model applications

## Training Data

- Mix of high-quality text and code samples
- The repository includes a minimal example corpus for demonstration
- For production use, replace with curated high-quality datasets

## Training Procedure

1. **Pretraining**: Standard causal language modeling on text corpus
   - Config: `configs/train_pretrain.yaml`
   - ~50k steps recommended for initial training

2. **Supervised Fine-Tuning (SFT)**: Instruction following and tool-calling
   - Config: `configs/train_sft.yaml`
   - ~2k steps on instruction-response pairs

3. **Optional DPO**: Refinement for JSON formatting and response quality
   - Improves structured output generation

## Evaluation

- Perplexity metrics on held-out text
- Code generation benchmarks
- Tool-calling accuracy validation

## Limitations

- As a small 100M parameter model, Helix has limited capacity compared to larger models
- May produce hallucinations or incorrect information
- Not suitable for production use without further training and evaluation
- Limited multilingual capabilities (primarily English)

## Safety & Ethics

- This model is **not aligned** for safety-critical applications
- Users should implement appropriate safeguards for production deployment
- May generate biased or harmful content without proper filtering

## Reproducibility

See the [README](./README.md) for complete training instructions and quickstart guide.

## Citation

If you use this model or training code, please cite the repository.
