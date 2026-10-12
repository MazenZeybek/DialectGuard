# DialectGuard — Week 1 Model Comparison

## Objective

Compare three Arabic-capable open-weight language models for the DialectGuard prototype, considering Arabic generation, local hardware feasibility, inference speed, licensing, and suitability for subsequent normalization and grounded regeneration.

## Hardware and Testing Environment

- Device: Apple M2 MacBook Pro
- Unified memory: 16 GB
- Local inference: MLX-LM and MLX-VLM
- Test language: Arabic
- Initial test question: ما هي عاصمة الأردن؟
- Expected answer: عمان

The following results are preliminary smoke tests rather than standardized benchmarks.

## Candidate Models

| Property | Qwen3-4B | Gemma 3 4B | Granite 4.0 H 3B Arabic |
|---|---|---|---|
| Model identifier | `mlx-community/Qwen3-4B-4bit` | `mlx-community/gemma-3-4b-it-4bit` | `ibm-research/granite-4.0-h-3b-ar` |
| Model family | Qwen3 | Gemma 3 | Granite 4.0 |
| Approximate parameter class | 4B | 4B | 3B |
| Local inference | Successful | Successful | Successful |
| Arabic smoke test | PASS | PASS | PASS |
| Generation speed | 72.219 tokens/s | Not reported | 26.522 tokens/s |
| Peak reported memory | 2.447 GB MLX | Not reported | 6.623 GB |
| Download size | Not recorded | 3.25 GB | 5.91 GB |
| Framework | MLX-LM 0.32.0 | MLX-VLM 0.7.6 | MLX-LM |
| License | To verify against model repository | Gemma terms; review required | Apache 2.0, as recorded in smoke test |

## Initial Observations

### Qwen3-4B

Qwen3-4B generated the expected Arabic answer in its initial test. With thinking mode disabled, the recorded generation speed was 72.219 tokens per second and peak MLX memory was 2.447 GB.

This makes it a promising candidate for local experimentation on the available hardware. However, the single-prompt speed result does not establish general performance across different prompts or output lengths.

### Gemma 3 4B

Gemma 3 4B successfully generated the expected Arabic answer using MLX-VLM. Its recorded model download was 3.25 GB.

Generation speed and peak memory were not reported in the initial smoke test, so a quantitative speed comparison with the other two models is not yet available. License terms require review before final deployment decisions.

### Granite 4.0 H 3B Arabic

Granite generated the expected Arabic answer and ran successfully on the Apple M2 system.

Recorded measurements were:

- Prompt processing: 38.135 tokens/second
- Generation: 26.522 tokens/second
- Peak memory: 6.623 GB
- Download: 5.91 GB

The test record identifies the model license as Apache 2.0.

Granite was selected as the initial generation model for the Week 1 normalization and evidence-grounded regeneration prototype.

## Week 1 Model Selection

**Selected model for the integrated prototype:** `ibm-research/granite-4.0-h-3b-ar`

The selection is provisional and based on successful Arabic inference, compatibility with the available local hardware, and its use in the implemented normalization and RAG pipeline.

Qwen3 and Gemma remain comparison models for baseline generation and cross-model disagreement analysis.

## Limitations

1. The initial performance measurements were collected from different model smoke tests and are not directly controlled benchmarks.
2. Gemma generation speed and peak memory were not recorded.
3. The tests establish basic Arabic generation functionality, not comprehensive Arabic or dialectal accuracy.
4. Model licenses should be checked against the corresponding upstream repositories before redistribution or deployment.
5. Broader evaluation across Arabic dialects and factual question categories remains separate from this initial hardware feasibility comparison.

## Conclusion

All three candidate models passed the initial Arabic smoke test and could be run locally on the available Apple Silicon hardware.

The Week 1 prototype uses Granite for normalization and grounded regeneration, while retaining Qwen3 and Gemma as baseline comparison models. Final model selection should consider subsequent factual reliability, dialect performance, resource use, and license verification.
