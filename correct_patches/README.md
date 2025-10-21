# Correct Patches

Each experiment has its own folder. Inside, each bug has a subfolder containing `.txt` files with the correct (semantically equivalent to developer) patches.

```
correct_patches/
├── gpt4o/                  # GPT-4o: 365 correct / 477 plausible
│   ├── Chart-1/
│   │   └── patch_1.txt
│   ├── Chart-4/
│   │   ├── patch_1.txt
│   │   ├── patch_2.txt     # multiple correct patches for the same bug
│   │   └── ...
│   └── ...
├── qwen_exp1/              # QwenCoder-32B run 1: 252 correct
├── qwen_exp2/              #                run 2: 251 correct
├── qwen_exp3/              #                run 3: 248 correct
├── codellama_exp1/         # CodeLlama-34B run 1: 108 correct
├── codellama_exp2/         #                run 2: 101 correct
├── codellama_exp3/         #                run 3: 103 correct
├── od_gains_gpt4o.md       # Overfitting detector gains
├── od_gains_qwen_exp*.md
└── od_gains_codellama_exp*.md
```

Each `.txt` file contains the file path, method name, and the corrected method body.

### Overfitting Detector Gains

The `od_gains_*.md` files document bugs where **all** pre-refinement patches were overfitting — meaning without OD refinement, these bugs would have zero correct patches. For each bug, the file shows root cause analysis, why the pre-patch overfits, what OD refinement changed, and diffs of the pre-OD patch, post-OD patch, and ground truth fix against the clean buggy code.
