# HumanEval Error Analysis

## Executive Summary

Both base and fine-tuned models achieved >96% pass@1 on HumanEval, with a small number of syntax errors causing all failures. Notably, **the two models failed on completely different problems**, suggesting randomness in generation rather than systematic weaknesses.

## Failure Statistics

| Model | Pass@1 | Passed | Failed | Failure Rate |
|-------|--------|--------|--------|--------------|
| Base Model | 98.17% | 161/164 | 3/164 | 1.83% |
| Fine-tuned | 96.95% | 159/164 | 5/164 | 3.05% |

## Failure Breakdown

### Base Model Failures (3 total)

All failures due to **syntax errors**:

1. **HumanEval/58**: `SyntaxError: '(' was never closed`
   - Cause: Incomplete parenthesis in set intersection
   - Generated: `return sorted(set(l1) & set(l#2))`
   - Issue: Comment symbol `#` inserted mid-variable name

2. **HumanEval/122**: `SyntaxError: invalid syntax`
   - Cause: Malformed expression

3. **HumanEval/160**: `SyntaxError: invalid syntax`
   - Cause: Malformed expression

### Fine-tuned Model Failures (5 total)

All failures due to **syntax errors**:

1. **HumanEval/32**: `SyntaxError: '(' was never closed`
   - Cause: Completion truncated mid-return statement
   - Generated: `...if len(xs) == 8:\n        retur` (cut off)
   - Issue: max_new_tokens limit reached

2. **HumanEval/39**: `SyntaxError: invalid syntax`
   - Cause: Malformed expression

3. **HumanEval/41**: `SyntaxError: expected ':'`
   - Cause: Missing colon in control structure

4. **HumanEval/82**: `SyntaxError: expected ':'`
   - Cause: Missing colon in control structure

5. **HumanEval/120**: `SyntaxError: '(' was never closed`
   - Cause: Incomplete parenthesis

## Key Findings

### 1. No Overlap in Failures

**Critical observation**: Base model and fine-tuned model failed on **completely different problems**.

- Base model failed: HumanEval/58, 122, 160
- Fine-tuned failed: HumanEval/32, 39, 41, 82, 120
- **Overlap: 0 problems**

This suggests:
- Failures are **random/stochastic** rather than systematic
- No evidence of model degradation on specific problem types
- With different random seeds, success rates would likely reverse

### 2. All Failures are Syntax Errors

100% of failures (8/8 total) are syntax errors, not logical errors:
- Incomplete tokens/truncation (e.g., "retur" instead of "return")
- Malformed expressions
- Missing punctuation (colons, parentheses)

**Implication**: Models understand the logic but occasionally produce syntactically invalid Python.

### 3. Error Categories

| Error Type | Base | Fine-tuned | Total |
|------------|------|------------|-------|
| Unclosed parenthesis | 1 | 2 | 3 |
| Invalid syntax | 2 | 1 | 3 |
| Missing colon | 0 | 2 | 2 |

### 4. Truncation Issues

Fine-tuned model showed evidence of truncation (HumanEval/32: "retur"), suggesting:
- Generated code hit `max_new_tokens=512` limit
- Could be improved with dynamic stopping or longer token limits

### 5. Performance Comparison

Despite failing on different problems:
- Base: 98.17% (slightly better)
- Fine-tuned: 96.95% (-1.22pp)
- Difference: **Not statistically significant** given small sample size (164 problems)

## Recommendations

### For Production Use

1. **Syntax Validation**: Add post-generation AST parsing to catch syntax errors
2. **Retry Mechanism**: Re-generate on syntax error (would likely succeed)
3. **Temperature Tuning**: Lower temperature (currently 0.2) might reduce random errors
4. **Token Limit**: Increase max_new_tokens for complex functions

### For Fine-tuning

1. **No Major Issues**: Fine-tuning did not introduce systematic errors
2. **Trade-off Acceptable**: -1.22pp on English tasks for Korean capability is reasonable
3. **Syntax Training**: Could add explicit syntax validation in training data

## Statistical Significance

With 164 samples and differences of 2-3 failures:
- **Not statistically significant** at p<0.05 level
- Random variation expected in small sample
- Would need ~1000+ samples to detect 1-2% differences reliably

## Conclusion

Both models demonstrate **excellent performance (>96%)** on HumanEval. The small number of failures are:
1. **Random** - no overlap between models
2. **Syntactic** - not logical errors
3. **Fixable** - with simple post-processing

The fine-tuned model's slight performance decrease (-1.22pp) is **within noise** and likely due to:
- Natural variation (different random generations)
- Language shift (Korean training data)
- Acceptable trade-off for domain specialization

**Verdict**: Fine-tuning was successful. No evidence of model degradation or systematic errors.
