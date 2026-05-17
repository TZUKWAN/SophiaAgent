"""LLM evaluation engine: prompt experiments, LLM-as-Judge, RAG evaluation.

Uses the SophiaAgent's provider for LLM calls.  Falls back to heuristic
scoring when the provider is unavailable.  All public methods accept
``args: dict`` and return ``str`` (JSON).
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Optional

from sophia.research._input import resolve_parent_ids

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(result: dict) -> str:
    """Serialize *result* to JSON, converting non-serializable types."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj
    return json.dumps(_convert(result), ensure_ascii=False)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _edit_distance_ratio(s1: str, s2: str) -> float:
    """Levenshtein similarity ratio (0-1)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return 0.0
    if len2 == 0:
        return 0.0

    # Build distance matrix with only two rows for memory efficiency
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    distance = prev[len2]
    max_len = max(len1, len2)
    return 1.0 - distance / max_len if max_len > 0 else 0.0


def _word_overlap_pct(response_tokens: List[str], context_tokens: List[str]) -> float:
    """Percentage of response tokens found in context tokens."""
    if not response_tokens:
        return 1.0
    context_set = set(context_tokens)
    if not context_set:
        return 0.0
    found = sum(1 for t in response_tokens if t in context_set)
    return found / len(response_tokens)


def _rouge_n(generated_tokens: List[str], reference_tokens: List[str], n: int = 1) -> float:
    """ROUGE-N recall: fraction of reference n-grams found in generated."""
    def _ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ref_ngrams = _ngrams(reference_tokens, n)
    gen_ngrams = set(_ngrams(generated_tokens, n))

    if not ref_ngrams:
        return 0.0
    matched = sum(1 for ng in ref_ngrams if ng in gen_ngrams)
    return matched / len(ref_ngrams)


def _count_sentences(text: str) -> int:
    """Rough sentence count by splitting on sentence-ending punctuation."""
    sentences = re.split(r'[.!?]+', text)
    return max(1, len([s for s in sentences if s.strip()]))


def _extract_score_from_text(text: str, scale_max: int) -> Optional[float]:
    """Try to extract a numeric score from LLM judge response."""
    # Look for patterns like "Score: 4", "Rating: 3", "4/5", etc.
    patterns = [
        r'(?:score|rating|grade|evaluation)\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*' + str(scale_max),
        r'(\d+(?:\.\d+)?)\s*(?:out of)\s*' + str(scale_max),
        r'(?:rated|scored|score)\s+(?:as\s+)?(\d+(?:\.\d+)?)',
        r'^.*?(\d+(?:\.\d+)?).*$',  # First number found
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= scale_max:
                    return val
            except (ValueError, IndexError):
                continue
    return None


def _call_provider(provider, prompt: str, model: Optional[str] = None) -> tuple:
    """Call provider.run() and return (response_text, latency_seconds).

    Returns (error_string, 0.0) on failure.
    """
    start = time.time()
    try:
        kwargs: Dict[str, Any] = {"prompt": prompt}
        if model:
            kwargs["model"] = model
        result = provider.run(**kwargs)
        latency = time.time() - start

        # provider.run() may return a string or a dict with "text" / "content"
        if isinstance(result, str):
            return result, latency
        if isinstance(result, dict):
            text = result.get("text") or result.get("content") or result.get("response", "")
            return str(text), latency
        return str(result), latency
    except Exception as exc:
        return f"Error: {exc}", 0.0


# ===========================================================================
# LLMEngine
# ===========================================================================

class LLMEngine:
    """LLM evaluation engine for prompt experiments, LLM-as-Judge, RAG eval.

    Every public method accepts ``args: dict`` (tool-dispatch payload)
    and returns ``str`` (JSON).
    """

    def __init__(self, provider=None, config=None, store=None, guard=None):
        self.provider = provider
        self.config = config
        self.store = store
        self.guard = guard
        self._results_cache: List[dict] = []

    # -----------------------------------------------------------------------
    # ResultStore plumbing
    # -----------------------------------------------------------------------

    def _sanitize_params(self, args: dict) -> dict:
        """Replace bulky arrays / long strings with summaries."""
        clean: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, list):
                if len(v) > 80:
                    clean[k] = f"<list len={len(v)}>"
                elif v and isinstance(v[0], str):
                    total_chars = sum(len(s) for s in v)
                    if total_chars > 4000:
                        clean[k] = f"<list of {len(v)} strings, total_chars={total_chars}>"
                    else:
                        clean[k] = v
                elif v and isinstance(v[0], dict):
                    total = sum(len(str(x)) for x in v)
                    if total > 4000:
                        clean[k] = f"<list of {len(v)} dicts>"
                    else:
                        clean[k] = v
                else:
                    clean[k] = v
            elif isinstance(v, dict):
                total = sum(len(str(x)) for x in v.values())
                if total > 4000:
                    clean[k] = f"<dict keys={len(v)}>"
                else:
                    clean[k] = v
            elif isinstance(v, str) and len(v) > 2000:
                clean[k] = f"<str len={len(v)}>"
            else:
                clean[k] = v
        return clean

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Persist a successful result to the store and embed result_id."""
        if "error" in result:
            return _json(result)
        if self.store is None:
            return _json(result)
        parents = resolve_parent_ids(args)
        sanitized = self._sanitize_params(args)
        rid = self.store.store(
            result,
            kind="result",
            tool=tool_name,
            params=sanitized,
            parents=parents,
        )
        result = {**result, "result_id": rid}
        return _json(result)

    # -----------------------------------------------------------------------
    # prompt_test
    # -----------------------------------------------------------------------

    def prompt_test(self, args: dict) -> str:
        """A/B test prompt variants.

        Args:
            variants: list of {name: str, template: str}
            test_inputs: list of {input: str, expected: str (optional)}
            model: str (optional)

        Returns per-variant results (responses, latency) and comparison table.
        """
        variants = args.get("variants", [])
        test_inputs = args.get("test_inputs", [])
        model = args.get("model")

        if not variants:
            return _json({"error": "variants is required (list of {name, template})."})
        if not test_inputs:
            return _json({"error": "test_inputs is required (list of {input, expected?})."})
        if self.provider is None:
            return _json({"error": "No LLM provider configured. Cannot run prompt tests."})

        results_per_variant: Dict[str, Any] = {}
        comparison_rows = []

        for variant in variants:
            name = variant.get("name", "unnamed")
            template = variant.get("template", "")
            if "{input}" not in template:
                return _json({"error": f"Variant '{name}' template missing {{input}} placeholder."})

            variant_responses = []
            total_latency = 0.0
            match_count = 0

            for ti in test_inputs:
                user_input = ti.get("input", "")
                expected = ti.get("expected", None)

                rendered = template.replace("{input}", user_input)
                response, latency = _call_provider(self.provider, rendered, model)
                total_latency += latency

                is_match = False
                if expected and not response.startswith("Error:"):
                    is_match = expected.lower().strip() in response.lower().strip()
                    if is_match:
                        match_count += 1

                variant_responses.append({
                    "input": user_input,
                    "response": response[:500],  # Truncate for readability
                    "latency_s": round(latency, 4),
                    "expected_match": is_match if expected else None,
                })

            avg_latency = total_latency / len(test_inputs) if test_inputs else 0.0
            match_rate = match_count / len(test_inputs) if test_inputs else 0.0

            results_per_variant[name] = {
                "responses": variant_responses,
                "avg_latency_s": round(avg_latency, 4),
                "total_latency_s": round(total_latency, 4),
                "match_rate": round(match_rate, 4),
            }

            comparison_rows.append({
                "variant": name,
                "avg_latency": round(avg_latency, 4),
                "match_rate": round(match_rate, 4),
                "n_tests": len(test_inputs),
            })

        return self._final(args, {
            "variants": results_per_variant,
            "comparison": comparison_rows,
            "n_variants": len(variants),
            "n_test_inputs": len(test_inputs),
        }, "research_prompt_test")

    # -----------------------------------------------------------------------
    # evaluate
    # -----------------------------------------------------------------------

    def evaluate(self, args: dict) -> str:
        """Evaluate LLM outputs (basic metrics + optional reference comparison).

        Args:
            prompts: list of str
            responses: list of str
            references: list of str (optional)
            criteria: list of str (e.g. ['accuracy', 'relevance', 'coherence'])

        Returns per-response scores and aggregate metrics.
        """
        prompts = args.get("prompts", [])
        responses = args.get("responses", [])
        references = args.get("references", [])
        criteria = args.get("criteria", [])

        if not responses:
            return _json({"error": "responses is required."})

        per_response = []
        for i, resp in enumerate(responses):
            tokens = _tokenize(resp)
            word_count = len(tokens)
            char_count = len(resp)
            sentence_count = _count_sentences(resp)

            metrics: Dict[str, Any] = {
                "index": i,
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(word_count / max(1, sentence_count), 2),
            }

            # Reference-based metrics
            if references and i < len(references):
                ref_tokens = _tokenize(references[i])
                ref_set = set(ref_tokens)
                resp_set = set(tokens)
                metrics["jaccard_similarity"] = round(_jaccard_similarity(resp_set, ref_set), 4)
                metrics["word_overlap"] = round(
                    len(resp_set & ref_set) / max(1, len(ref_set)), 4
                )
                metrics["reference_length_ratio"] = round(
                    word_count / max(1, len(ref_tokens)), 4
                )

            per_response.append(metrics)

        # Aggregate
        n = len(responses)
        agg = {
            "n_responses": n,
            "avg_word_count": round(sum(m["word_count"] for m in per_response) / n, 2),
            "avg_char_count": round(sum(m["char_count"] for m in per_response) / n, 2),
            "avg_sentence_count": round(sum(m["sentence_count"] for m in per_response) / n, 2),
        }

        if references:
            jaccards = [m.get("jaccard_similarity", 0) for m in per_response if "jaccard_similarity" in m]
            if jaccards:
                agg["avg_jaccard_similarity"] = round(sum(jaccards) / len(jaccards), 4)

        return self._final(args, {
            "per_response": per_response,
            "aggregate": agg,
            "criteria_evaluated": criteria if criteria else ["length", "word_count", "sentence_count"],
        }, "research_evaluate")

    # -----------------------------------------------------------------------
    # llm_judge
    # -----------------------------------------------------------------------

    def llm_judge(self, args: dict) -> str:
        """LLM-as-Judge evaluation.

        Args:
            responses: list of {prompt: str, response: str}
            criteria: str or list (e.g. 'helpfulness', 'accuracy', 'safety')
            scale: str ('1-5'|'1-10')
            judge_prompt: str (optional custom judge prompt template)
            reference: str (optional reference answer)
            model: str (optional)

        Returns per-response score, average, distribution.
        """
        responses_input = args.get("responses", [])
        criteria = args.get("criteria", "helpfulness")
        scale = args.get("scale", "1-5")
        judge_prompt_template = args.get("judge_prompt")
        reference = args.get("reference")
        model = args.get("model")

        if not responses_input:
            return _json({"error": "responses is required."})

        if isinstance(criteria, list):
            criteria_str = ", ".join(criteria)
        else:
            criteria_str = str(criteria)

        scale_parts = scale.split("-")
        try:
            scale_min = int(scale_parts[0])
            scale_max = int(scale_parts[-1])
        except (ValueError, IndexError):
            scale_min, scale_max = 1, 5

        per_item = []
        for idx, item in enumerate(responses_input):
            prompt_text = item.get("prompt", "")
            response_text = item.get("response", "")

            if judge_prompt_template:
                judge_prompt = judge_prompt_template.replace("{response}", response_text)
                judge_prompt = judge_prompt.replace("{prompt}", prompt_text)
                judge_prompt = judge_prompt.replace("{criteria}", criteria_str)
                judge_prompt = judge_prompt.replace("{scale}", scale)
            else:
                judge_prompt = (
                    f"You are an expert evaluator. Rate the following response on a scale of "
                    f"{scale_min} to {scale_max} for: {criteria_str}.\n\n"
                    f"Prompt: {prompt_text}\n\n"
                    f"Response: {response_text}\n"
                )
                if reference:
                    judge_prompt += f"\nReference answer: {reference}\n"
                judge_prompt += (
                    f"\nProvide your rating as a single number between {scale_min} and {scale_max}. "
                    f"You may include a brief explanation, but start your response with the numeric score."
                )

            score = None
            judge_response = None

            if self.provider is not None:
                resp_text, latency = _call_provider(self.provider, judge_prompt, model)
                judge_response = resp_text[:300]
                score = _extract_score_from_text(resp_text, scale_max)
            else:
                # Heuristic fallback: score based on response length and keyword overlap
                resp_tokens = _tokenize(response_text)
                prompt_tokens = _tokenize(prompt_text)
                overlap = _jaccard_similarity(set(resp_tokens), set(prompt_tokens))
                length_score = min(1.0, len(resp_tokens) / 50.0)
                heuristic = 0.3 * overlap + 0.7 * length_score
                score = round(scale_min + heuristic * (scale_max - scale_min), 1)
                judge_response = f"Heuristic scoring (no provider): overlap={overlap:.2f}, length_score={length_score:.2f}"

            per_item.append({
                "index": idx,
                "score": score,
                "judge_response": judge_response,
            })

        # Aggregate
        valid_scores = [item["score"] for item in per_item if item["score"] is not None]
        avg_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

        # Distribution
        distribution: Dict[str, int] = {}
        for s in valid_scores:
            key = str(int(round(s))) if s is not None else "N/A"
            distribution[key] = distribution.get(key, 0) + 1

        return self._final(args, {
            "per_response": per_item,
            "average_score": avg_score,
            "n_scored": len(valid_scores),
            "distribution": distribution,
            "criteria": criteria_str,
            "scale": f"{scale_min}-{scale_max}",
        }, "research_llm_judge")

    # -----------------------------------------------------------------------
    # rag_eval
    # -----------------------------------------------------------------------

    def rag_eval(self, args: dict) -> str:
        """RAG evaluation (faithfulness, relevance, completeness).

        Args:
            queries: list of str
            contexts: list of str
            responses: list of str
            references: list of str
            metrics: list of str (e.g. ['faithfulness', 'relevance', 'completeness'])

        Returns per-query scores and aggregate.
        """
        queries = args.get("queries", [])
        contexts = args.get("contexts", [])
        responses_list = args.get("responses", [])
        references = args.get("references", [])
        metrics = args.get("metrics", ["faithfulness", "relevance", "completeness"])

        n = len(queries)
        if n == 0:
            return _json({"error": "queries is required."})

        # Pad lists if needed
        while len(contexts) < n:
            contexts.append("")
        while len(responses_list) < n:
            responses_list.append("")
        while len(references) < n:
            references.append("")

        per_query = []
        for i in range(n):
            query_tokens = _tokenize(queries[i])
            context_tokens = _tokenize(contexts[i])
            response_tokens = _tokenize(responses_list[i])
            reference_tokens = _tokenize(references[i])

            scores: Dict[str, Any] = {"index": i, "query": queries[i][:100]}

            if "faithfulness" in metrics:
                # What % of response words appear in context
                scores["faithfulness"] = round(
                    _word_overlap_pct(response_tokens, context_tokens), 4
                )

            if "relevance" in metrics:
                # Word overlap between context and query
                scores["relevance"] = round(
                    _jaccard_similarity(set(context_tokens), set(query_tokens)), 4
                )

            if "completeness" in metrics and reference_tokens:
                # Word overlap between response and reference
                scores["completeness"] = round(
                    _jaccard_similarity(set(response_tokens), set(reference_tokens)), 4
                )

            per_query.append(scores)

        # Aggregate
        agg: Dict[str, Any] = {"n_queries": n}
        for metric in metrics:
            vals = [s.get(metric) for s in per_query if metric in s]
            if vals:
                agg[f"avg_{metric}"] = round(sum(vals) / len(vals), 4)
                agg[f"min_{metric}"] = round(min(vals), 4)
                agg[f"max_{metric}"] = round(max(vals), 4)

        return self._final(args, {
            "per_query": per_query,
            "aggregate": agg,
            "metrics_computed": metrics,
        }, "research_rag_eval")

    # -----------------------------------------------------------------------
    # benchmark
    # -----------------------------------------------------------------------

    def benchmark(self, args: dict) -> str:
        """Run benchmark evaluation.

        Args:
            dataset: list of {input: str, expected: str}
            max_samples: int (optional)

        Returns accuracy, per_sample_results, errors.
        """
        dataset = args.get("dataset", [])
        max_samples = args.get("max_samples")

        if not dataset:
            return _json({"error": "dataset is required (list of {input, expected})."})

        if self.provider is None:
            return _json({"error": "No LLM provider configured. Cannot run benchmark."})

        if max_samples:
            dataset = dataset[:max_samples]

        per_sample = []
        exact_matches = 0
        contains_matches = 0
        errors = []

        for i, sample in enumerate(dataset):
            user_input = sample.get("input", "")
            expected = sample.get("expected", "")

            response, latency = _call_provider(self.provider, user_input)

            is_error = response.startswith("Error:")
            exact = False
            contains = False

            if not is_error:
                resp_clean = response.strip().lower()
                exp_clean = expected.strip().lower()
                exact = resp_clean == exp_clean
                contains = exp_clean in resp_clean

                if exact:
                    exact_matches += 1
                if contains:
                    contains_matches += 1
            else:
                errors.append({"index": i, "error": response})

            per_sample.append({
                "index": i,
                "input": user_input[:200],
                "expected": expected[:200],
                "response": response[:200],
                "exact_match": exact,
                "contains_match": contains,
                "latency_s": round(latency, 4),
                "error": is_error,
            })

        n = len(dataset)
        return self._final(args, {
            "n_samples": n,
            "exact_match_accuracy": round(exact_matches / n, 4) if n else 0.0,
            "contains_match_accuracy": round(contains_matches / n, 4) if n else 0.0,
            "n_errors": len(errors),
            "per_sample": per_sample,
            "errors": errors[:10],  # Cap error list
        }, "research_benchmark")

    # -----------------------------------------------------------------------
    # quality_score
    # -----------------------------------------------------------------------

    def quality_score(self, args: dict) -> str:
        """Output quality scoring (BLEU, ROUGE-like, Jaccard, edit distance).

        Args:
            generated: list of str
            references: list of str
            metrics: list of str (e.g. ['bleu', 'rouge', 'jaccard', 'edit_distance'])

        Returns per_sample and aggregate scores.
        """
        generated = args.get("generated", [])
        references = args.get("references", [])
        metrics = args.get("metrics", ["bleu", "rouge", "jaccard", "edit_distance"])

        if not generated:
            return _json({"error": "generated is required."})
        if not references:
            return _json({"error": "references is required."})

        n = min(len(generated), len(references))
        per_sample = []

        for i in range(n):
            gen_text = generated[i]
            ref_text = references[i]
            gen_tokens = _tokenize(gen_text)
            ref_tokens = _tokenize(ref_text)
            gen_set = set(gen_tokens)
            ref_set = set(ref_tokens)

            scores: Dict[str, Any] = {"index": i}

            if "bleu" in metrics:
                if HAS_NLTK and ref_tokens:
                    try:
                        bleu = sentence_bleu(
                            [ref_tokens], gen_tokens,
                            smoothing_function=SmoothingFunction().method1
                        )
                        scores["bleu"] = round(bleu, 4)
                    except Exception:
                        # Manual BLEU fallback
                        scores["bleu"] = round(_manual_bleu(gen_tokens, ref_tokens), 4)
                elif ref_tokens:
                    scores["bleu"] = round(_manual_bleu(gen_tokens, ref_tokens), 4)
                else:
                    scores["bleu"] = 0.0

            if "rouge" in metrics:
                scores["rouge_1_recall"] = round(_rouge_n(gen_tokens, ref_tokens, 1), 4)
                scores["rouge_2_recall"] = round(_rouge_n(gen_tokens, ref_tokens, 2), 4)

            if "jaccard" in metrics:
                scores["jaccard"] = round(_jaccard_similarity(gen_set, ref_set), 4)

            if "edit_distance" in metrics:
                scores["edit_distance_ratio"] = round(_edit_distance_ratio(gen_text, ref_text), 4)

            per_sample.append(scores)

        # Aggregate
        agg: Dict[str, Any] = {"n_samples": n}
        all_metric_keys = set()
        for s in per_sample:
            for k in s:
                if k != "index":
                    all_metric_keys.add(k)

        for key in sorted(all_metric_keys):
            vals = [s[key] for s in per_sample if key in s]
            if vals:
                agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4)
                agg[f"min_{key}"] = round(min(vals), 4)
                agg[f"max_{key}"] = round(max(vals), 4)

        return self._final(args, {
            "per_sample": per_sample,
            "aggregate": agg,
            "metrics_computed": metrics,
        }, "research_quality_score")

    # -----------------------------------------------------------------------
    # prompt_generate
    # -----------------------------------------------------------------------

    def prompt_generate(self, args: dict) -> str:
        """Generate prompt variants for testing.

        Args:
            base_prompt: str
            n_variants: int (default 3)
            variation_type: str ('rephrase'|'add_constraints'|'change_role')

        Returns list of prompt variants.
        """
        base_prompt = args.get("base_prompt", "")
        n_variants = int(args.get("n_variants", 3))
        variation_type = args.get("variation_type", "rephrase")

        if not base_prompt:
            return _json({"error": "base_prompt is required."})

        # Try LLM-based generation first
        if self.provider is not None:
            meta_prompt = (
                f"Generate {n_variants} different versions of the following prompt. "
                f"The variation style should be: {variation_type}.\n\n"
                f"Original prompt:\n{base_prompt}\n\n"
                f"Return ONLY the {n_variants} variant prompts, numbered 1-{n_variants}, "
                f"each on its own section clearly separated."
            )
            response, _ = _call_provider(self.provider, meta_prompt)
            if not response.startswith("Error:"):
                # Parse numbered variants
                variants = _parse_numbered_variants(response, n_variants)
                if variants:
                    return self._final(args, {
                        "method": "llm_generated",
                        "base_prompt": base_prompt,
                        "variation_type": variation_type,
                        "variants": variants,
                    }, "research_prompt_generate")

        # Fallback: simple deterministic transformations
        variants = []
        if variation_type == "rephrase":
            # Reorder sentences
            sentences = re.split(r'(?<=[.!?])\s+', base_prompt)
            if len(sentences) > 1:
                variants.append(" ".join(reversed(sentences)))
            # Add detail request
            variants.append(base_prompt + " Please provide a detailed and thorough response.")
            # Make it a question
            if not base_prompt.strip().endswith("?"):
                variants.append(
                    base_prompt.rstrip(".!?") + "? Please explain your reasoning."
                )
            # Add examples request
            if len(variants) < n_variants:
                variants.append(
                    base_prompt + " Include specific examples to illustrate your points."
                )

        elif variation_type == "add_constraints":
            constraints = [
                " Be concise and accurate.",
                " Structure your answer with clear headings.",
                " Limit your response to under 200 words.",
                " Use bullet points for clarity.",
            ]
            for c in constraints[:n_variants]:
                variants.append(base_prompt + c)

        elif variation_type == "change_role":
            roles = [
                "You are an expert researcher in this field. ",
                "You are a senior professor with decades of experience. ",
                "You are a critical thinker who evaluates evidence rigorously. ",
                "You are a practical consultant who gives actionable advice. ",
            ]
            for r in roles[:n_variants]:
                variants.append(r + base_prompt)

        else:
            return _json({"error": f"Unknown variation_type: {variation_type}"})

        # Pad if needed
        while len(variants) < n_variants:
            variants.append(base_prompt + " [Variant " + str(len(variants) + 1) + "]")

        return self._final(args, {
            "method": "heuristic",
            "base_prompt": base_prompt,
            "variation_type": variation_type,
            "variants": variants[:n_variants],
        }, "research_prompt_generate")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _manual_bleu(gen_tokens: List[str], ref_tokens: List[str]) -> float:
    """Simplified BLEU score without NLTK."""
    if not gen_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(gen_tokens))) if len(gen_tokens) < len(ref_tokens) else 1.0

    # Unigram precision
    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    clipped = 0
    gen_counts: Dict[str, int] = {}
    for t in gen_tokens:
        gen_counts[t] = gen_counts.get(t, 0) + 1

    for t, count in gen_counts.items():
        clipped += min(count, ref_counts.get(t, 0))

    precision = clipped / len(gen_tokens) if gen_tokens else 0.0
    return bp * precision


def _parse_numbered_variants(text: str, expected: int) -> List[str]:
    """Parse numbered variants from LLM response."""
    # Try pattern: "1. ..." or "1) ..." or "Variant 1: ..."
    variants = []
    pattern = r'(?:^|\n)\s*(?:\d+[\.\)]|Variant\s+\d+\s*[:\-.])\s*(.*?)(?=(?:\n\s*(?:\d+[\.\)]|Variant\s+\d+\s*[:\-.]))|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        cleaned = m.strip()
        if cleaned:
            variants.append(cleaned)

    if len(variants) >= expected:
        return variants[:expected]

    # Fallback: split by double newlines
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    result = []
    for p in parts:
        # Remove leading numbers
        cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', p).strip()
        if cleaned and len(cleaned) > 10:
            result.append(cleaned)

    return result[:expected] if result else []
