import re
import string
import warnings
from collections import Counter
from pathlib import Path
from tqdm import tqdm


class DatasetLoader:
    """
    Loads all prompt sets once and provides them to all method runs.
    Prompts are loaded once and shared — never reloaded per method.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.synthetic_prompts = []      # list of (prompt_str, seq_len_target)
        self.wikitext_examples = []      # list of str (raw text for ppl eval)
        self.longbench_examples = {}     # task_name -> list of dicts

    def load_all(self):
        """Load all enabled datasets. Call once at startup."""
        if self.config['datasets']['synthetic']['enabled']:
            self._load_synthetic()
        if self.config['datasets']['wikitext']['enabled']:
            self._load_wikitext()
        if self.config['datasets']['longbench']['enabled']:
            self._load_longbench()

    # ── SYNTHETIC ────────────────────────────────────────────────────────────

    def _load_synthetic(self):
        from datasets import load_dataset
        tqdm.write("Loading WikiText-103 for synthetic prompts...")
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        target_lengths = self.config['datasets']['sequence_lengths']
        n_per_length = self.config['datasets']['synthetic']['n_per_length']

        # Filter out empty lines and section headers
        texts = [
            ex['text'] for ex in wiki
            if ex['text'].strip() and not ex['text'].strip().startswith('=')
        ]

        for target_len in target_lengths:
            collected = 0
            passage_start = 0
            while collected < n_per_length and passage_start < len(texts):
                # Build a passage by concatenating wiki text until we reach target_len
                combined = ""
                idx = passage_start
                while idx < len(texts):
                    combined += " " + texts[idx]
                    idx += 1
                    token_ids = self.tokenizer(
                        combined, return_tensors='pt', truncation=False
                    )['input_ids'][0]
                    if len(token_ids) >= target_len:
                        break

                token_ids = self.tokenizer(
                    combined, return_tensors='pt', truncation=False
                )['input_ids'][0]

                if len(token_ids) < target_len:
                    # Not enough text — skip this length
                    break

                # Truncate to exactly target_len tokens
                truncated_ids = token_ids[:target_len]
                truncated_text = self.tokenizer.decode(
                    truncated_ids, skip_special_tokens=True
                )

                prompt = f"Summarize the following text:\n\n{truncated_text}\n\nSummary:"
                self.synthetic_prompts.append({
                    'prompt': prompt,
                    'seq_len': target_len,
                    'prompt_id': f"synthetic_{target_len}_{collected}",
                })
                collected += 1
                passage_start = idx  # start next passage where this one ended

        tqdm.write(f"Loaded {len(self.synthetic_prompts)} synthetic prompts "
                   f"across lengths {target_lengths}")

    # ── WIKITEXT (PERPLEXITY) ─────────────────────────────────────────────────

    def _load_wikitext(self):
        from datasets import load_dataset
        n_examples = self.config['datasets']['wikitext']['n_examples']
        tqdm.write(f"Loading WikiText-103 test set ({n_examples} examples for PPL)...")
        wiki_test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

        count = 0
        for ex in wiki_test:
            if ex['text'].strip() and not ex['text'].strip().startswith('='):
                self.wikitext_examples.append(ex['text'].strip())
                count += 1
                if count >= n_examples:
                    break

        tqdm.write(f"Loaded {len(self.wikitext_examples)} WikiText-103 test examples")

    # ── LONGBENCH ─────────────────────────────────────────────────────────────

    def _load_longbench(self):
        from datasets import load_dataset
        tasks = self.config['datasets']['longbench']['tasks']
        n_per_task = self.config['datasets']['longbench']['n_per_task']
        hf_path = self.config['datasets']['longbench'].get(
            'hf_path', 'vm2825/longbench-llama2-filtered'
        )

        tqdm.write(f"Loading LongBench tasks from {hf_path}: {tasks}")

        for task in tasks:
            try:
                dataset = load_dataset(
                    hf_path, task, split="test",
                    trust_remote_code=(hf_path == 'THUDM/LongBench'),
                )
                examples = []
                for i, ex in enumerate(dataset):
                    if i >= n_per_task:
                        break
                    examples.append(ex)
                self.longbench_examples[task] = examples
                tqdm.write(f"  Loaded {len(examples)} examples for {task}")
            except Exception as e:
                warnings.warn(
                    f"LongBench task '{task}' failed to load: {e}. "
                    "Skipping this task."
                )

        if not self.longbench_examples:
            warnings.warn(
                "All LongBench tasks failed. Falling back to synthetic prompts only."
            )

    # ── PERPLEXITY COMPUTATION ────────────────────────────────────────────────

    def compute_perplexity(self, model, tokenizer, device='cuda', max_length=512):
        """
        Compute perplexity on wikitext_examples using direct forward passes.
        Does NOT go through generate_with_method.
        Returns: float perplexity
        """
        import torch
        import math

        if not self.wikitext_examples:
            return None

        model.eval()
        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in tqdm(self.wikitext_examples, desc="Computing PPL", leave=False):
                enc = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                input_ids = enc['input_ids']
                if input_ids.shape[1] < 2:
                    continue

                outputs = model(input_ids, labels=input_ids)
                # outputs.loss is mean NLL over tokens
                n_tokens = input_ids.shape[1] - 1  # shifted by 1 for LM loss
                total_nll += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

        if total_tokens == 0:
            return None

        return math.exp(total_nll / total_tokens)

    # ── LONGBENCH SCORING ─────────────────────────────────────────────────────

    @staticmethod
    def score_longbench(task, predictions, references):
        """
        Score LongBench predictions.
          QA tasks            : token-level F1 (normalized, Counter-based)
          Summarization tasks : ROUGE-L
          Code tasks          : fuzzy line-similarity (fuzz.ratio)
        references may be list[str] or list[list[str]]; max is taken over multiple refs.
        Returns: float score in [0, 1]
        """
        summarization_tasks = {'gov_report', 'qmsum', 'multi_news'}
        code_tasks = {'lcc', 'repobench-p'}

        if task in summarization_tasks:
            score_fn = DatasetLoader._single_rouge_l
        elif task in code_tasks:
            score_fn = DatasetLoader._single_code_sim
        else:
            score_fn = DatasetLoader._single_f1

        scores = []
        for pred, ref in zip(predictions, references):
            ref_list = ref if isinstance(ref, list) else [ref]
            scores.append(max(score_fn(str(pred), str(r)) for r in ref_list))
        return sum(scores) / len(scores) if scores else 0.0

    # ── per-example score helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize(s):
        """Lowercase, remove articles and punctuation (matches LongBench qa_f1_score)."""
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @staticmethod
    def _single_f1(prediction, ground_truth):
        """Token-level F1 with normalization and Counter (matches LongBench qa_f1_score)."""
        pred_tokens = DatasetLoader._normalize(prediction).split()
        ref_tokens = DatasetLoader._normalize(ground_truth).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _single_rouge_l(prediction, ground_truth):
        """ROUGE-L F1 for a single (prediction, reference) pair."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            return scorer.score(ground_truth, prediction)['rougeL'].fmeasure
        except ImportError:
            warnings.warn("rouge-score not installed, returning 0.0 for ROUGE-L")
            return 0.0

    @staticmethod
    def _single_code_sim(prediction, ground_truth):
        """
        Fuzzy line-similarity for code completion (matches LongBench code_sim_score).
        Takes the first non-comment, non-fence line from prediction.
        """
        try:
            from thefuzz import fuzz
        except ImportError:
            try:
                from fuzzywuzzy import fuzz
            except ImportError:
                warnings.warn("thefuzz not installed; falling back to difflib for lcc")
                import difflib
                return difflib.SequenceMatcher(None, prediction, ground_truth).ratio()

        lines = prediction.lstrip('\n').split('\n')
        first_code_line = ""
        for line in lines:
            if '`' not in line and '#' not in line and '//' not in line:
                first_code_line = line
                break
        return fuzz.ratio(first_code_line, ground_truth) / 100
