"""Utilities for computing model and QA metrics, plus LLM-as-judge scores.

This module supports being imported as part of the `wfa` package as well as
being executed directly as a script. Relative imports are handled gracefully
when run directly.
"""

# Write function to calculate the accuracy of the model
import os
import sys
import re
from typing import Optional

# Robust import path handling so the file can be executed directly
try:
    from .magic import time_magic as tm
    from ..prompt_library.llm_as_judge_prompts import (
        data_accuracy_prompt,
        explanability_prompt,
        jargon_avoidance_prompt,
        redundancy_avoidance_prompt,
        citation_quality_prompt,
    )
except ImportError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Add project `src` dir to sys.path so `import wfa.*` works when run directly
    SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    # Retry as absolute imports after adjusting sys.path
    from src.wfa.util.magic import time_magic as tm
    from src.wfa.prompt_library.llm_as_judge_prompts import (
        data_accuracy_prompt,
        explanability_prompt,
        jargon_avoidance_prompt,
        redundancy_avoidance_prompt,
        citation_quality_prompt,
    )

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from langchain_litellm import ChatLiteLLM

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

class WildFireMetrics:
    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.roc_auc = 0
        self.ap = 0
        self.cm = None
        self.report = None
        
    def calculate_accuracy(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or accuracy_score is None:
            raise ImportError("scikit-learn is required for accuracy calculation. Please install 'scikit-learn'.")
        acc = accuracy_score(y_true, y_pred)
        self.accuracy = acc
        return acc
        
    def calculate_precision(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or precision_score is None:
            raise ImportError("scikit-learn is required for precision calculation. Please install 'scikit-learn'.")
        prec = precision_score(y_true, y_pred)
        self.precision = prec
        return prec
        
    def calculate_recall(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or recall_score is None:
            raise ImportError("scikit-learn is required for recall calculation. Please install 'scikit-learn'.")
        rec = recall_score(y_true, y_pred)
        self.recall = rec
        return rec
        
    def calculate_f1_score(self, y_true, y_pred):   
        if not _SKLEARN_AVAILABLE or f1_score is None:
            raise ImportError("scikit-learn is required for F1 calculation. Please install 'scikit-learn'.")
        f1 = f1_score(y_true, y_pred)
        self.f1_score = f1
        return f1
        
    def calculate_roc_auc(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or roc_auc_score is None:
            raise ImportError("scikit-learn is required for ROC AUC calculation. Please install 'scikit-learn'.")
        roc_auc = roc_auc_score(y_true, y_pred)
        self.roc_auc = roc_auc
        return roc_auc
        
    def calculate_ap(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or average_precision_score is None:
            raise ImportError("scikit-learn is required for average precision calculation. Please install 'scikit-learn'.")
        ap = average_precision_score(y_true, y_pred)
        self.ap = ap
        return ap
        
    def calculate_confusion_matrix(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or confusion_matrix is None:
            raise ImportError("scikit-learn is required for confusion matrix calculation. Please install 'scikit-learn'.")
        cm = confusion_matrix(y_true, y_pred)
        self.cm = cm
        return cm
        
    def calculate_classification_report(self, y_true, y_pred):
        if not _SKLEARN_AVAILABLE or classification_report is None:
            raise ImportError("scikit-learn is required for classification report. Please install 'scikit-learn'.")
        report = classification_report(y_true, y_pred)
        self.report = report
        return report

class QAMetrics:
    def __init__(self):
        self.bleu_score_1 = 0
        self.bleu_score_4 = 0
        self.rouge_score = 0

    def calculate_bleu(self, references, hypothesis, n_gram=4, weights=None):
        """
        Calculate BLEU score for a candidate sentence.

        :param references: list of list of tokenized references (e.g., [[ref1_tokens], [ref2_tokens], ...])
        :param hypothesis: list of tokenized predicted sentence
        :param n_gram: maximum n-gram considered (default 4)
        :param weights: custom weights for BLEU score (default None, uses uniform for n_gram)
        :return: BLEU score (float)
        """
        if not weights:
            # Uniform weights for up to n_gram
            weights = tuple(1.0 / n_gram for _ in range(n_gram)) + tuple(0.0 for _ in range(4 - n_gram))

        # sentence_bleu expects: references (list of lists), hypothesis (list), weights (tuple)
        bleu_score = sentence_bleu(references, hypothesis, weights=weights[:n_gram])
        return bleu_score

    def calculate_rouge(self, references, hypothesis, use_stemmer=False):
        """
        Calculate ROUGE-N (N=1,2) and ROUGE-L scores for a candidate sentence.

        :param references: list of reference strings (one or more)
        :param hypothesis: candidate string
        :param use_stemmer: whether to use Porter stemmer in scoring (default: False)
        :return: dict with 'rouge-1', 'rouge-2', 'rouge-l' F1-scores
        """
        # rouge_scorer expects strings (not tokenized)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)

        # Compute score for each reference and take the best (for ROUGE-N and L)
        scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
        best_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

        for reference in references:
            result = scorer.score(reference, hypothesis)
            for k, out_key in [('rouge1', 'rouge-1'), ('rouge2', 'rouge-2'), ('rougeL', 'rouge-l')]:
                f1 = result[k].fmeasure
                if f1 > best_scores[out_key]:
                    best_scores[out_key] = f1

        scores.update(best_scores)
        self.rouge_score = scores
        return scores

class LLMAsJudgeMetrics:
    def __init__(self, llm=None, model=None, max_tokens=4000, max_retries=2):
        self.metrics_prompts = {
            'data_accuracy': data_accuracy_prompt,
            'explanability': explanability_prompt,
            'jargon_avoidance': jargon_avoidance_prompt,
            'redundancy_avoidance': redundancy_avoidance_prompt,
            'citation_quality': citation_quality_prompt
        }
        # Allow caller to inject an LLM instance (preferred) or provide a model name
        if llm is not None:
            self.llm = llm
        else:
            model_name = model or "ollama_chat/llama3.1:8b"
            self.llm = ChatLiteLLM(
                model=model_name,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        self.metrics = {}

    def calculate_metrics(self, question, final_answer, generated_answer):
        """
        Calculate all LLM-as-judge metrics and measure the duration for each prompt using a timing magic.
        Stores a dict mapping metric name to (score, time_taken_sec) tuples in self.metrics_with_time.
        """
        self.metrics_with_time = {}
        for metric, prompt in self.metrics_prompts.items():
            formatted_prompt = prompt.format(
                question=question,
                final_answer=final_answer,
                generated_answer=generated_answer
            )

            @tm
            def run_llm(formatted_prompt):
                return self.llm.invoke(formatted_prompt)

            response, elapsed = run_llm(formatted_prompt)
            print('Response for metric: ', metric, ' is: ', response.content, 'time taken: ', elapsed)
            # Extract token 
            um = getattr(response, "usage_metadata", None) or {} 
            rm = getattr(response, "response_metadata", None) or {}
            tu = rm.get("token_usage", {}) or {}
            prompt_tokens = um.get("input_tokens", tu.get("prompt_tokens", 0)) or 0 
            completion_tokens = um.get("output_tokens", tu.get("completion_tokens", 0)) or 0 
            total_tokens = um.get("total_tokens", tu.get("total_tokens", 0)) or 0 
            print('Prompt tokens: ', prompt_tokens, 'Completion tokens: ', completion_tokens, 'Total tokens: ', total_tokens)

            match = re.search(r"[-+]?\d*\.\d+|\d+", response.content)
            score = float(match.group()) if match else None
            self.metrics[metric] = (score, elapsed, prompt_tokens, completion_tokens, total_tokens)

if __name__ == "__main__":
    metrics = LLMAsJudgeMetrics()
    question = "What is the capital of France?"
    final_answer = "Paris"
    generated_answer = "Paris is the capital of France."
    metrics.calculate_metrics(question, final_answer, generated_answer)
    print(metrics.metrics)


# if __name__ == "__main__":

#     wildfire_metrics = WildFireMetrics()
#     y_true = [0, 1, 0, 1, 0]
#     y_pred = [0, 1, 0, 1, 1]
#     wildfire_metrics.calculate_accuracy(y_true, y_pred)
#     wildfire_metrics.calculate_precision(y_true, y_pred)
#     wildfire_metrics.calculate_recall(y_true, y_pred)
#     wildfire_metrics.calculate_f1_score(y_true, y_pred)
#     wildfire_metrics.calculate_roc_auc(y_true, y_pred)
#     wildfire_metrics.calculate_ap(y_true, y_pred)

#     print(f'''
#     Wildfire Metrics: 
#     Accuracy: {wildfire_metrics.accuracy}
#     Precision: {wildfire_metrics.precision}
#     Recall: {wildfire_metrics.recall}
#     F1 Score: {wildfire_metrics.f1_score}
#     ROC AUC: {wildfire_metrics.roc_auc}
#     AP: {wildfire_metrics.ap}
#     ''')

#     metrics = QAMetrics()
#     references = ["The quick brown fox jumps over the lazy dog"]
#     hypothesis = "The quick brown fox jumps over the slacky dog"
    
#     print(f'''
#     QA Metrics: 
#     BLEU Score @1: {metrics.calculate_bleu(references, hypothesis, n_gram=1)}
#     BLEU Score @4: {metrics.calculate_bleu(references, hypothesis, n_gram=4)}
#     ROUGE Score: {metrics.calculate_rouge(references, hypothesis)}
#     ''')

