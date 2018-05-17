"""Evaluation utils."""

from collections import Counter
import math
import numpy as np

class BLEU(object):
    def bleu_stats(self, hypothesis, reference):
        """Compute statistics for BLEU."""
        stats = []
        stats.append(len(hypothesis))
        stats.append(len(reference))
        for n in range(1, 5):
            s_ngrams = Counter(
                [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
            )
            r_ngrams = Counter(
                [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
            )
            stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
            stats.append(max([len(hypothesis) + 1 - n, 0]))
        return stats


    def bleu(self, stats):
        """Compute BLEU given n-gram statistics."""
        if len([x for x in stats if x == 0]) > 0:
            return 0
        (c, r) = stats[:2]
        log_bleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
        return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


    def get_bleu(self, hypotheses, reference):
        """Get validation BLEU score for dev set."""
        stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for hyp, ref in zip(hypotheses, reference):
            stats += np.array(self.bleu_stats(hyp, ref))
        return 100 * self.bleu(stats)
