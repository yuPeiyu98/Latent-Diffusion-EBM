import logging
from collections import Counter
import numpy as np
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from dgmvae.utils import get_dekenize, get_tokenize
import logging
from dgmvae.dataset.corpora import EOS, BOS
from collections import defaultdict
import torch

logger = logging.getLogger()

class EvaluatorBase(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp, domain='default'):
        raise NotImplementedError

    def get_report(self, include_error=False):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BleuEvaluator(EvaluatorBase):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    logger = logging.getLogger(__name__)

    def __init__(self, data_name):
        self.data_name = data_name
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def initialize(self):
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def add_example(self, ref, hyp, domain='default'):
        self.domain_labels[domain].append(ref)
        self.domain_hyps[domain].append(hyp)

    def get_report(self, include_error=False, get_value = False):
        reports = []
        tokenize = get_tokenize()
        
        value = {}

        for domain, labels in self.domain_labels.items():
            predictions = self.domain_hyps[domain]
            self.logger.info("Generate report for {} for {} samples".format(domain, len(predictions)))
            refs, hyps = [], []
            for label, hyp in zip(labels, predictions):
                label = label.replace(EOS, '').replace(BOS, '')
                hyp = hyp.replace(EOS, '').replace(BOS, '')
                ref_tokens = tokenize(label)[2:]
                hyp_tokens = tokenize(hyp)[2:]

                refs.append([ref_tokens])
                hyps.append(hyp_tokens)

            # compute corpus level scores
            bleu = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
            value[domain + "BLEU"] = bleu
            report = "\nDomain: %s BLEU %f\n" % (domain, bleu)
            reports.append(report)

        if get_value:
            return value
        else:
            return "\n==== REPORT ===={report}".format(report="========".join(reports))


class Word2VecEvaluator():
    def __init__(self, word2vec_file):
        print("Loading word2vecs")
        f = open(word2vec_file, "r")
        self.word2vec = {}
        for line in f:
            line_split = line.strip().split()
            word = line_split[0]
            try:
                vecs = list(map(float, line_split[1:]))
            except:
                pass
                # print(line_split)
            self.word2vec[word] = torch.FloatTensor(np.array(vecs))
        f.close()

    def _sent_vec(self, wvecs):
        m = torch.stack(wvecs, dim=0)
        average = torch.mean(m, dim=0)

        extrema_max, _ = torch.max(m, dim=0)
        extrema_min, _ = torch.min(m, dim=0)
        extrema_min_abs = torch.abs(extrema_min)
        extrema = extrema_max * (extrema_max > extrema_min_abs).float() + extrema_min * (extrema_max <= extrema_min_abs).float()

        average = average / torch.sqrt(torch.sum(average * average))
        extrema = extrema / torch.sqrt(torch.sum(extrema * extrema))
        return average, extrema

    def _cosine(self, v1, v2):
        return torch.sum((v1 * v2) / torch.sqrt(torch.sum(v1 * v1)) / torch.sqrt(torch.sum(v2 * v2)))

    def _greedy(self, wlist1, wlist2):

        max_cosine_list = []
        for v1 in wlist1:
            max_cosine = -2.0
            for v2 in wlist2:
                cos = self._cosine(v1, v2)
                max_cosine = max(cos, max_cosine)
            if max_cosine > -2.0:
                max_cosine_list.append(max_cosine)

        simi = sum(max_cosine_list) / len(max_cosine_list)

        return simi.item()


    def eval_from_file(self, fn):
        f = open(fn, "r")
        tgt_s = []
        pred_s = []
        for line in f:
            if "Target:" in line:
                tgt = line[7:].strip().split()
                tgt = [w for w in tgt if w[0] != "<" and w[-1] != ">"] # remove illegal words
                tgt_s.append(tgt)
            if "Predict:" in line:
                pred = line[8:].strip().split()
                pred = [w for w in pred if w[0] != "<" and w[-1] != ">"]  # remove illegal words
                pred_s.append(pred)

        ave_scores = []
        ext_scores = []
        grd_scores = []
        for tgt, pred in zip(tgt_s, pred_s):
            tgt_vecs = [self.word2vec[w] for w in tgt if w in self.word2vec]
            pred_vecs = [self.word2vec[w] for w in pred if w in self.word2vec]
            if len(tgt_vecs) == 0 or len(pred_vecs) == 0:
                continue
            else:
                ave_tgt, ext_tgt = self._sent_vec(tgt_vecs)
                ave_pred, ext_pred = self._sent_vec(pred_vecs)
                ave_scores.append(torch.sum(ave_tgt * ave_pred).item())
                ext_scores.append(torch.sum(ext_tgt * ext_pred).item())
                grd_scores.append((self._greedy(tgt_vecs, pred_vecs) + self._greedy(pred_vecs, tgt_vecs)) / 2)

        logger.info("Average: %lf" % (sum(ave_scores) / len(ave_scores)))
        logger.info("Extrema: %lf" % (sum(ext_scores) / len(ext_scores)))
        logger.info("Greedy: %lf" % (sum(grd_scores) / len(grd_scores)))









