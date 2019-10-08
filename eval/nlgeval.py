from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def compute_metrics(hyp_list, ref_list, meteor=True, rouge=True, cider=True,
                    bleu=False, verbose=False):
    '''
    This function is adapted from # git+https://github.com/Maluuba/nlg-eval.git@master
    '''
    refs = {idx: [line] for (idx, line) in enumerate(ref_list)}
    hyps = {idx: [line] for (idx, line) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    scorers = []
    if meteor: scorers.append((Meteor(), "METEOR"))
    if rouge: scorers.append((Rouge(), "ROUGE_L"))
    if cider: scorers.append((Cider(), "CIDEr"))
    if bleu: scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))

    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(refs, hyps)
        except:
            import pdb;pdb.set_trace()
            score, scores = scorer.compute_score(refs, hyps)

        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                if verbose: print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            if verbose: print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
    del scorers
    return ret_scores
