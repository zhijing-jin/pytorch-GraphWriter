import os
import json
import itertools

import torch
from collections import defaultdict
from time import time

import torchtext
from torchtext.data import Field, RawField, TabularDataset, Iterator, \
    BucketIterator

import try_utils
from try_utils import Struct, ENT0_END, ENT1_END, tensor2list, \
    show_var, show_time, fwrite


class Dataset:
    def __init__(self, proc_id=0,
                 data_dir='./tmp/', train_fname='train.csv', skip_header=True,
                 preprocessed=True, lower=True, max_input_sent_len=450,
                 vocab_max_size=50000, emb_dim=100,
                 save_vocab_fname='vocab.json', verbose=True,
                 ):

        self.verbose = verbose and (proc_id == 0)

        if self.verbose:
            show_time("[Info] Start building TabularDataset from: {}"
                      .format(os.path.join(data_dir, 'train.csv')))

        self._get_datafields(lower=lower, preprocessed=preprocessed,
                             max_input_sent_len=max_input_sent_len)

        datasets = \
            TabularDataset.splits(
                path=data_dir,
                train=train_fname,
                validation=train_fname.replace("train", "valid"),
                test=train_fname.replace("train", "test"),
                format=train_fname.split('.')[-1],
                skip_header=skip_header,
                fields=self.fields,
            )
        self.train_ds, self.valid_ds, self.test_ds = datasets

        self._build_vocab(datasets, vocab_max_size=vocab_max_size,
                          emb_dim=emb_dim,
                          save_to=save_vocab_fname)

        # ENTITY.vocab.freqs.most_common(10)
        # train_ds[0].__dict__.keys()
        # train_ds[0].tgt
        #
        # len(REL.vocab)
        # len(train_ds)
        # self.train_ds[-1].ners

        if self.verbose:
            msg = "[Info] Finished building vocab: {} ENTITY, {} NER, {} REL, {} TGT" \
                .format(len(self.ENTITY.vocab), len(self.NER.vocab),
                        len(self.REL.vocab), len(self.TGT.vocab))
            show_time(msg)

    def _get_datafields(self, lower=True, max_input_sent_len=450,
                        preprocessed=True):
        tokenize = lambda x: x.split()[:max_input_sent_len] \
            if preprocessed else 'spacy'

        self.ENTITY = Field(sequential=True, batch_first=True,
                            lower=lower)
        self.NER = Field(sequential=True, batch_first=True,
                         lower=lower)
        self.REL = Field(sequential=True, batch_first=True)
        self.ORDERED_REL = Field(sequential=True, batch_first=True)
        self.NUM = Field(sequential=True, batch_first=True, use_vocab=False)
        self.STR = RawField()
        self.SHOW_INP = RawField()
        self.TGT = Field(sequential=True, batch_first=True,
                         init_token="<bos>", eos_token="<eos>",
                         include_lengths=True, lower=lower, tokenize=tokenize)
        self.TGT_NON_TEMPL = Field(sequential=True, batch_first=True,
                                   init_token=ENT1_END, eos_token=ENT0_END,
                                   lower=lower, tokenize=tokenize)
        self.TGT_TXT = Field(sequential=True, batch_first=True,
                             init_token="<bos>", eos_token="<eos>",
                             include_lengths=True, lower=lower)
        self.ADJ_SEQ = Field(sequential=True, batch_first=True,
                             init_token="<bos>", eos_token="<eos>",
                             include_lengths=True, lower=lower)

        self.fields = [("triples", None),
                       ("tgt", self.TGT),
                       ("tgt_non_templ", self.TGT_NON_TEMPL),
                       ("tgt_txt", self.TGT_TXT),

                       ("ents", self.ENTITY),
                       ("ners", self.NER),
                       ("rels", self.REL),
                       ("ordered_rels", self.ORDERED_REL),
                       ("ordered_ents", self.STR),

                       ("ent_lens", self.STR),
                       ("ner2ent", self.STR),

                       ("adj", self.STR),
                       ("adj_seq", self.ADJ_SEQ),
                       ("show_inp", self.SHOW_INP),
                       ]

    def _build_vocab(self, datasets, vocab_max_size=50000, emb_dim=100,
                     save_to=None):
        self.ENTITY.build_vocab(*datasets)
        self.REL.build_vocab(*datasets)
        self.ORDERED_REL.build_vocab(*datasets)
        self.NER.build_vocab(*datasets)
        self.TGT_TXT.build_vocab(*datasets, max_size=vocab_max_size)
        self.TGT_NON_TEMPL.build_vocab(*datasets, max_size=vocab_max_size,
                                       specials=["<bos>", "<eos>"],)
        self.TGT.build_vocab(*datasets, max_size=vocab_max_size)
        self.ADJ_SEQ.build_vocab(*datasets)

        # INPUT.build_vocab(*datasets, max_size=vocab_max_size,
        #                   vectors=GloVe(name='6B', dim=emb_dim),
        #                   unk_init=torch.Tensor.normal_, )
        # # load_vocab(hard_dosk) like opennmt
        # # use Elmo instead

        self.TGT.vocab.itos = [i for i in self.TGT.vocab.itos
                               if i not in set(self.NER.vocab.itos[2:])]
        self.tgt_no_ner_vocab_size = len(self.TGT.vocab.itos)

        assert not (set(self.TGT.vocab.itos) - set(self.TGT_NON_TEMPL.vocab.itos))

        self.TGT_NON_TEMPL.vocab.itos = \
            self.TGT.vocab.itos + \
            [i for i in self.TGT_NON_TEMPL.vocab.itos
             if i not in set(self.TGT.vocab.itos)]
        self.TGT_NON_TEMPL.vocab.stoi.update(
            {s: i for i, s in enumerate(self.TGT_NON_TEMPL.vocab.itos)})
        assert set(self.TGT_NON_TEMPL.vocab.stoi.keys()) == set(
            self.TGT_NON_TEMPL.vocab.itos)

        self.TGT.vocab.itos += self.NER.vocab.itos[2:]
        self.TGT.vocab.stoi.update(
            {s: i for i, s in enumerate(self.TGT.vocab.itos)})

        self.vocab_size = {
            'ents': len(self.ENTITY.vocab),
            'rels': len(self.REL.vocab),
            'seqs': len(self.TGT.vocab),
            'tgt_no_ner': self.tgt_no_ner_vocab_size,
        }
        self.ent_itos = self.ENTITY.vocab.itos
        self.tgt_itos = self.TGT.vocab.itos
        self.tgt_stoi = self.TGT.vocab.stoi

        if False:
            if self.verbose and (save_to is not None):
                writeout = {
                    # 'rel_vocab':
                    #     {
                    #         'itos': self.REL.vocab.itos,
                    #         'stoi': self.REL.vocab.stoi,
                    #     },
                    # 'ner_vocab':
                    #     {
                    #         'itos': self.NER.vocab.itos,
                    #         'stoi': self.NER.vocab.stoi,
                    #     },
                    # 'entity_vocab':
                    #     {
                    #         'itos': self.ENTITY.vocab.itos,
                    #         'stoi': self.ENTITY.vocab.stoi,
                    #     },
                    # 'tgt_vocab':
                    #     {
                    #         'itos': self.TGT.vocab.itos,
                    #         'stoi': self.TGT.vocab.stoi,
                    #     },
                    't2g_tgt_vocab':
                        {
                            'itos': self.ORDERED_REL.vocab.itos,
                            'stoi': self.ORDERED_REL.vocab.stoi,
                        },
                    't2g_inp_vocab':
                        {
                            'itos': self.TGT_NON_TEMPL.vocab.itos,
                            'stoi': self.TGT_NON_TEMPL.vocab.stoi,
                        },

                    'g2t_tgt_vocab':
                        {
                            'itos': [],
                            'stoi': [],
                        },
                    'g2t_inp_vocab':
                        {
                            'itos': [],
                            'stoi': [],
                        },

                }
                fwrite(json.dumps(writeout, indent=4), save_to)

        if self.verbose and (save_to is not None):
            import os
            import json
            from efficiency.log import fwrite
            folder = '/home/ubuntu/proj/zhijing_g/tmp_intermediate/'
            if not os.path.isdir(folder):
                os.mkdir(folder)

            import pdb;pdb.set_trace()
            fwrite(json.dumps(
                {
                    'itos': self.REL.vocab.itos,
                    'stoi': self.REL.vocab.stoi,
                }
            ), folder + 'try_rel_vocab.json')

            fwrite(json.dumps(
                {
                    'itos': self.OUTP.vocab.itos,
                    'stoi': self.OUTP.vocab.stoi,
                }
            ), folder + 'outp_vocab.json')
            fwrite(json.dumps(
                {
                    'itos': self.TGT.vocab.itos,
                    'stoi': self.TGT.vocab.stoi,
                }
            ), folder + 'tgt_vocab.json')
            fwrite(json.dumps(
                {
                    'itos': self.NERD.vocab.itos,
                    'stoi': self.NERD.vocab.stoi,
                }
            ), folder + 'nerd_vocab.json')
            fwrite(json.dumps(
                {
                    'itos': self.INP.vocab.itos,
                    'stoi': self.INP.vocab.stoi,
                }
            ), folder + 'inp_vocab.json')
            fwrite(json.dumps(
                {
                    'itos': self.ENT.itos,
                    'stoi': self.ENT.stoi,
                }
            ), folder + 'ent_vocab.json')

            import pdb;pdb.set_trace()


    def get_dataloader(self, proc_id=0, n_gpus=1, batch_size=8,
                       device=torch.device('cpu'),
                       use_dgl=False, timer=False):
        def _distribute_dataset(dataset):
            n = len(dataset)
            part = dataset[n * proc_id // n_gpus: n * (proc_id + 1) // n_gpus]
            return torchtext.data.Dataset(part, dataset.fields)

        train_ds = _distribute_dataset(self.train_ds)
        # self.valid_ds = _distribute_dataset(self.valid_ds)
        # self.test_ds = _distribute_dataset(self.test_ds)

        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, self.valid_ds),
            batch_sizes=(batch_size, batch_size),
            # batch_size_fn,
            sort_within_batch=True,
            sort_key=lambda x: len(x.tgt),
            repeat=False,
            device=device,
        )

        # batch = next(train_iter.__iter__())
        # batch.__dict__.keys()

        test_iter = Iterator(self.test_ds, batch_size=1, train=False,
                             sort=False, sort_within_batch=False, repeat=False,
                             device=device)

        train_dl = BatchWrapper(train_iter, self.tgt_no_ner_vocab_size,
                                use_dgl=use_dgl, timer=timer)
        valid_dl = BatchWrapper(valid_iter, self.tgt_no_ner_vocab_size,
                                use_dgl=use_dgl)
        test_dl = BatchWrapper(test_iter, self.tgt_no_ner_vocab_size,
                               use_dgl=use_dgl)

        # next(train_dl.__iter__())

        return train_dl, valid_dl, test_dl


class Wrapper:
    def __init__(self):
        self.unk_value = 0
        self.pad_value = 1

    def get_ner2ent(self, raw_ner2ent, lower=True):
        ner2ent = [json.loads(item) for item in raw_ner2ent]
        if lower:
            ner2ent = [{k.lower(): v for k, v in sent.items()} for sent in
                       ner2ent]
        return ner2ent

    def get_tgt(self, raw_tgt, raw_ners, tgt_no_ner_vocab_size):
        tgt = raw_tgt[0]
        ners = [sent_ners[sent_ners != self.pad_value] for sent_ners in
                raw_ners]
        ner2ix = [{ner + tgt_no_ner_vocab_size - 2: ix for ix, ner in
                   enumerate(sent)} for sent in ners]
        for sent_ix, (sent_tgt, sent_ner2ix) in enumerate(zip(tgt, ner2ix)):
            # sent_tgt, batch.show_inp[sent_ix].split('<ENT_TGT_SEP> ')[-1].split(' <TGT_TXT_SEP>')[0]

            for word_ix, word in enumerate(sent_tgt):

                if word >= tgt_no_ner_vocab_size:
                    if word.item() in sent_ner2ix:
                        tgt[sent_ix][word_ix] = tgt_no_ner_vocab_size + \
                                                sent_ner2ix[word.item()]
                    else:
                        tgt[sent_ix][word_ix] = \
                            0  # in webnlg, sometimes the template contains
                        # AGENT-1 but AGENT-1 is not in the ner2ent dictionary,
                        # so we assign '0' UNK to this word.
        return tgt

    def get_ent_lens(self, raw_ent_lens):
        return [json.loads(doc_ent_lens) for doc_ent_lens in raw_ent_lens]

    def get_rel(self, batch_adj, ent_lens, batch_rels, use_dgl=False):
        batch_adj = [json.loads(adj_str) for adj_str in batch_adj]
        batch_rels = [rels[rels != self.pad_value] for rels in batch_rels]

        return try_utils.get_rel(batch_adj, ent_lens, batch_rels, use_dgl=use_dgl)

    def get_sents_n_rels(self, tgt_non_templ, raw_ent_phrases, ordered_ents,
                         ordered_rels, ENTITY_vocab, TGT_NON_TEMPL_vocab):
        sents = []
        rels = []
        combinations = []
        ent_phrases = [[tensor2list(ent_phrase) for ent_phrase in doc]
                       for doc in raw_ent_phrases]
        for sent_ix, sent_phrases in enumerate(ent_phrases):
            for phrase_ix, phrase in enumerate(sent_phrases):
                for word_ix, word in enumerate(phrase):
                    w = ENTITY_vocab.itos[word]
                    w_ix = TGT_NON_TEMPL_vocab.stoi[w]
                    ent_phrases[sent_ix][phrase_ix][word_ix] = w_ix
        for sent, sent_ents, sent_ord_ents, sent_ord_rels in \
                zip(tgt_non_templ, ent_phrases, ordered_ents, ordered_rels):
            sent = tensor2list(sent[sent != self.pad_value])

            sent_ord_rels = sent_ord_rels[sent_ord_rels != self.pad_value]

            sent_ord_ents = json.loads(sent_ord_ents)

            ents_combi = list(itertools.permutations(range(len(sent_ents)), 2))
            sent_ord_ents = list(zip(sent_ord_ents[::2], sent_ord_ents[1::2]))
            combination = sent_ord_ents + \
                          list(set(ents_combi) - set(sent_ord_ents))

            sent_variation = [
                sent_ents[ix0] + sent[-1:] +
                sent_ents[ix1] + sent[:-1]
                for ix0, ix1 in combination]
            sent_variation = [tgt_non_templ.new_tensor(sent_v)
                              for sent_v in sent_variation]
            # import pdb;
            # pdb.set_trace()  # TODO: check combination correctness

            sent_variation = try_utils.list2padseq(sent_variation, tgt_non_templ,
                                               padding_value=self.pad_value)

            rel = sent_ord_rels.new(len(sent_variation)).fill_(self.unk_value)
            if len(rel) < len(sent_ord_rels): import pdb;pdb.set_trace()
            rel[:len(sent_ord_rels)] = sent_ord_rels
            rels += [rel]
            sents += [sent_variation]
            combinations += [combination]
        return sents, rels, combinations, ent_phrases

    def get_ent_phrases(self, ent_lens, ents): \
            # [batch,] for ent start idx
        ent_st = [[sum(doc_ent_lens[:i])
                   for i in range(len(doc_ent_lens))]
                  for doc_ent_lens in ent_lens
                  ]

        # word_enc for packed_doc [batch, n_word, dim]
        ent_phrases = [[doc_ents[st:st + lens]
                        for st, lens in zip(doc_ent_st, doc_ent_lens)]
                       for doc_ent_st, doc_ent_lens, doc_ents in
                       zip(ent_st, ent_lens, ents)
                       ]
        return ent_phrases

    def get_ent(self, ent_lens, raw_ents, ent_phrases):

        ent = [None, None, None]
        ent[2] = raw_ents.new_tensor([len(doc_ent_lens)
                                      for doc_ent_lens in ent_lens])
        ent[1] = raw_ents.new_tensor(
            [l for doc_ent_lens in ent_lens for l in doc_ent_lens])

        ent_phrases = [e for d in ent_phrases for e in d]
        ent[0] = try_utils.list2padseq(ent_phrases, ent_phrases[0],
                                   padding_value=self.pad_value)

        return ent


class BatchWrapper:
    def __init__(self, dataloader, tgt_no_ner_vocab_size, use_dgl=False,
                 timer=False):
        self.dataloader = dataloader
        self.tgt_no_ner_vocab_size = tgt_no_ner_vocab_size
        self.wrapper = Wrapper()
        self.use_dgl = use_dgl
        self.iter_ix = 0
        self.timer = defaultdict(int) if timer else None

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if self.timer is None:
            return self.iter()
        else:
            return self.iter_with_timing()

    def iter(self):
        wrapper = self.wrapper
        NER_field = self.dataloader.dataset.fields["ners"]
        ENTITY_vocab = self.dataloader.dataset.fields["ents"].vocab
        TGT_NON_TEMPL_vocab = self.dataloader.dataset.fields[
            "tgt_non_templ"].vocab

        for batch in self.dataloader:
            ner2ent = wrapper.get_ner2ent(batch.ner2ent,
                                          lower=NER_field.lower)
            tgt = wrapper.get_tgt(batch.tgt, batch.ners,
                                  self.tgt_no_ner_vocab_size)

            ent_lens = wrapper.get_ent_lens(batch.ent_lens)
            rel = wrapper.get_rel(batch.adj, ent_lens, batch.rels, self.use_dgl)
            ent_phrases = wrapper.get_ent_phrases(ent_lens, batch.ents)
            ent = wrapper.get_ent(ent_lens, batch.ents, ent_phrases)

            sents, ordered_rels, combinations, t2g_ent_phrases = \
                wrapper.get_sents_n_rels(
                    batch.tgt_non_templ, ent_phrases, batch.ordered_ents,
                    batch.ordered_rels, ENTITY_vocab, TGT_NON_TEMPL_vocab)

            batch_dic = {
                'public_graph': {
                    'ners': batch.ners,  # just for eval

                    'tgt': tgt,  # pad is 1
                    'show_inp': batch.show_inp,  # just for eval

                    'ner2ent': ner2ent,  # just for eval
                    'ent': ent,
                },
                'public_text': {
                    'ordered_rels': ordered_rels,
                    'combinations': combinations,
                    't2g_ent_phrases': t2g_ent_phrases,
                },
                'graph': {
                    'rel': rel,
                },
                'text': {
                    'sents': sents,
                }

                # 'adj_seq': batch.adj_seq[0].to(self.device),
            }

            s = Struct(**batch_dic)
            yield s

    def iter_with_timing(self):
        wrapper = self.wrapper
        NER_field = self.dataloader.dataset.fields["ners"]
        ENTITY_vocab = self.dataloader.dataset.fields["ents"].vocab
        TGT_NON_TEMPL_vocab = self.dataloader.dataset.fields[
            "tgt_non_templ"].vocab

        for batch in self.dataloader:
            if self.iter_ix in [10, 100, 1000, 5000]: show_var(['self.timer'])
            self.iter_ix += 1

            torch.cuda.synchronize();
            self.timer['ner2ent'] -= time()
            ner2ent = wrapper.get_ner2ent(batch.ner2ent,
                                          lower=NER_field.lower)
            torch.cuda.synchronize();
            self.timer['ner2ent'] += time()

            torch.cuda.synchronize();
            self.timer['uniq_ners'] -= time()
            uniq_ners = wrapper.get_uniq_ners(batch.ners)
            torch.cuda.synchronize();
            self.timer['uniq_ners'] += time()

            torch.cuda.synchronize();
            self.timer['tgt'] -= time()
            tgt = wrapper.get_tgt(batch.tgt, uniq_ners,
                                  self.tgt_no_ner_vocab_size)
            torch.cuda.synchronize();
            self.timer['tgt'] += time()

            torch.cuda.synchronize();
            self.timer['ent_lens'] -= time()
            ent_lens = wrapper.get_ent_lens(batch.ent_lens)
            torch.cuda.synchronize();
            self.timer['ent_lens'] += time()

            torch.cuda.synchronize();
            self.timer['rel'] -= time()
            rel = wrapper.get_rel(batch.adj, ent_lens, batch.rels, self.use_dgl)
            torch.cuda.synchronize();
            self.timer['rel'] += time()

            torch.cuda.synchronize();
            self.timer['ent_phrases'] -= time()
            ent_phrases = wrapper.get_ent_phrases(ent_lens, batch.ents)
            torch.cuda.synchronize();
            self.timer['ent_phrases'] += time()

            torch.cuda.synchronize();
            self.timer['ent'] -= time()
            ent = wrapper.get_ent(ent_lens, batch.ents, ent_phrases)
            torch.cuda.synchronize();
            self.timer['ent'] += time()

            torch.cuda.synchronize();
            self.timer['sents'] -= time()
            sents, ordered_rels, combinations = wrapper.get_sents_n_rels(
                batch.tgt_non_templ, ent_phrases, batch.ordered_ents,
                batch.ordered_rels, ENTITY_vocab, TGT_NON_TEMPL_vocab)
            torch.cuda.synchronize();
            self.timer['sents'] += time()

            batch_dic = {
                # 'ners': batch.ners,  # just for eval
                # 'uniq_ners': uniq_ners,  # just for eval
                #
                # 'out': tgt,
                # 'tgt': tgt,  # pad is 1
                # 'show_inp': batch.show_inp,  # just for eval
                #
                # 'ner2ent': ner2ent,  # just for eval
                # 'rel': rel,
                # 'ent': ent,
                #
                # 'ordered_rels': ordered_rels,
                # 'sents': sents,
                # 'combinations': combinations,

                # 'adj_seq': batch.adj_seq[0].to(self.device),
            }

            s = Struct(**batch_dic)
            yield s
        show_var(['self.timer'])


def main():
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser('dataloader args')
    parser.add_argument('-dataset', choices=['webnlg', 'agenda'], type=str,
                        default='agenda', help='name of the dataset to use')
    args = parser.parse_args()
    data_dir = os.path.join('data/', args.dataset)

    dataset = Dataset(proc_id=0, data_dir=data_dir)
    train_dl, valid_dl, test_dl = dataset.get_dataloader(
        0, 1, batch_size=4, device=torch.device('cuda:0'), use_dgl=True,
        timer=False)
    for epoch in range(3):
        for batch in tqdm(train_dl):
            continue
            import pdb
            pdb.set_trace()
            print(batch)


if __name__ == "__main__":
    main()
