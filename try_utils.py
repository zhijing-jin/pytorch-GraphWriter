from __future__ import division, print_function, unicode_literals
import sys
import os.path
import pdb
import json
import torch

ENT0_END = 'ENT0_END'
ENT1_END = 'ENT1_END'

TRIPLE_SEP = ';;\t'
ENT_TGT_SEP = '<ENT_TGT_SEP>'
TGT_TXT_SEP = '<TGT_TXT_SEP>'
REL_INV = '_inv'


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def tensor2list(tensor):
    return tensor.cpu().numpy().tolist()


def list2padseq(ls, longtensor, padding_value=1):
    from torch.nn.utils.rnn import pad_sequence

    order, sorted_ls = zip(
        *sorted(enumerate(ls), key=lambda x: -len(x[1])))
    rev_order, _ = zip(
        *sorted(enumerate(order), key=lambda x: x[1]))
    rev_order = longtensor.new_tensor(rev_order)

    padded = pad_sequence(sorted_ls, batch_first=True,
                          padding_value=padding_value)
    padded = padded[rev_order]
    return padded


def init_weights(m):
    from torch import nn
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)


def fwrite(new_doc, path, mode='w', no_overwrite=False):
    if not path:
        print("[Info] Path does not exist in fwrite():", str(path))
        return
    if no_overwrite and os.path.isfile(path):
        print("[Error] pls choose whether to continue, as file already exists:",
              path)
        import pdb
        pdb.set_trace()
        return
    with open(path, mode) as f:
        f.write(new_doc)


def show_time(what_happens='', cat_server=False, printout=True):
    import datetime

    disp = '⏰ Time: ' + \
           datetime.datetime.now().strftime('%m%d%H%M-%S')
    disp = disp + '\t' + what_happens if what_happens else disp
    if printout:
        print(disp)
    curr_time = datetime.datetime.now().strftime('%m%d%H%M')

    if cat_server:
        hostname = socket.gethostname()
        prefix = "rosetta"
        if hostname.startswith(prefix):
            host_id = hostname[len(prefix):]
            try:
                host_id = int(host_id)
                host_id = "{:02d}".format(host_id)
            except:
                pass
            hostname = prefix[0] + host_id
        else:
            hostname = hostname[0]
        curr_time += hostname
    return curr_time


def show_var(expression,
             joiner='\n', print=print):
    '''
    Prints out the name and value of variables.
    Eg. if a variable with name `num` and value `1`,
    it will print "num: 1\n"
    Parameters
    ----------
    expression: ``List[str]``, required
        A list of varible names string.
    Returns
    ----------
        None
    '''

    import json

    var_output = []

    for var_str in expression:
        frame = sys._getframe(1)
        value = eval(var_str, frame.f_globals, frame.f_locals)

        if ' object at ' in repr(value):
            value = vars(value)
            value = json.dumps(value, indent=2)
            var_output += ['{}: {}'.format(var_str, value)]
        else:
            var_output += ['{}: {}'.format(var_str, repr(value))]

    if joiner != '\n':
        output = "[Info] {}".format(joiner.join(var_output))
    else:
        output = joiner.join(var_output)
    print(output)
    return output


def shell(cmd, working_directory='.', stdout=False, stderr=False):
    import subprocess
    from subprocess import PIPE, Popen

    subp = Popen(cmd, shell=True, stdout=PIPE,
                 stderr=subprocess.STDOUT, cwd=working_directory)
    subp_stdout, subp_stderr = subp.communicate()

    if subp_stdout: subp_stdout = subp_stdout.decode("utf-8")
    if subp_stderr: subp_stderr = subp_stderr.decode("utf-8")

    if stdout and subp_stdout:
        print("[stdout]", subp_stdout, "[end]")
    if stderr and subp_stderr:
        print("[stderr]", subp_stderr, "[end]")

    return subp_stdout, subp_stderr


def set_seed(seed=0, verbose=False):
    import random

    if seed is None:
        seed = int(show_time())
    if verbose: print("[Info] seed set to: {}".format(seed))

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def flatten_list(nested_list):
    from itertools import chain
    return list(chain.from_iterable(nested_list))


def get_rel(batch_adj, ent_lens, batch_rels, use_dgl=False):
    rel = [None, None]
    rel[0] = _make_graphs(batch_adj, ent_lens, batch_rels, use_dgl=use_dgl)
    rel[1] = batch_rels
    return rel


def _make_graphs(batch_adj, batch_ents, batch_rels, use_dgl=False):
    if use_dgl: import dgl

    device = batch_rels[0].device

    def _make_one_graph(adj_list, n_ent, n_rel):
        adjsize = n_ent + n_rel

        cordinates = torch.LongTensor(adj_list).transpose(0, 1)
        if use_dgl:
            g = dgl.DGLGraph()
            g.add_nodes(adjsize)
            g.add_edges(cordinates[0], cordinates[1])
            g.add_edges(torch.arange(n_ent), n_ent)
            g.add_edges(n_ent, torch.arange(n_ent))
            g.add_edges(torch.arange(adjsize), torch.arange(adjsize))
            return g
        else:
            values = torch.ones_like(cordinates[0])
            adj_torch = torch.sparse.LongTensor(
                cordinates, values, torch.Size([adjsize, adjsize])
            ).to_dense()

            # all nodes are connected to ROOT
            adj_torch[:n_ent, n_ent] = 1
            adj_torch[n_ent, :n_ent] = 1
            adj_torch += torch.eye(adjsize, dtype=adj_torch.dtype)
            return adj_torch.to(device)

    adj_torch = []
    for adj, ents, rels in zip(batch_adj, batch_ents, batch_rels):
        adj_torch += [_make_one_graph(adj, len(ents), len(rels))]

    if use_dgl:
        return dgl.batch(adj_torch)
    else:
        return adj_torch
