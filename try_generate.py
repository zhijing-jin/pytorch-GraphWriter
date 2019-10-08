import sys
from random import shuffle
import os
from math import exp
import torch
from torch import nn
from torch.nn import functional as F
from lastDataset import dataset
from pargs import pargs, dynArgs
from models.newmodel import model


def tgtreverse(tgts,entlist,order):
  entlist = entlist[0]
  order = [int(x) for x in order[0].split(" ")]
  tgts = tgts.split(" ")
  k = 0
  for i,x in enumerate(tgts):
    if x[0] == "<" and x[-1]=='>':
      tgts[i] = entlist[order[k]]
      k+=1
  return " ".join(tgts)


def train(m, o, ds, args):
    print("Training", end="\t")
    loss = 0
    ex = 0
    trainorder = [('1', ds.t1_iter), ('2', ds.t2_iter), ('3', ds.t3_iter)]
    # trainorder = reversed(trainorder)
    shuffle(trainorder)
    for spl, train_iter in trainorder:
        print(spl)
        for count, b in enumerate(train_iter):
            if count % 100 == 99:
                print(ex, "of like 40k -- current avg loss ", (loss / ex))
            b = ds.fixBatch(b)
            p, z, planlogits = m(b)
            p = p[:, :-1, :].contiguous()

            tgt = b.tgt[:, 1:].contiguous().view(-1).to(args.device)
            l = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt,
                           ignore_index=1)
            # copy coverage (each elt at least once)
            if args.cl:
                z = z.max(1)[0]
                cl = nn.functional.mse_loss(z, torch.ones_like(z))
                l = l + args.cl * cl
            if args.plan:
                pl = nn.functional.cross_entropy(
                    planlogits.view(-1, planlogits.size(2)),
                    b.sordertgt[0].view(-1), ignore_index=1)
                l = l + args.plweight * pl

            l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), args.clip)
            loss += l.item() * len(b.tgt)
            o.step()
            o.zero_grad()
            ex += len(b.tgt)
    loss = loss / ex
    print("AVG TRAIN LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))

def test(args,ds,m,epoch='cmdline'):
  args.vbsz = 1
  model = args.save.split("/")[-1]
  ofn = "../outputs/"+model+".beam_predictions"
  m.eval()
  k = 0
  data = ds.mktestset(args)
  ofn = "../outputs/"+model+".inputs.beam_predictions."+epoch
  pf = open(ofn,'w')
  preds = []
  golds = []
  for b in data:
    #if k == 10: break
    print(k,len(data))
    b = ds.fixBatch(b)
    '''
    p,z = m(b)
    p = p[0].max(1)[1]
    gen = ds.reverse(p,b.rawent)
    '''
    gen = m.beam_generate(b,beamsz=4,k=6)
    gen.sort()
    gen = ds.reverse(gen.done[0].words,b.rawent)
    k+=1
    gold = ds.reverse(b.tgt[0][1:],b.rawent)
    print(gold)
    print(gen)
    print()
    preds.append(gen.lower())
    golds.append(gold.lower())
    #tf.write(ent+'\n')
    pf.write(gen.lower()+'\n')
  m.train()
  return preds,golds


def evaluate(m, ds, args):
    print("Evaluating", end="\t")
    m.eval()
    loss = 0
    ex = 0
    for b in ds.val_iter:
        b = ds.fixBatch(b)
        p, z, planlogits = m(b)
        p = p[:, :-1, :]
        tgt = b.tgt[:, 1:].contiguous().view(-1).to(args.device)
        l = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
        if ex == 0:
            g = p[0].max(1)[1]
            print(ds.reverse(g, b.rawent[0]))
        loss += l.item() * len(b.tgt)
        ex += len(b.tgt)
    loss = loss / ex
    print("VAL LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))
    m.train()
    return loss


def main(args):
    args.eval = True

    ds = dataset(args)
    args = dynArgs(args, ds)
    m = model(args)
    print(args.device)
    m = m.to(args.device)
    '''
    with open(args.save+"/commandLineArgs.txt") as f:
      clargs = f.read().strip().split("\n") 
      argdif =[x for x in sys.argv[1:] if x not in clargs]
      assert(len(argdif)==2); 
      assert([x for x in argdif if x[0]=='-']==['-ckpt'])
    '''
    cpt = torch.load(args.ckpt)
    m.load_state_dict(cpt)
    starte = int(args.ckpt.split("/")[-1].split(".")[0]) + 1
    args.lr = float(args.ckpt.split("-")[-1])
    print('ckpt restored')

    o = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)

    # early stopping based on Val Loss
    lastloss = 1000000

    for e in range(starte, starte + 1):
        print("epoch ", e, "lr", o.param_groups[0]['lr'])
        vloss = evaluate(m, ds, args)
        print('vloss:{:.2f}'.format(vloss))

    args.vbsz = 1
    preds, gold = test(args, ds, m)

if __name__ == "__main__":
    args = pargs()
    main(args)
