"""A count-preserving control paired to the original larger-model source stream.

Only positions 1..n_ctx (zero-based) are shuffled. Each sequence receives its
own indexed RNG; source generation and model initialization use no shuffle RNG.
The first token and complete per-sequence counts remain exactly unchanged.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

import train_shared_base_scaling as ordered

CONTROL = 'independent-tail-position-shuffle'
SHUFFLE_CONTRACT = {
    'version': 'shared-base-tail-shuffle-v1',
    'control': CONTROL,
    'source': ordered.SOURCE_CONTRACT,
    'original_batch': 'train_shared_base_scaling.draw_batch(seed, zero_based_update, batch, n_ctx)',
    'shuffle_rng': 'numpy-PCG64-SeedSequence[0x53485346, stream, seed_lo, seed_hi, update_lo, update_hi, row]',
    'train_stream': 0,
    'validation_stream': 1,
    'first_token': 'Retained unchanged.',
    'positions': 'Independent uniform permutation of positions 2 through n_ctx+1 for each sequence.',
    'targets': 'The shuffled sequence shifted by one token; no original-order target is retained.',
    'preserves': 'Exact token counts of each original finite training sequence.',
    'interpretation': 'Finite-length exchangeable tail; not independent draws from the stationary law.'
}


def shuffle_rng(seed, update, row, *, stream=0):
    for name,value in [('seed',seed),('update',update),('row',row),('stream',stream)]:
        if type(value) is not int or value < 0 or value >= 2**64:
            raise ValueError(f'{name} must be a nonnegative uint64 integer')
    if row >= 2**32 or stream >= 2**32:
        raise ValueError('row and stream must fit uint32')
    return np.random.default_rng(np.random.SeedSequence([
        0x53485346, stream, seed & 0xffffffff, seed >> 32,
        update & 0xffffffff, update >> 32, row]))


def tail_shuffle(tokens, seed, update, *, stream=0, return_indices=False):
    """Return a shuffled copy; neither tokens nor any global RNG are changed."""
    if tokens.ndim != 2 or tokens.shape[1] < 2 or tokens.dtype != np.int64:
        raise ValueError('expected a two-dimensional int64 token array of length >=2')
    indices=np.empty(tokens.shape,dtype=np.int64)
    indices[:,0]=0
    for row in range(len(tokens)):
        indices[row,1:]=1+shuffle_rng(seed,update,row,stream=stream).permutation(tokens.shape[1]-1)
    result=np.take_along_axis(tokens,indices,axis=1)
    return (result,indices) if return_indices else result


def draw_batch(seed, index, batch, n_ctx):
    return tail_shuffle(ordered.draw_batch(seed,index,batch,n_ctx),seed,index)


class IndexedShuffledSharedBase(torch.utils.data.Dataset):
    def __init__(self, seed, batch, n_ctx, start, stop):
        self.seed,self.batch,self.n_ctx,self.start,self.stop=seed,batch,n_ctx,start,stop
    def __len__(self):
        return self.stop-self.start
    def __getitem__(self,index):
        if not 0<=index<len(self):
            raise IndexError(index)
        return torch.from_numpy(draw_batch(self.seed,self.start+index,self.batch,self.n_ctx))


def draw_validation(cfg):
    """Shuffle the original held-out monitoring sequences with a separate stream."""
    original=ordered.draw_validation(cfg)
    return {name:torch.from_numpy(tail_shuffle(tokens.numpy(),0,index,stream=1))
            for index,(name,tokens) in enumerate(original.items())}


def array_hash(tokens):
    return hashlib.sha256(np.ascontiguousarray(tokens,dtype='<i8').tobytes()).hexdigest()


def pairing_probe(seed, index, batch=64, n_ctx=1024):
    """Verify a full-size batch against an independently reconstructed legacy call."""
    from source import Source
    reference_rng=np.random.default_rng(np.random.SeedSequence([
        0x53424C53,seed & 0xffffffff,seed>>32,index & 0xffffffff,index>>32]))
    reference=Source(64,n_ctx+1,alpha=(.5,1.5),ls=(0,8),permute=True).batch(batch,reference_rng)[0]
    original=ordered.draw_batch(seed,index,batch,n_ctx)
    if not np.array_equal(original,reference):
        raise ValueError('Original indexed sampler does not match independent source reconstruction')
    shuffled,indices=tail_shuffle(original,seed,index,return_indices=True)
    if not np.array_equal(shuffled[:,0],original[:,0]) or not np.array_equal(
            np.sort(shuffled,axis=1),np.sort(original,axis=1)):
        raise ValueError('Shuffle changed first tokens or per-sequence counts')
    return dict(seed=seed,update_index=index,batch=batch,n_ctx=n_ctx,
                original_tokens_sha256=array_hash(original),shuffled_tokens_sha256=array_hash(shuffled),
                position_permutations_sha256=array_hash(indices),
                distinct_tail_permutations=len(np.unique(indices[:,1:],axis=0)))
