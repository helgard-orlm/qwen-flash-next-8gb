"""Capture existing one-token DeltaNet arithmetic without fusing reductions.

36 independent graph records; weights stay shared. CPU routing, disk, MoE,
attention, PLE and prefill remain outside capture. Stable recurrent buffers
are copied back in the graph; ordinary snapshots/restores use their usual API.
"""
import torch


class DeltaGraphs:
    def __init__(self, engine):
        self.original = engine.deltanet
        self.original_forward = engine.forward
        self.enabled = False
        self.records = {}
        self.calls = 0
        # Every layer is replayed in capture order on the same stream. Share
        # scratch storage; 36 private pools would consume ~0.8 GiB needlessly.
        self.pool = torch.cuda.graph_pool_handle()
        engine.deltanet = self.call
        engine.forward = self.forward

    def forward(self, ids, *args, **kwargs):
        # Reset/snapshot restore can detach live state from the graph buffers.
        # A later long prefill then holds both sets (~108 MiB each), exhausting
        # this 8 GiB card. Drop captures before allocating prompt activations;
        # engine state still owns any live recurrent tensors. Decode recaptures.
        if len(ids) >= 1024 and self.records:
            torch.cuda.synchronize()
            self.records.clear()
            # A pool token cannot be reused after all its graph owners were
            # destroyed and empty_cache released it (allocator use_count=0).
            self.pool = torch.cuda.graph_pool_handle()
            torch.cuda.empty_cache()
        return self.original_forward(ids, *args, **kwargs)

    @torch.no_grad()
    def call(self, layer, state, x):
        if (not self.enabled or x.shape[0] != 1 or state.get('S') is None
                or state.get('conv') is None):
            return self.original(layer, state, x)
        key = id(layer)
        if key not in self.records:
            bx = torch.empty_like(x)
            bs = torch.empty_like(state['S'])
            bc = torch.empty_like(state['conv'])
            bx.copy_(x); bs.copy_(state['S']); bc.copy_(state['conv'])
            # Resolve Triton kernels and library setup outside capture. Use a
            # disposable state dictionary so production state isn't advanced.
            for _ in range(3):
                self.original(layer, {'S':bs,'conv':bc}, bx)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Claude #324: thread_local — потоки чтения экспертов (ev.synchronize) иначе ломают запись (cudaErrorStreamCaptureInvalidated, 02.10 20:56 после промта 3520)
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                local = {'S':bs,'conv':bc}
                output = self.original(layer, local, bx)
                bs.copy_(local['S'])
                bc.copy_(local['conv'])
            self.records[key] = (graph, bx, bs, bc, output)
        graph, bx, bs, bc, output = self.records[key]
        bx.copy_(x)
        if state['S'].data_ptr() != bs.data_ptr():
            bs.copy_(state['S'])
        if state['conv'].data_ptr() != bc.data_ptr():
            bc.copy_(state['conv'])
        graph.replay()
        state['S'], state['conv'] = bs, bc
        self.calls += 1
        return output
