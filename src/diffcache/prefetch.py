"""Prefetching for the 2-GPU live baseline -- ported essentially unchanged
from `training_from_jupyter/live_prefetch.py`, which is itself already
pipeline-agnostic (operates on generic tensor dicts/lists via
`map_tensors`/`iter_tensors`, nothing Qwen-specific) and was copied verbatim,
by design, across all 4 of the paper's benchmark projects.

A producer thread encodes sample i+1 on the encoder GPU while the main
thread trains sample i on the DiT GPU, instead of alternating between them
the way the plain live baseline (`phases/live.py`) does -- each GPU has its
own work, so there's no SM contention; what this measures is how much of the
encoder cost can be hidden behind the DiT step.

Stream details (the timing bug fixed 2026-09-23 across the other three
pipelines came from exactly this: synchronizing the wrong device):
  - encode runs on `device_encode`'s own stream (`enc_stream`);
  - the device_encode -> device_train copy runs on `device_train`'s own
    stream (`copy_stream`), outside `enc_stream`'s context, so the copy
    isn't ordered behind whatever training step is currently using
    `device_train`'s default stream;
  - the producer waits on its own events on the host; the main thread only
    waits on the "copied" event before training, and records the tensors on
    its own stream (`record_stream`) so the allocator can't reuse them too
    early.
`torch.cuda.synchronize()` is never called without a device inside the loop.
"""
from __future__ import annotations

import queue
import threading
import time

import torch

_STOP = object()


def map_tensors(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_tensors(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_tensors(v, fn) for v in obj)
    return obj


def iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_tensors(v)


class EncoderPrefetcher:
    """Iterates `schedule` (a list of `(epoch, indices)`), yielding per step
    `(epoch, batch already on device_train, encode_ms, copy_ms, wait_ms)`.

    `load_fn(indices)` -> CPU-side samples (image decoding, outside GPU
    timing); `encode_fn(samples)` -> a batch on `device_encode`. `depth` is
    how many already-encoded batches may sit queued ahead of the consumer.
    """

    def __init__(self, schedule, load_fn, encode_fn, device_encode, device_train, depth=1):
        self.schedule = schedule
        self.load_fn = load_fn
        self.encode_fn = encode_fn
        self.device_encode = torch.device(device_encode)
        self.device_train = torch.device(device_train)
        self.q = queue.Queue(maxsize=depth)
        self.errors = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            torch.cuda.set_device(self.device_encode)
            # Each stream has its own pool in PyTorch's allocator: encoding on
            # enc_stream can't reuse blocks the model-loading step cached on
            # the default stream. On a tight encoder GPU (e.g. FLUX's live
            # baseline at 15.1/16 GiB) the stream couldn't even be created
            # without this (OOM, caught 2026-09-23).
            torch.cuda.empty_cache()
            enc_stream = torch.cuda.Stream(device=self.device_encode)
            copy_stream = torch.cuda.Stream(device=self.device_train)
            with torch.no_grad():
                for epoch, indices in self.schedule:
                    samples = self.load_fn(indices)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(enc_stream):
                        e0.record(enc_stream)
                        batch = self.encode_fn(samples)
                        e1.record(enc_stream)
                    e1.synchronize()
                    c0 = torch.cuda.Event(enable_timing=True)
                    c1 = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(copy_stream):
                        c0.record(copy_stream)
                        moved = map_tensors(batch, lambda t: t.to(self.device_train, non_blocking=True))
                        c1.record(copy_stream)
                    c1.synchronize()
                    del batch
                    self.q.put((epoch, moved, c1, e0.elapsed_time(e1), c0.elapsed_time(c1)))
        except BaseException as exc:  # re-raised on the main thread
            self.errors.append(exc)
        finally:
            self.q.put(_STOP)

    def __iter__(self):
        train_stream = torch.cuda.current_stream(self.device_train)
        self.thread.start()
        while True:
            t0 = time.perf_counter()
            item = self.q.get()
            wait_ms = (time.perf_counter() - t0) * 1000.0
            if item is _STOP:
                break
            epoch, batch, ev_ready, encode_ms, copy_ms = item
            train_stream.wait_event(ev_ready)
            for t in iter_tensors(batch):
                t.record_stream(train_stream)
            yield epoch, batch, encode_ms, copy_ms, wait_ms
        self.thread.join()
        if self.errors:
            raise self.errors[0]


class StepRecorder:
    """Accumulates per epoch: real wall-clock interval between step ends,
    GPU training time, encode time, copy time, queue wait, and loss."""

    KEYS = ("step_wall", "train_gpu", "encode_gpu", "copy_gpu", "queue_wait", "loss")

    def __init__(self, num_epochs):
        self.per_epoch = {e: {k: [] for k in self.KEYS} for e in range(num_epochs)}
        self.t_prev = time.perf_counter()

    def start(self):
        self.t_prev = time.perf_counter()

    def add(self, epoch, train_ms, encode_ms, copy_ms, wait_ms, loss_value):
        now = time.perf_counter()
        rec = self.per_epoch[epoch]
        rec["step_wall"].append((now - self.t_prev) * 1000.0)
        self.t_prev = now
        rec["train_gpu"].append(train_ms)
        rec["encode_gpu"].append(encode_ms)
        rec["copy_gpu"].append(copy_ms)
        rec["queue_wait"].append(wait_ms)
        rec["loss"].append(loss_value)
        return rec

    def report(self, epoch_stats):
        def mean_excl_first(x):
            x = x[1:] if len(x) > 1 else x
            return sum(x) / len(x)

        epochs, flat = [], {k: [] for k in self.KEYS}
        for e, rec in self.per_epoch.items():
            st = epoch_stats(rec["step_wall"], rec["loss"], e)
            for k in ("train_gpu", "encode_gpu", "copy_gpu", "queue_wait"):
                st[f"epoch_mean_{k}_ms"] = sum(rec[k]) / len(rec[k])
            epochs.append(st)
            for k in self.KEYS:
                flat[k] += rec[k]
        timing = {f"{k}_all": flat[k] for k in self.KEYS if k != "loss"}
        timing.update({f"{k}_mean_excl_first": mean_excl_first(flat[k]) for k in self.KEYS if k != "loss"})
        timing["step_mean_excl_first"] = timing["step_wall_mean_excl_first"]
        timing["total_train_wall_clock_s"] = sum(flat["step_wall"]) / 1000.0
        return epochs, timing, flat["loss"]
