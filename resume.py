"""Checkpoint / resume helper for preemptible jobs (off unless PHYSDIAL_RESUMABLE=1).

With PHYSDIAL_RESUMABLE=1 a trainer saves model, optimizer, scheduler and RNG state to <out_dir>/<name>.resume.pt periodically, on SIGUSR1,
or shortly before PHYSDIAL_DEADLINE (unix time), then exits with code 99 so a scheduler can requeue the job; the next start resumes."""
import os, signal, sys, time
import torch

class Resumer:
    def __init__(s, out_dir, name, rank=0, world=1, margin_s=600, every_min=30):
        s.on = os.environ.get("PHYSDIAL_RESUMABLE", "0") == "1"
        s.path = os.path.join(out_dir, name + ".resume.pt"); s.rng_path = lambda r: os.path.join(out_dir, f"{name}.resume.rng{r}.pt")
        s.rank, s.world, s.margin, s.every = rank, world, margin_s, every_min * 60
        s.deadline = float(os.environ.get("PHYSDIAL_DEADLINE", "0") or 0) or None
        s.flag, s.last_save, s.t_start = False, time.time(), time.time()
        if s.on and world == 1:
            try: signal.signal(signal.SIGUSR1, s._usr1)
            except ValueError: pass   # not the main thread
        if s.on: print(f"[resume] on: {s.path}; deadline in {(s.deadline - time.time()) / 60:.0f} min" if s.deadline else f"[resume] on: {s.path}; no deadline (USR1 only)", flush=True)

    def _usr1(s, *_): s.flag = True; print("[resume] USR1 received", flush=True)

    def load(s, model, opt=None, sch=None, gens=()):
        """Returns (start_step, extras) — (0, {}) when off or nothing saved."""
        if not (s.on and os.path.exists(s.path)): return 0, {}
        ck = torch.load(s.path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if opt is not None and ck.get("opt") is not None: opt.load_state_dict(ck["opt"])
        if sch is not None and ck.get("sch") is not None: sch.load_state_dict(ck["sch"])
        torch.set_rng_state(ck["cpu_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None: torch.cuda.set_rng_state(ck["cuda_rng"])
        rp = s.rng_path(s.rank)
        if os.path.exists(rp):
            st = torch.load(rp, map_location="cpu", weights_only=False)
            if isinstance(st, dict) and st.get("step") != ck["step"]:
                print(f"[resume] rank {s.rank}: generator state of step {st.get('step')} does not match model step {ck['step']}; generators keep their seeded state", flush=True)
            else:
                for g, gs in zip(gens, st["states"] if isinstance(st, dict) else st): g.set_state(gs)
        print(f"[resume] restored step {ck['step']} (saved {ck.get('saved_at', '?')}, restart {os.environ.get('SLURM_RESTART_COUNT', '?')})", flush=True)
        return ck["step"], ck.get("extra", {})

    def save(s, step, model, opt=None, sch=None, gens=(), extra=None):
        if not s.on: return
        torch.save({"step": step, "states": [g.get_state() for g in gens]}, s.rng_path(s.rank) + ".tmp"); os.replace(s.rng_path(s.rank) + ".tmp", s.rng_path(s.rank))
        if s.rank == 0:
            ck = {"step": step, "model": model.state_dict(), "opt": opt.state_dict() if opt is not None else None,
                  "sch": sch.state_dict() if sch is not None else None, "cpu_rng": torch.get_rng_state(),
                  "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None, "extra": extra or {},
                  "saved_at": time.strftime("%F %T")}
            torch.save(ck, s.path + ".tmp"); os.replace(s.path + ".tmp", s.path)
        s.last_save = time.time(); print(f"[resume] saved step {step}", flush=True)

    def _agree(s, local):
        if s.world > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([1.0 if local else 0.0], device="cuda"); torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX); return bool(t.item() > 0)
        return local

    def should_stop(s, step, check_every=20, need_s=None):
        """Call once per iteration on every rank (collective every `check_every` steps). need_s overrides the margin."""
        if not s.on or step % check_every: return False
        m = s.margin if need_s is None else need_s
        return s._agree(s.flag or (s.deadline is not None and time.time() > s.deadline - m))

    def periodic_now(s, step, check_every=200):
        """Collective: True on every rank when the periodic save is due (checked every `check_every` steps)."""
        if not s.on or step % check_every: return False
        return s._agree(time.time() - s.last_save > s.every)

    def exit_for_requeue(s):
        print("[resume] exiting with 99 for requeue", flush=True); sys.stdout.flush()
        fl = os.environ.get("PHYSDIAL_REQUEUE_FLAG")   # torchrun turns a worker's exit 99 into a generic failure, so also leave a flag file
        if fl and s.rank == 0: open(fl, "w").write(time.strftime("%F %T") + "\n")
        if s.world > 1 and torch.distributed.is_initialized():
            try: torch.distributed.barrier(); torch.distributed.destroy_process_group()
            except Exception: pass
        sys.exit(99)

    def finish(s):
        if not s.on: return
        for p in [s.path] + [s.rng_path(r) for r in range(s.world)]:
            try: os.remove(p)
            except FileNotFoundError: pass
