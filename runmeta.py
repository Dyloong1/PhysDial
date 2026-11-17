"""Run provenance, shared by the trainers.

run_meta(paths)  -> dict stored as res["meta"] in every result json: gpu, torch/cuda, code version (PHYSDIAL_GIT_SHA if set, else the git
                    sha of this checkout if available), number of training files and a hash of their basenames (so a partial data copy is
                    visible).
mem_peak()       -> peak memory of the process (CPU, GPU, Slurm job)."""
import hashlib, os, subprocess


def mem_peak():
    """peak memory in GB: this process's and its finished children's
    peak RSS, the GPU peak of this process when it used CUDA, and the Slurm job's unified-cgroup peak / limit; keys missing where unreadable."""
    import resource
    out = {"rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "children_rss_peak_gb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20}
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized(): out["gpu_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30; out["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    except Exception:
        pass
    try:
        d = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().splitlines()[-1].split(":", 2)[2]
        while len(d) > len("/sys/fs/cgroup") and not os.path.basename(d).startswith("job_"): d = os.path.dirname(d)
        if os.path.basename(d).startswith("job_"):
            readf = lambda f: open(os.path.join(d, f)).read().strip(); mx = readf("memory.max")
            out.update(job_peak_gb=int(readf("memory.peak")) / 2**30, job_max_gb=None if mx == "max" else int(mx) / 2**30)
    except (OSError, ValueError, IndexError):
        pass
    return out


def run_meta(train_paths=None):
    m = {}
    try:
        import torch
        m["torch"] = torch.__version__; m["cuda"] = torch.version.cuda
        if torch.cuda.is_available(): m["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if os.environ.get("PHYSDIAL_GIT_SHA"): m["git_sha"] = os.environ["PHYSDIAL_GIT_SHA"]   # may be set by a job launcher
    else:
      try:
        m["git_sha"] = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True, timeout=5).stdout.strip() or None
      except Exception:
        m["git_sha"] = None
    if train_paths is not None:
        names = sorted(os.path.basename(str(p)) for p in train_paths if p)
        m["n_train_files"] = len(names); m["train_files_sha1"] = hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
    return m
