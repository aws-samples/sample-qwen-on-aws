"""Compare eval checkpoints with the base student and teacher on the same GSM8K problems.

Run inside any pod that mounts /fsx:  python3 summarize.py <name>=<eval_dir>/samples_after.jsonl ...
BASE_DIR selects the baseline (default /fsx/opd/baseline-gsm8k-2b: 2B before + 9B teacher; /fsx/opd/baseline-gsm8k for the 0.8B).
"""
import json, math, sys, os
BASE=os.environ.get("BASE_DIR", "/fsx/opd/baseline-gsm8k-2b")
L=lambda p:{r["id"]:r for r in map(json.loads,open(p))}
B=L(f"{BASE}/samples_before.jsonl"); T=L(f"{BASE}/samples_teacher.jsonl")
bb=sum(v["correct"] for v in B.values())/len(B); tt=sum(v["correct"] for v in T.values())/len(T)
print(f"{'base ('+os.path.basename(BASE)+')':16} acc={bb:.3f}"); print(f"{'teacher 9B':16} acc={tt:.3f}")
for arg in sys.argv[1:]:
    name,path=arg.split("=",1)
    if not os.path.exists(path): print(f"{name:16} (not yet)"); continue
    A=L(path); ids=sorted(A.keys()&B.keys()); n=len(ids); acc=sum(A[i]["correct"] for i in ids)/n
    f=sum(A[i]["correct"] and not B[i]["correct"] for i in ids); b=sum(B[i]["correct"] and not A[i]["correct"] for i in ids)
    toks=[A[i]["output_tokens"] or 0 for i in ids]
    print(f"{name:16} acc={acc:.3f} marker={sum(A[i]['has_marker'] for i in ids)/n:.3f} capped={sum(t>=1020 for t in toks)/n:.3f} "
          f"tok={sum(toks)/n:.0f} fixed/broken={f}/{b} z={(f-b)/math.sqrt(max(1,f+b)):+.2f} gap_closed={(acc-bb)/(tt-bb):+.1%}")
