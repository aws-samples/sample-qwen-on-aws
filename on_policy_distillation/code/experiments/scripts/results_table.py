"""Print every run's per-checkpoint table (CSV) from the eval dirs on /fsx (run inside a pod that mounts it).

Each run is compared with its own student's baseline (base accuracy, paired z, gap closed).
"""
import json, math, os, csv, sys
L=lambda p:{r["id"]:r for r in map(json.loads,open(p))}
RUNS=[  # name, student, eval dir, steps evaluated, lr, total steps, baseline dir (before + teacher)
    ("run13","Qwen3.5-0.8B","/fsx/opd/eval-run13",[25,50,75,100],"1e-5",100,"/fsx/opd/baseline-gsm8k"),
    ("run14","Qwen3.5-0.8B","/fsx/opd/eval-run14",[25,50,75,100,125,150,175,200],"1e-5",200,"/fsx/opd/baseline-gsm8k"),
    ("run15","Qwen3.5-0.8B","/fsx/opd/eval-run15",[25,50,75,100],"2e-5",100,"/fsx/opd/baseline-gsm8k"),
    ("run16","Qwen3.5-2B","/fsx/opd/eval-run16",[25,50],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b"),
    ("run17","Qwen3.5-2B (LoRA r=128)","/fsx/opd/eval-run17",[25,50],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b"),
    ("run18","Qwen3.5-2B (r=128, T=1.0, 8/prompt)","/fsx/opd/eval-run18",[25,50],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b"),
    ("run19","Qwen3.5-2B (r=128, + top-20 KL)","/fsx/opd/eval-run19",[25,50],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b"),
    ("run20","Qwen3.5-2B (r=128, native zero-shot boxed)","/fsx/opd/eval-run20",[25,50],"2e-5",50,"/fsx/opd/baseline-gsm8k-native-2b"),
    ("run22","Qwen3.5-2B (r=128), AWS DLC images","/fsx/opd/eval-run22",[25,50],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b-dlc"),
    # run17's step 25 again, after phase decoded with an 8-vote majority (base + teacher greedy)
    ("run17-maj8","Qwen3.5-2B (LoRA r=128), after = maj@8","/fsx/opd/eval-maj8-run17",[25],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b"),
    ("run22-maj8","Qwen3.5-2B (r=128), AWS DLC images, after = maj@8","/fsx/opd/eval-maj8-run22",[50],"2e-5",50,"/fsx/opd/baseline-gsm8k-2b-dlc"),
]
w=csv.writer(sys.stdout)
w.writerow(["run","student","lr","total_steps","step","base_accuracy","accuracy","delta_vs_base_pp","fixed","broken",
            "z_paired","gap_closed","answer_marker","capped","mean_tokens"])
for name,student,d,steps,lr,total,base in RUNS:
    B=L(f"{base}/samples_before.jsonl"); T=L(f"{base}/samples_teacher.jsonl")
    bb=sum(v["correct"] for v in B.values())/len(B); tt=sum(v["correct"] for v in T.values())/len(T)
    for s in steps:
        p=f"{d}/step{s}/samples_after.jsonl"
        if not os.path.exists(p): continue
        A=L(p); ids=sorted(A.keys()&B.keys()); n=len(ids); acc=sum(A[i]["correct"] for i in ids)/n
        f=sum(A[i]["correct"] and not B[i]["correct"] for i in ids); b=sum(B[i]["correct"] and not A[i]["correct"] for i in ids)
        toks=[A[i]["output_tokens"] or 0 for i in ids]
        cap=json.load(open(f"{d}/step{s}/summary.json"))["config"]["max_tokens"]   # 1024, or 1536 for run20
        w.writerow([name,student,lr,total,s,f"{bb:.4f}",f"{acc:.4f}",f"{(acc-bb)*100:+.1f}",f,b,f"{(f-b)/math.sqrt(max(1,f+b)):+.2f}",
                    f"{(acc-bb)/(tt-bb):.3f}",f"{sum(A[i]['has_marker'] for i in ids)/n:.3f}",f"{sum(t>=cap-4 for t in toks)/n:.3f}",f"{sum(toks)/n:.0f}"])
