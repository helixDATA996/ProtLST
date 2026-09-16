from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ARMS=("esm","vae_z3","z3_transformer","trajectory_frozen","joint_function","joint_multitask")
SEEDS=(17,31,47)

def run(command,env):
    print(" ".join(command),flush=True); subprocess.run(command,check=True,env=env)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--shard-index",required=True); ap.add_argument("--train-text-cache",required=True); ap.add_argument("--validation-text-cache",required=True); ap.add_argument("--test-text-cache",default=""); ap.add_argument("--vae",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS)); ap.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS)); ap.add_argument("--include-test",action="store_true"); ap.add_argument("--test-only",action="store_true",help="with --evaluate-only, evaluate only the locked test split"); ap.add_argument("--evaluate-only",action="store_true",help="evaluate existing arm.seed checkpoint files without retraining"); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--device",default="cuda:0"); a=ap.parse_args()
    if a.test_only and not a.evaluate_only: ap.error("--test-only requires --evaluate-only")
    if (a.include_test or a.test_only) and not a.test_text_cache: ap.error("test evaluation requires --test-text-cache")
    root=Path(__file__).resolve().parents[2]; python=sys.executable; out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env["PYTHONPATH"]=f"{root.parent}:{env.get('PYTHONPATH','')}"
    seeds=(a.seeds[0],) if a.smoke else tuple(a.seeds); epochs=1 if a.smoke else 3; model_dim=64 if a.smoke else None; layers=1 if a.smoke else 2; batch_size=4 if a.smoke else 16; bucket_size=128 if a.smoke else 256; bootstrap=10 if a.smoke else 500
    results=[]
    for arm in a.arms:
      for seed in seeds:
        checkpoint=out/f"{arm}.seed{seed}.pt"
        if not a.evaluate_only:
          command=[python,str(root/"scripts/vae_trajectory/train_attribution_experiment.py"),"--manifest",a.manifest,"--shard-index",a.shard_index,"--text-cache",a.train_text_cache,"--text-cache",a.validation_text_cache,"--vae",a.vae,"--arm",arm,"--out",str(checkpoint),"--epochs",str(epochs),"--batch-size",str(batch_size),"--bucket-size",str(bucket_size),"--layers",str(layers),"--heads","8","--seed",str(seed),"--device",a.device]
          if model_dim is not None: command += ["--model-dim",str(model_dim)]
          run(command,env)
        elif not checkpoint.exists():
          raise FileNotFoundError(f"missing checkpoint for --evaluate-only: {checkpoint}")
        splits = [("test",a.test_text_cache)] if a.test_only else [("validation",a.validation_text_cache)]+([( "test",a.test_text_cache)] if a.include_test else [])
        for split,cache in splits:
          result=out/f"{arm}.seed{seed}.{split}.json"; run([python,str(root/"scripts/vae_trajectory/evaluate_attribution_experiment.py"),"--checkpoint",str(checkpoint),"--manifest",a.manifest,"--shard-index",a.shard_index,"--text-cache",cache,"--split",split,"--batch-size",str(batch_size),"--bootstrap",str(bootstrap),"--device",a.device,"--out",str(result)],env); results.append(result)
    summary_results=sorted(out.glob("*.validation.json"))
    if a.include_test or a.test_only: summary_results += sorted(out.glob("*.test.json"))
    run([python,str(root/"scripts/vae_trajectory/summarize_attribution_experiment.py"),*[x for result in summary_results for x in ("--input",str(result))],"--out",str(out/"summary.json")],env)
if __name__=="__main__":main()
