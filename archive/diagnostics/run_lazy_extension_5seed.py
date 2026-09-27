import argparse
import json
from pathlib import Path

import numpy as np

from bird_olsc import run_bird
from chasing_pytorch import TorchActor
from restartable_strategies import generate_base_feedback, make_context


MULTS=[64,128,256,512]


def stats(xs):
    a=np.asarray(xs,dtype=float)
    return {
        "mean":float(a.mean()),
        "std":float(a.std(ddof=1)) if len(a)>1 else 0.0,
        "min":float(a.min()),
        "max":float(a.max()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--actor",default="artifacts/ddpg_actor_critic.pt")
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--output",default="artifacts/olsc_lazy_extension.json")
    args=ap.parse_args()

    actor=TorchActor(args.actor)
    rows=[]
    for seed in range(args.seeds):
        ctx=make_context(seed=seed,T=5000,inventory=200)
        trajectories,rewards=generate_base_feedback(ctx,actor)
        for family in ["fll","fll-star"]:
            for m in MULTS:
                r=run_bird(
                    actor,
                    seed=seed,
                    selector_name=family,
                    selector_multiplier=float(m),
                    ctx=ctx,
                    trajectories=trajectories,
                    rewards=rewards,
                )
                rows.append({
                    "seed":seed,
                    "family":family,
                    "multiplier":m,
                    "loss_pct":r["revenue_loss_pct"],
                    "revenue":r["BIRD"],
                    "switches":r["switches"],
                    "restarts":r["restarts"],
                    "missing_steps":r["missing_steps"],
                    "selector_epsilon":r["selector_epsilon"],
                })

    summary={}
    for family in ["fll","fll-star"]:
        summary[family]={}
        for m in MULTS:
            rr=[x for x in rows if x["family"]==family and x["multiplier"]==m]
            summary[family][str(m)]={
                "loss_pct":stats([x["loss_pct"] for x in rr]),
                "revenue":stats([x["revenue"] for x in rr]),
                "switches":stats([x["switches"] for x in rr]),
                "selector_epsilon":rr[0]["selector_epsilon"],
                "per_seed_loss":{str(x["seed"]):x["loss_pct"] for x in rr},
            }

    payload={"rows":rows,"summary":summary}
    Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
