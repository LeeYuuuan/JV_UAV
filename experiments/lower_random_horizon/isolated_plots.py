"""Standalone plots for the20/10step lower-only experiment."""
from pathlib import Path
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


def save(fig,path):
    path=Path(path);tmp=path.with_suffix(".tmp.png");fig.savefig(tmp,dpi=140);tmp.replace(path);fig.clear()


def curves(trainer,output,evaluations):
    output=Path(output);history=trainer.history
    compact=[{k:v for k,v in row.items() if k not in ("return_events","failure_events","warmup_exploration_counts_total")} for row in history]
    temp=output/"training_curves.tmp.json"
    temp.write_text(json.dumps(compact,allow_nan=False),encoding="utf-8");temp.replace(output/"training_curves.json")
    fig=Figure(figsize=(12,9),layout="constrained");FigureCanvasAgg(fig);axes=fig.subplots(3,2)
    panels=[("lower_return","Episode return"),("sensor_coverage_fraction","Fraction of sensors visited"),
            ("mean_max_backlog","Mean maximum backlog / packets"),("lower_collected_packets","Packets collected"),
            ("long_success_ratio","Long-group return success fraction"),("reward_terms","Mean reward components / step")]
    for ax,(key,title) in zip(axes.flat,panels):
        if key=="reward_terms":
            ax.set_yscale("symlog",linthresh=.01)
            for term in ("system_max_post_service","topk_clear","oob","return_failure","return_distance"):
                if history:ax.plot([r["episode"] for r in history],[r["lower_reward_terms_mean"].get(term,0) for r in history],label=term,lw=.8)
            if history:ax.legend(fontsize=6)
        elif history:
            values=[r["long_return_success"]/max(r["long_return_count"],1) if key=="long_success_ratio" else r[key] for r in history]
            ax.plot([r["episode"] for r in history],values,lw=.8,alpha=.6)
            if len(values)>=20:ax.plot([r["episode"] for r in history][19:],np.convolve(values,np.ones(20)/20,mode="valid"),lw=1.5)
        ax.set(title=title,xlabel="Completed episode");ax.grid(alpha=.2)
    fig.suptitle("Lower-only SAC | random20/10step serving groups")
    save(fig,output/"lower_training_curves.png")
    fig=Figure(figsize=(11,6),layout="constrained");FigureCanvasAgg(fig);axes=fig.subplots(2,2)
    for ax,keys,title in zip(axes.flat,[("sac_actor_loss","sac_entropy_loss"),("sac_critic_loss",),("sac_q_mean","sac_target_mean"),("sac_alpha",)],
                           ["Actor loss and entropy term","Critic loss","Q and target means","SAC alpha"]):
        for key in keys:
            rows=[r for r in history if key in r]
            if rows:ax.plot([r["low_steps"] for r in rows],[r[key] for r in rows],label=key,lw=.8)
        ax.set(title=title,xlabel="Training lower steps");ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    save(fig,output/"loss_curves.png")
    fig=Figure(figsize=(11,6),layout="constrained");FigureCanvasAgg(fig);axes=fig.subplots(2,2)
    for seed in sorted({r["seed"] for r in evaluations}):
        rows=[r for r in evaluations if r["seed"]==seed]
        x=[r["low_steps"] for r in rows]
        ys=[[r["sensor_coverage_fraction"] for r in rows],[r["mean_max_backlog"] for r in rows],
            [r["long_return_success"]/max(r["long_return_count"],1) for r in rows],[r["lower_return"] for r in rows]]
        for ax,y in zip(axes.flat,ys):ax.plot(x,y,label=f"seed{seed}",lw=1)
    for ax,title in zip(axes.flat,["Sensor coverage","Mean max backlog / packets","Long-group return success","Lower episode return"]):
        ax.set(title=title,xlabel="Training lower steps");ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    save(fig,output/"evaluation_curves.png")


def dashboard(env,path):
    scene=env.scene;fig=Figure(figsize=(13,10),layout="constrained");FigureCanvasAgg(fig)
    axes=fig.subplots(2,2);ax=axes[0,0]
    points=ax.scatter(scene.sensor_pos[:,0],scene.sensor_pos[:,1],c=scene.sensor_backlog,cmap="YlOrBr",s=20,zorder=4)
    fig.colorbar(points,ax=ax,label="Backlog / packets",shrink=.7)
    colors=["#177d9a","#dc9b23","#8d64b1","#d9617e","#359679","#6878bc"]
    for uid,positions in env.paths.items():
        xy=np.asarray(positions);kind="long" if uid in env.long_ids else "short"
        ax.plot(xy[:,0],xy[:,1],color=colors[uid],label=f"U{uid} {kind}",lw=1.3)
        for point in xy[1:]:ax.add_patch(Circle(point,scene.cfg["uavs"]["coverage_radius_m"],color=colors[uid],alpha=.035,lw=0))
    for event in env.return_events:
        uid=event["uav_id"];p=np.asarray(event["service_endpoint"])
        if event["success"]:ax.plot([p[0],scene.airship_pos[0]],[p[1],scene.airship_pos[1]],ls="--",lw=.8,color=colors[uid],alpha=.6)
        else:ax.scatter(*p,marker="x",color="red",s=70,zorder=8)
    ax.scatter(*scene.airship_pos,marker="*",s=140,color="#253b53",zorder=9)
    ax.add_patch(Circle(scene.airship_pos,scene.cfg["sensors"].get("partition_radius_m",1800),fill=False,ls="--",ec="#52697c",lw=1))
    ax.set(xlim=(0,scene.world_size),ylim=(0,scene.world_size),aspect="equal",xlabel="East / m",ylabel="North / m",title="Service paths and per-step footprints")
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
    ax=axes[0,1]
    for uid in range(6):ax.plot(np.array(env.soc_history)[:,uid]*100,label=f"U{uid}",color=colors[uid])
    ax.axvline(env.short_steps,ls="--",color="gray");ax.set(xlabel="Lower step",ylabel="SOC (%)",title="Real SOC, including automatic return deductions");ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2)
    ax=axes[1,0];trace=env.metrics.trace
    ax.plot(trace.backlog_max_post_timeline,label="Maximum");ax.plot(trace.backlog_mean_post_timeline,label="Mean")
    ax.set(xlabel="Lower step",ylabel="Packets",title="Sensor backlog after collection");ax.legend();ax.grid(alpha=.2)
    ax=axes[1,1];events=sorted(env.return_events,key=lambda r:r["uav_id"])
    ids=np.arange(len(events));ax.bar(ids-.18,[r["soc_before_return"]*100 for r in events],width=.36,label="SOC before return")
    ax.bar(ids+.18,[r["required_return_soc"]*100 for r in events],width=.36,label="Return energy required")
    ax.set(xticks=ids,xticklabels=[f"U{r['uav_id']}" for r in events],ylabel="SOC (%)",title="Automatic return feasibility");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    coverage=np.mean(env.metrics.trace.sensor_visit_count>0)
    fig.suptitle(f"Lower-only | long{env.long_ids.tolist()} / short{env.short_ids.tolist()} | {env.steps}steps | sensors visited {coverage:.1%}")
    fig.supxlabel("Dashed returns show settled endpoints, not flight-time samples. No charging or upper policy is simulated.",fontsize=8)
    save(fig,path)
