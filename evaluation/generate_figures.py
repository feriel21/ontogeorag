
#!/usr/bin/env python3
"""generate_all_figures.py — OntoGeoRAG journal paper figures
Usage: python generate_all_figures.py --outdir figures/
"""
import argparse, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors

C1="#2166AC"; C2="#F4A582"; C3="#D6604D"; CG="#4DAC26"; CR="#D01C8B"; CB="#636363"
matplotlib.rc("font",family="serif",size=10)
matplotlib.rc("axes",labelsize=11,titlesize=12)

ITER_NAMES=["Iter 1","Iter 2","Iter 3","Iter 4","Iter 5\n(Run6)","Iter 6\n(Run7)","Final\n(Tiered)"]
ITER_TRIPS=[45,89,112,137,60,60,105]
ITER_HALL=[78.,62.,45.,37.9,22.8,0.0,2.9]
ITER_REC=[7.7,11.5,15.4,65.4,19.2,19.2,34.6]
ITER_PREC=[22.,31.,42.,22.,69.,100.,97.1]
REL_T1={"hasDescriptor":29,"causes":6,"occursIn":10,"triggers":11,"affects":8,"partOf":1,"controls":2,"indicates":1,"underlies":1,"overlies":0}
REL_T2={"hasDescriptor":12,"causes":12,"occursIn":6,"triggers":1,"affects":2,"partOf":2,"controls":0,"indicates":0,"underlies":0,"overlies":1}
DESC_ITERS={
    "chaotic":      [1,1,1,1,1,1,1],"transparent":[0,1,1,1,1,1,1],
    "blocky":       [0,0,1,1,1,1,1],"layered":    [0,0,0,1,0,0,1],
    "parallel":     [0,0,1,1,1,1,1],"continuous": [0,1,1,1,1,1,1],
    "discontinuous":[0,0,1,1,1,1,1],"massive":    [0,0,0,1,0,1,1],
    "low-amplitude":[1,1,1,1,1,1,1],"high-amplitude":[0,0,0,1,1,1,1],
    "deformed":     [0,0,0,1,1,1,1],"stratified": [0,0,0,0,0,0,0],
    "undeformed":   [0,0,0,0,0,0,0],
}
TOP_ENTS=[("mass transport deposit",42),("gas hydrate dissociation",7),("slope failure",4),
          ("basal shear surface",3),("sea-level lowstands",3),("debris flow",3),
          ("folds and thrusts",3),("continental slope",3),("wave action",2),
          ("pore pressure",2),("headscarp",2),("turbidite",2)]
STRAT={"descriptor":{"total":43,"tier1":29,"recall":8},"causal":{"total":40,"tier1":34,"recall":4},
       "context":{"total":22,"tier1":16,"recall":2},"rescue":{"total":10,"tier1":8,"recall":2}}
T1_EX=[
    ("mass transport deposit","hasDescriptor","chaotic","basal shear surface separates chaotic seismic facies of the MTD"),
    ("mass transport deposit","hasDescriptor","high-amplitude","MTD upper surface is commonly a high amplitude positive reflection"),
    ("gas hydrate dissociation","causes","pore pressure increase","gas hydrate dissociation generates excess pore pressure"),
    ("slope failure","triggers","mass transport deposit","slope failures generate mass transport deposits"),
    ("sea-level lowstands","triggers","slope failure","sea-level lowstands increase sedimentation rates...slope instability"),
    ("basal shear surface","indicates","slide plane","basal shear surface represents the slide plane of the mass transport"),
]
T2_EX=[
    ("mass transport deposit","hasDescriptor","layered","internal layered reflections observed within MTD sequences"),
    ("wave action","causes","slope instability","storm wave loading can destabilize poorly consolidated slopes"),
    ("debris flow","occursIn","continental slope","debris flows are common on continental slopes"),
]

def sv(fig,outdir,name):
    p=str(outdir/name); fig.savefig(p,bbox_inches="tight",dpi=200); plt.close(fig)
    print(f"  OK  {name}")

def fig01(outdir):
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(14,5)); ax.set_xlim(0,14); ax.set_ylim(0,5); ax.axis("off")
    bxs=[(0.3,2.3,2.0,1.3,"Corpus\n41 papers\n(LB2019)","#AEC6CF"),
         (2.7,2.3,2.0,1.3,"BM25 Indexing\n& Chunking","#B5EAD7"),
         (5.1,2.3,2.0,1.3,"Ontology-guided\nQuery Gen.\n220+ queries","#FFDAC1"),
         (7.5,2.3,2.0,1.3,"LLM Extraction\nQwen 7B\nfew-shot","#FFB7B2"),
         (9.9,2.3,2.0,1.3,"Evidence-based\nVerification\n(CoT)","#C7CEEA"),
         (12.1,2.3,1.7,1.3,"Tiered KG\n105 triples\n2.9% hall.","#E2F0CB")]
    for x,y,w,h,lbl,clr in bxs:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07",facecolor=clr,edgecolor="#555",lw=1.2))
        ax.text(x+w/2,y+h/2,lbl,ha="center",va="center",fontsize=8.5,fontweight="bold",multialignment="center")
    ay=2.3+1.3/2
    for x in [2.3,4.7,7.1,9.5,11.9]:
        ax.annotate("",xy=(x+0.4,ay),xytext=(x,ay),arrowprops=dict(arrowstyle="->",color="#333",lw=1.5))
    ax.text(7.0,4.5,"LB2019 Ontology (88 nodes, 173 edges)",ha="center",va="center",fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3",facecolor="#FFF9C4",edgecolor="#AAA"))
    for xt in [6.1,8.5,10.9]:
        ax.annotate("",xy=(xt,3.6),xytext=(7.0,4.2),
                    arrowprops=dict(arrowstyle="->",color="#999",lw=1.0,linestyle="dashed"))
    ax.set_title("OntoGeoRAG Pipeline Overview",fontsize=13,fontweight="bold",pad=8)
    plt.tight_layout(); sv(fig,outdir,"fig01_pipeline_overview.pdf")

def fig02(outdir):
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(ITER_NAMES))
    colors=[C1 if h<5 else C2 if h<50 else C3 for h in ITER_HALL]
    bars=ax.bar(x,ITER_HALL,color=colors,edgecolor="white",lw=0.8,zorder=3)
    for bar,h,n in zip(bars,ITER_HALL,ITER_TRIPS):
        ax.text(bar.get_x()+bar.get_width()/2,h+1.5,f"{h:.1f}%\n({n}T)",ha="center",va="bottom",fontsize=8)
    ax.plot([0,1],[sep,sep],
        color="#666",
        lw=1.5,
        ls="--",
        transform=ax.transAxes)
    ax.set_xticks(x); ax.set_xticklabels(ITER_NAMES,fontsize=8.5)
    ax.set_ylabel("Hallucination Rate (%)"); ax.set_ylim(0,92)
    ax.set_title("Hallucination Rate Across Pipeline Iterations\n(T=triples retained)",fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.3,zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.annotate("Final tiered KG\n2.9% hallucination\n105 triples",xy=(6,2.9),xytext=(4.5,38),
                fontsize=8.5,color=CG,arrowprops=dict(arrowstyle="->",color=CG,lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor=CG,alpha=0.9))
    plt.tight_layout(); sv(fig,outdir,"fig02_hallucination_evolution.pdf")

def fig03(outdir):
    fig,ax=plt.subplots(figsize=(7,5))
    colors=[C3,C3,C3,C2,C2,C1,CG]
    for i,(r,p,nm,n,c) in enumerate(zip(ITER_REC,ITER_PREC,ITER_NAMES,ITER_TRIPS,colors)):
        ax.scatter(r,p,s=n*1.8,color=c,edgecolor="white",lw=1.2,zorder=3,alpha=0.85)
        dx,dy=(2,3) if i<4 else (-16,5)
        ax.annotate(nm.replace("\n"," "),(r,p),textcoords="offset points",xytext=(dx,dy),fontsize=7.5)
    for i in range(len(ITER_REC)-1):
        ax.annotate("",xy=(ITER_REC[i+1],ITER_PREC[i+1]),xytext=(ITER_REC[i],ITER_PREC[i]),
                    arrowprops=dict(arrowstyle="-|>",color="#BBB",lw=1.0))
    ax.axhspan(85,108,alpha=0.07,color=CG,label="High precision zone (>85%)")
    ax.set_xlabel("Recall vs LB2019 Reference Edges (%)"); ax.set_ylabel("Precision / Non-Hallucination Rate (%)")
    ax.set_title("Precision-Recall Trade-off Across Iterations\n(circle size proportional to triple count)",fontsize=11)
    ax.set_xlim(-2,75); ax.set_ylim(15,112); ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); sv(fig,outdir,"fig03_precision_recall_tradeoff.pdf")

def fig04(outdir):
    descs=list(DESC_ITERS.keys()); inames=["It.1","It.2","It.3","It.4","It.5","It.6","Final"]
    mat=np.array([DESC_ITERS[d] for d in descs])
    fig,ax=plt.subplots(figsize=(9,5.5))
    cmap=matplotlib.colors.ListedColormap(["#F0F0F0",C1])
    ax.imshow(mat,aspect="auto",cmap=cmap,vmin=0,vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(inames)
    ax.set_yticks(range(len(descs))); ax.set_yticklabels(descs,fontsize=9)
    ax.set_title("Descriptor Coverage Across Iterations\n(blue=recovered, grey=missing)",fontsize=11)
    for i in range(len(descs)):
        for j in range(7):
            v=mat[i,j]
            ax.text(j,i,"+" if v else "-",ha="center",va="center",fontsize=10,
                    color="white" if v else "#CCC",fontweight="bold")
    for j in range(7):
        ax.text(j,len(descs)+0.1,f"{int(mat[:,j].sum())}/13",ha="center",va="bottom",fontsize=8,fontweight="bold",color=C1)
    ax.text(6.6,11.0,"= layered",fontsize=7.5,color="#888",ha="left",va="center")
    ax.text(6.6,12.0,"rare in corpus",fontsize=7.5,color="#888",ha="left",va="center")
    ax.set_xlim(-0.5,7.2); ax.set_ylim(-0.5,len(descs)+0.7)
    plt.tight_layout(); sv(fig,outdir,"fig04_descriptor_coverage_heatmap.pdf")

def fig05(outdir):
    rels=sorted(REL_T1.keys(),key=lambda r:-(REL_T1[r]+REL_T2.get(r,0)))
    rels=[r for r in rels if REL_T1[r]+REL_T2.get(r,0)>0]
    t1=[REL_T1.get(r,0) for r in rels]; t2=[REL_T2.get(r,0) for r in rels]
    x=np.arange(len(rels)); w=0.38
    fig,ax=plt.subplots(figsize=(9,4.5))
    b1=ax.bar(x-w/2,t1,w,label="Tier 1 — Verified",color=C1,edgecolor="white")
    b2=ax.bar(x+w/2,t2,w,label="Tier 2 — Literature-implied",color=C2,edgecolor="white")
    for bar in list(b1)+list(b2):
        h=bar.get_height()
        if h>0: ax.text(bar.get_x()+bar.get_width()/2,h+0.2,str(int(h)),ha="center",va="bottom",fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(rels,rotation=30,ha="right",fontsize=9)
    ax.set_ylabel("Number of Triples")
    ax.set_title("Relation Distribution by Confidence Tier\n(Final Tiered KG, 105 triples)",fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); sv(fig,outdir,"fig05_relation_distribution.pdf")
def fig06(outdir):
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    ax=axes[0]
    bars=ax.bar(["Tier 1\nVerified","Tier 2\nLit.-implied"],[69,36],color=[C1,C2],edgecolor="white",width=0.5)
    for bar,c in zip(bars,[69,36]):
        ax.text(bar.get_x()+bar.get_width()/2,c+0.5,str(c),ha="center",va="bottom",fontweight="bold",fontsize=11)
    ax.set_title("Triples by Tier",fontsize=11); ax.set_ylabel("Count"); ax.set_ylim(0,85)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax=axes[1]
    xlbls=["Desc.\n(n/13)","Recall\n(n/26)","Non-\nHalluc.\n(%)"]
    v1=[9,6,100.]; v12=[11,9,97.1]; x=np.arange(3); w=0.3
    ax.bar(x-w/2,v1,w,label="Tier 1",color=C1,edgecolor="white")
    ax.bar(x+w/2,v12,w,label="Tier 1+2",color=C2,edgecolor="white")
    for xi,mv in zip(x,[13,26,100]): ax.plot([xi-0.45,xi+0.45],[mv,mv],":",color="#999",lw=1)
    ax.set_xticks(x); ax.set_xticklabels(xlbls,fontsize=8)
    ax.set_title("Quality Metrics by Tier",fontsize=11); ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax=axes[2]
    ax.pie([69,36],labels=["Tier 1\nVerified\n(69)","Tier 2\nLit.-implied\n(36)"],colors=[C1,C2],
           autopct="%1.0f%%",startangle=90,wedgeprops=dict(edgecolor="white",lw=2),textprops={"fontsize":8.5})
    ax.set_title("KG Composition\n(105 total triples)",fontsize=11)
    plt.suptitle("Final Tiered Knowledge Graph — Summary Statistics",fontsize=12,fontweight="bold",y=1.02)
    plt.tight_layout(); sv(fig,outdir,"fig06_tiered_kg_stats.pdf")

def fig07(outdir):
    try: import networkx as nx
    except ImportError: print("  SKIP fig07 — pip install networkx"); return
    G=nx.DiGraph()
    edges=[("mass transport\ndeposit","hasDescriptor","chaotic",C1),("mass transport\ndeposit","hasDescriptor","transparent",C1),
           ("mass transport\ndeposit","hasDescriptor","blocky",C1),("mass transport\ndeposit","hasDescriptor","high-amplitude",C1),
           ("mass transport\ndeposit","hasDescriptor","low-amplitude",C1),("mass transport\ndeposit","hasDescriptor","continuous",C1),
           ("mass transport\ndeposit","hasDescriptor","discontinuous",C1),("slope failure","triggers","mass transport\ndeposit",C1),
           ("gas hydrate\ndissociation","causes","pore pressure\nincrease",C1),("pore pressure\nincrease","triggers","slope failure",C1),
           ("sea-level\nlowstands","triggers","slope failure",C1),("mass transport\ndeposit","occursIn","continental slope",C1),
           ("basal shear\nsurface","indicates","slide plane",C1),("mass transport\ndeposit","affects","slope stability",C1),
           ("debris flow","hasDescriptor","massive",C2),("mass transport\ndeposit","hasDescriptor","layered",C2),
           ("wave action","causes","slope failure",C2),("mass transport\ndeposit","partOf","mass transport\ncomplex",C2)]
    nc={}
    for s,r,o,c in edges:
        G.add_edge(s,o,label=r,color=c)
        if s not in nc: nc[s]=c
        if o not in nc: nc[o]=c
    nc["mass transport\ndeposit"]="#1A1A2E"
    pos=nx.spring_layout(G,seed=42,k=2.5)
    fig,ax=plt.subplots(figsize=(13,8)); ax.axis("off")
    nx.draw_networkx_nodes(G,pos,node_color=[nc.get(n,CB) for n in G.nodes()],
                           node_size=[3500 if n=="mass transport\ndeposit" else 1800 for n in G.nodes()],alpha=0.88,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=7,font_color="white",font_weight="bold",ax=ax)
    for s,o,d in G.edges(data=True):
        nx.draw_networkx_edges(G,pos,edgelist=[(s,o)],edge_color=d["color"],width=1.8,alpha=0.7,
                               arrows=True,arrowsize=15,connectionstyle="arc3,rad=0.08",ax=ax)
    nx.draw_networkx_edge_labels(G,pos,edge_labels={(s,o):d["label"] for s,o,d in G.edges(data=True)},
                                 font_size=6.5,bbox=dict(boxstyle="round,pad=0.1",facecolor="white",alpha=0.7),ax=ax)
    ax.legend(handles=[mpatches.Patch(color=C1,label="Tier 1 — Verified"),
                       mpatches.Patch(color=C2,label="Tier 2 — Literature-implied"),
                       mpatches.Patch(color="#1A1A2E",label="Central MTD node")],
              loc="lower left",fontsize=9,framealpha=0.9)
    ax.set_title("Knowledge Graph Subgraph — Mass Transport Deposits\n(edge color = confidence tier)",
                 fontsize=12,fontweight="bold")
    plt.tight_layout(); sv(fig,outdir,"fig07_kg_subgraph.pdf")

def fig08(outdir):
    ents=[e for e,_ in TOP_ENTS]; cnts=[c for _,c in TOP_ENTS]
    type_c={"mass transport deposit":"#1A1A2E","gas hydrate dissociation":C3,"slope failure":C3,
            "debris flow":C3,"pore pressure":C3,"basal shear surface":"#7B68EE","headscarp":"#7B68EE",
            "sea-level lowstands":"#20B2AA","folds and thrusts":"#20B2AA","continental slope":"#20B2AA",
            "turbidite":C3,"wave action":CB}
    fig,ax=plt.subplots(figsize=(9,5))
    y=range(len(ents))
    bars=ax.barh(list(y),cnts,color=[type_c.get(e,CB) for e in ents],edgecolor="white",height=0.65)
    for bar,c in zip(bars,cnts):
        ax.text(c+0.2,bar.get_y()+bar.get_height()/2,str(c),va="center",fontsize=9,fontweight="bold")
    ax.set_yticks(list(y)); ax.set_yticklabels(ents,fontsize=9)
    ax.set_xlabel("Number of Triples (as subject or object)")
    ax.set_title("Top Entities in the Knowledge Graph",fontsize=11)
    ax.invert_yaxis(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x",alpha=0.25); ax.set_xlim(0,max(cnts)+7)
    ax.legend(handles=[mpatches.Patch(color="#1A1A2E",label="Central MTD concept"),
                       mpatches.Patch(color=C3,label="Process/Trigger"),
                       mpatches.Patch(color="#7B68EE",label="Morphological feature"),
                       mpatches.Patch(color="#20B2AA",label="Setting/Environment")],fontsize=8,loc="lower right")
    plt.tight_layout(); sv(fig,outdir,"fig08_entity_distribution.pdf")

def fig09(outdir):
    strats=list(STRAT.keys())
    totals=[STRAT[s]["total"] for s in strats]; t1s=[STRAT[s]["tier1"] for s in strats]; recs=[STRAT[s]["recall"] for s in strats]
    x=np.arange(len(strats)); w=0.28
    fig,ax1=plt.subplots(figsize=(8,4.5)); ax2=ax1.twinx()
    ax1.bar(x-w,totals,w,label="Total triples",color=CB,alpha=0.7,edgecolor="white")
    ax1.bar(x,t1s,w,label="Tier 1 triples",color=C1,edgecolor="white")
    ax2.plot(x,recs,"o--",color=CR,lw=2,ms=7,label="Recall contribution (n/26)",zorder=5)
    for xi,r in zip(x,recs): ax2.text(xi,r+0.1,str(r),ha="center",va="bottom",fontsize=9,color=CR,fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([s.capitalize() for s in strats],fontsize=10)
    ax1.set_ylabel("Number of Triples"); ax2.set_ylabel("Recall Contribution",color=CR)
    ax2.tick_params(axis="y",labelcolor=CR); ax2.set_ylim(0,12)
    ax1.set_title("Contribution by Query Strategy",fontsize=11)
    l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,lb1+lb2,fontsize=8.5,loc="upper right")
    ax1.spines["top"].set_visible(False); ax1.grid(axis="y",alpha=0.2)
    plt.tight_layout(); sv(fig,outdir,"fig09_strategy_contribution.pdf")

def fig10(outdir):
    fig,ax=plt.subplots(figsize=(14,6.5)); ax.axis("off")
    headers=["Subject","Relation","Object","Tier","Evidence (excerpt)"]
    col_x=[0.0,0.18,0.32,0.50,0.57]; col_w=[0.18,0.14,0.18,0.07,0.43]
    for h,x0,cw in zip(headers,col_x,col_w):
        ax.add_patch(mpatches.FancyBboxPatch((x0,0.93),cw,0.06,boxstyle="square,pad=0",
                     facecolor=C1,edgecolor="none",transform=ax.transAxes))
        ax.text(x0+cw/2,0.96,h,ha="center",va="center",fontsize=9,fontweight="bold",color="white",transform=ax.transAxes)
    rows=[(s,r,o,"Tier 1",ev,C1) for s,r,o,ev in T1_EX]+[(s,r,o,"Tier 2",ev,C2) for s,r,o,ev in T2_EX]
    rh=0.095; y0=0.88
    for i,(subj,rel,obj,tier,ev,c) in enumerate(rows):
        y=y0-i*rh; bg="#EBF5FB" if c==C1 else "#FEF9E7"
        ax.add_patch(mpatches.FancyBboxPatch((0,y-rh*0.85),1.0,rh*0.9,boxstyle="square,pad=0",
                     facecolor=bg,edgecolor="#DDD",lw=0.5,transform=ax.transAxes))
        cells=[subj,rel,obj,tier,ev[:72]+("..." if len(ev)>72 else "")]
        for j,(cell,x0,cw) in enumerate(zip(cells,col_x,col_w)):
            fc=C1 if (j==3 and tier=="Tier 1") else (C2 if j==3 else "#222")
            ax.text(x0+cw/2,y-rh*0.35,cell,ha="center",va="center",fontsize=7.8,color=fc,
                    fontweight="bold" if j==3 else "normal",transform=ax.transAxes)
    sep=y0-(len(T1_EX)-0.5)*rh
    ax.axhline(sep,color="#666",lw=1.5,ls="--",transform=ax.transAxes)
    ax.text(0.5,sep+0.008,"── Tier 1 above  ──  Tier 2 below ──",ha="center",va="bottom",
            fontsize=7.5,color="#666",transform=ax.transAxes)
    ax.set_title("Qualitative Examples: Extracted Triples with Evidence",fontsize=12,fontweight="bold",pad=14)
    plt.tight_layout(); sv(fig,outdir,"fig10_qualitative_examples.pdf")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--outdir",default="figures"); args=parser.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    print(f"Generating figures in: {outdir}/\n" + "="*50)
    fig01(outdir); fig02(outdir); fig03(outdir); fig04(outdir); fig05(outdir)
    fig06(outdir); fig07(outdir); fig08(outdir); fig09(outdir); fig10(outdir)
    print("="*50+f"\nDone — 10 figures in {outdir}/")
    for f in sorted(outdir.glob("fig*.pdf")): print(f"  {f.name}")

if __name__=="__main__":
    main()