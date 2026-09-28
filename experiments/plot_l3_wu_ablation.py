"""Plot saved development raw vs sampling-only corrected drift checks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    report=json.loads((a.directory/'report.json').read_text())
    outputs=[a.directory/'sampling_drift.png',a.directory/'sampling_drift.svg']
    if any(p.exists() for p in outputs): raise FileExistsError('Plot already exists')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,kind in zip(axes,('W','U')):
        for n in sorted({r['fit_paths'] for r in report['grid']}):
            rows=sorted([r for r in report['grid'] if r['fit_paths']==n],key=lambda r:r['kappa'])
            x=[r['kappa'] for r in rows]
            raw=[r[kind]['raw_rate'] for r in rows]
            corrected=[r[kind]['sampling_corrected_rate'] for r in rows]
            line=ax.plot(x,raw,'o-',label=f'fit={n}: raw')[0]
            ax.plot(x,corrected,'s--',color=line.get_color(),label=f'fit={n}: sampling UCB')
        ax.set_title(kind+' drift checks at development anchors')
        ax.set_xscale('log',base=2); ax.set_xlabel('Successors per anchor (kappa)')
        ax.set_ylim(-.04,1.04); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Pass fraction (not recurrence probability)')
    fig.suptitle('Fixed architecture; no cell correction or global L3 proof',fontsize=11)
    fig.tight_layout()
    for file in outputs: fig.savefig(file,dpi=160)
    plt.close(fig)


if __name__=='__main__': main()
