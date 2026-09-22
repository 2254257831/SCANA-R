"""Evidence-derived publication figures for the MuJoCo extension (Python only)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/mujoco_extension'
TASKS=['reach-v3','push-v3','pick-place-v3','door-open-v3','drawer-open-v3','button-press-v3']
NAMES=['Reach','Push','Pick and place','Open door','Open drawer','Press button']
METHODS=['Clean repeat','Gaussian recovery','Frozen observations','SCANA-R','Demonstration kNN']
LABELS=['Clean','Gaussian','Frozen','SCANA-R','kNN']
mpl.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.linewidth':.6,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
CMAP=LinearSegmentedColormap.from_list('muted_teal',['#f3f5f5','#c3d6d3','#6d9797','#315b66'])

def export(fig,name):
    for ext in ['svg','pdf','png','tiff']:
        folder=ROOT/'outputs/publication_tiff' if ext=='tiff' else OUT
        folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.04)
        if ext=='svg':
            path=folder/f'{name}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)

def matrix(ax,values,rows,columns):
    masked=np.ma.masked_invalid(values);im=ax.imshow(masked,vmin=0,vmax=100,cmap=CMAP,aspect='auto')
    ax.set_xticks(range(len(columns)),columns);ax.set_yticks(range(len(rows)),rows)
    ax.tick_params(axis='both',length=0,pad=7)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v=values[i,j]
            if np.isfinite(v):ax.text(j,i,f'{v:.0f}',ha='center',va='center',color='white' if v>=63 else '#28383e',fontsize=9)
    return im

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.05))
    for ax,task,label in zip(axs.flat,TASKS,NAMES):
        ax.imshow(plt.imread(ROOT/'outputs/mujoco_extension_media'/f'{task}-reference.png'))
        ax.set_axis_off();ax.text(.5,-.055,label,ha='center',va='top',transform=ax.transAxes,fontsize=10)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=.06,wspace=.035,hspace=.15)
    export(fig,'figure5_task_scenes')
    d=pd.read_csv(ROOT/'results/metaworld_extension_v1/summary.csv')
    # One continuous matrix; the blank separator divides matched evaluation conditions.
    values=np.full((6,11),np.nan)
    for col,(condition,metric) in enumerate([('nominal','success'),('pulse','post_pulse_success')]):
        sub=d[(d.scenario==condition)&(d.metric==metric)&(d.task!='Macro average')]
        for i,t in enumerate(TASKS):
            for j,m in enumerate(METHODS):values[i,col*6+j]=100*float(sub[(sub.task==t)&(sub.method==m)].iloc[0]['mean'])
    fig,ax=plt.subplots(figsize=(7.2,3.1));fig.subplots_adjust(left=.16,right=.94,bottom=.18,top=.83)
    im=matrix(ax,values,NAMES,LABELS+['']+LABELS)
    ax.text(2,-1.12,'Standard layouts',ha='center',fontweight='bold')
    ax.text(8,-1.12,'After execution pulse',ha='center',fontweight='bold')
    for j,label in enumerate(ax.get_xticklabels()):
        label.set_fontsize(8)
        if j in [3,9]:label.set_fontweight('bold')
    cax=fig.add_axes([.962,.18,.013,.65]);cb=fig.colorbar(im,cax=cax,ticks=[0,50,100]);cb.ax.tick_params(labelsize=8,length=2);cb.outline.set_visible(False)
    export(fig,'figure6_multitask_success')
    a=pd.read_csv(ROOT/'results/act_extension_v1/summary.csv')
    cond=['replan_1','replan_4','nominal','replan_16','pulse_early','pulse_late','observation_noise']
    method=['Clean repeat','Gaussian recovery','Frozen context','Calibrated recovery'];v=np.full((9,7),np.nan)
    for k,t in enumerate(['transfer_cube','insertion']):
        for i,m in enumerate(method):
            for j,c in enumerate(cond):
                metric='post_pulse_success' if c.startswith('pulse') else 'success'
                v[k*5+i,j]=100*float(a[(a.task==t)&(a.method==m)&(a.scenario==c)&(a.metric==metric)].iloc[0]['mean'])
    fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.19,right=.94,bottom=.16,top=.96)
    im=matrix(ax,v,['Clean','Gaussian','Frozen obs.','SCANA-R','','Clean','Gaussian','Frozen obs.','SCANA-R'],
              ['Replan 1','Replan 4','Replan 8','Replan 16','Early pulse','Late pulse','Obs. noise'])
    ax.text(-2.2,1.5,'Transfer Cube',rotation=90,va='center',ha='center',fontweight='bold')
    ax.text(-2.2,6.5,'Insertion',rotation=90,va='center',ha='center',fontweight='bold')
    for tick in ax.get_xticklabels():tick.set_fontsize(8)
    for tick in ax.get_yticklabels():
        if tick.get_text()=='SCANA-R':tick.set_fontweight('bold')
    cax=fig.add_axes([.965,.16,.012,.8]);cb=fig.colorbar(im,cax=cax,ticks=[0,50,100]);cb.ax.tick_params(labelsize=8,length=2);cb.outline.set_visible(False)
    export(fig,'figure7_replanning_stress')
    (OUT/'figure_contract.json').write_text(json.dumps(dict(backend='Python matplotlib',font='Times New Roman',width_inches=7.2,
        figure5='Six official simulator task scenes from accepted reference episodes; no learned-policy success claim.',
        figure6='Task-specific nominal and post-pulse performance; fixed 0-100% scale; 5 policy seeds x20 layouts.',
        figure7='Frozen ACT-policy sensitivity to action replanning interval, contact timing and current-observation noise.',
        statistics='Means shown. Per-seed variability and paired intervals in companion CSV/tables; no iid window inference.',
        image_integrity='Simulator camera only; no object edits, selective retouching or generated imagery.'),indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Exported Figures 5-7 as SVG, PDF, PNG and TIFF.')

if __name__=='__main__':main()
