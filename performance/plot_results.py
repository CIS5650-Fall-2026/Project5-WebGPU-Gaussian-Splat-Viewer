from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

out = Path(__file__).resolve().parent
scenes = {}
for scene in ['bonsai', 'bicycle']:
    data = json.loads((out / 'results' / f'{scene}.json').read_text())
    for result in data['results']:
        result['round_medians_gpu_ms'] = [
            statistics.median(sample['gpu_ms'] for sample in result['samples'] if sample['round'] == i)
            for i in range(3)
        ]
    scenes[scene] = {result['condition']: result for result in data['results']}

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
colors={'bonsai':'#167e9b','bicycle':'#d27025'}
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for scene,results in scenes.items():
    x=[64,128,256]
    pre=[results[f'gaussian_wg{v}']['median_preprocess_ms'] for v in x]
    axs[0,0].plot([str(v) for v in x],pre,'o-',label=scene.capitalize(),color=colors[scene])
    for i,v in enumerate(x):
        rounds=[statistics.median(s['preprocess_ms'] for s in results[f'gaussian_wg{v}']['samples'] if s['round']==r) for r in range(3)]
        axs[0,0].vlines(i,min(rounds),max(rounds),color=colors[scene],lw=2)
    total=[results[f'gaussian_wg{v}']['median_gpu_ms'] for v in x]
    axs[0,1].plot([str(v) for v in x],total,'o-',label=scene.capitalize(),color=colors[scene])
    for i,v in enumerate(x):
        rounds=results[f'gaussian_wg{v}']['round_medians_gpu_ms']
        axs[0,1].vlines(i,min(rounds),max(rounds),color=colors[scene],lw=2)
    keys=['gaussian_count25','gaussian_count50','gaussian_wg256']
    axs[1,1].plot([results[k]['input_count']/1000 for k in keys],[results[k]['median_gpu_ms'] for k in keys],'o-',label=scene.capitalize(),color=colors[scene])
for i,(scene,results) in enumerate(scenes.items()):
    for j,key in enumerate(['gaussian_wg256','gaussian_no_culling']):
        x=i+(-.18 if j==0 else .18)
        value=results[key]['median_gpu_ms']
        axs[1,0].bar(x,value,.32,color=['#167e9b','#d27025'][j],label=['Culling on','Culling off'][j] if i==0 else None)
        axs[1,0].text(x,value+.5,f'{value:.2f}',ha='center',fontsize=10)
        rounds=results[key]['round_medians_gpu_ms']
        axs[1,0].vlines(x,min(rounds),max(rounds),color='black',lw=1.5)
axs[1,0].set_xticks([0,1],['Bonsai','Bicycle'])
for ax,title,ylabel,xlabel in [
    (axs[0,0],'Preprocess workgroup size','Preprocess time (ms)','Threads per workgroup'),
    (axs[0,1],'Workgroup size: full GPU pipeline','GPU pipeline time (ms)','Threads per workgroup'),
    (axs[1,0],'View-frustum culling','GPU pipeline time (ms)',''),
    (axs[1,1],'Input Gaussian count','GPU pipeline time (ms)','Input Gaussians (thousands)')]:
    ax.set(title=title,ylabel=ylabel,xlabel=xlabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
fig.suptitle('WebGPU Gaussian Splatting — measured performance\nApple M1 Pro · 960 × 640 · camera preset 0 · 90 samples per condition',fontsize=16)
fig.supxlabel('Medians of 90 GPU timestamp samples. Vertical marks show the range of three round medians. Lower is better.',fontsize=10)
fig.savefig(out/'performance.png',dpi=160)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,key,title in zip(axs,['pointcloud','gaussian_wg256'],['Point cloud: yellow centers','Gaussian: projected ellipses + alpha blending']):
    im=Image.open(out/'images'/f'bicycle-{key}.png').convert('RGBA')
    bg=Image.new('RGBA',im.size,'black');bg.alpha_composite(im)
    ax.imshow(bg.convert('RGB'));ax.set_title(title);ax.axis('off')
fig.suptitle('Same bicycle scene, camera and resolution',fontsize=16)
fig.savefig(out/'comparison.png',dpi=150)
plt.close(fig)
