"""Save the current RGB view and both current room maps during evaluation."""
from pathlib import Path
import shutil


def render_live(root, case_id, source, data, step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    root = Path(root)
    start = np.asarray(source['episode']['start_position'])
    path = np.asarray([p['position'] for p in source['poses']])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(data['rgb']); axes[0].axis('off')
    axes[0].set_title(f'Current observation | step {step}')
    maps = []
    for method in ['active', 'occusg']:
        with np.load(root / method / f'{case_id:02d}' / 'latest.npz') as checkpoint:
            labels = checkpoint['labels'].copy()
            if method == 'active':
                occupied, explored = checkpoint['occupied'], checkpoint['explored']
                known, free = explored > 0, (explored > 0) & (occupied == 0)
                h, w = labels.shape
                known, free, labels = [a[::-1, :].T for a in [known, free, labels]]
                extent = [24+start[0]-(h-.5)*.05, 24+start[0]+.025,
                          -24-start[2]-.025, -24-start[2]+(w-.5)*.05]
            else:
                occupancy = checkpoint['occupancy']
                known, free = occupancy >= 0, occupancy == 0
                ox, oy = checkpoint['origin']; res = float(checkpoint['resolution'])
                extent = [ox, ox+known.shape[1]*res, oy, oy+known.shape[0]*res]
        maps.append((known, free, labels, extent))
    bounds = [path[:, 0].min(), path[:, 0].max(), (-path[:, 2]).min(), (-path[:, 2]).max()]
    for known, _, _, extent in maps:
        cells = np.argwhere(known)
        if len(cells):
            h, w = known.shape
            x = extent[0]+(cells[:, 1]+.5)*(extent[1]-extent[0])/w
            y = extent[2]+(cells[:, 0]+.5)*(extent[3]-extent[2])/h
            bounds = [min(bounds[0],x.min()),max(bounds[1],x.max()),min(bounds[2],y.min()),max(bounds[3],y.max())]
    palette = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    for ax, name, (known, free, labels, extent) in zip(axes[1:], ['Active', 'OccuSG'], maps):
        background = np.full(known.shape, .45); background[known] = 0; background[free] = 1
        ax.imshow(background, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
        colors = np.ma.masked_where((labels == 0) | ~free, (labels-1) % 20)
        ax.imshow(colors, origin='lower', extent=extent, cmap=palette, vmin=0, vmax=19, alpha=.8)
        ax.plot(path[:,0],-path[:,2],c='red',lw=1)
        ax.scatter([path[-1,0]],[-path[-1,2]],c='red',marker='o',s=25)
        ax.set_xlim(bounds[0]-.5,bounds[1]+.5);ax.set_ylim(bounds[2]-.5,bounds[3]+.5)
        ax.set_aspect('equal');ax.set_title(name+' | current map')
        ax.set_xlabel('world x (m)');ax.set_ylabel('-world z (m)')
    fig.suptitle(f'Online updates before next navigation step | {Path(source["episode"]["scene_id"]).stem}\nActive topology updates every 10 frames; colors are method-local')
    fig.tight_layout()
    folder = root / 'visualizations' / f'{case_id:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / f'{step:04d}.png'
    fig.savefig(image, dpi=100)
    plt.close(fig)
    temporary = root / 'live.tmp.png'
    shutil.copy2(image, temporary)
    temporary.replace(root / 'live.png')
