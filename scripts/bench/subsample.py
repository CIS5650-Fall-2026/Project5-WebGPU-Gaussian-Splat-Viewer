# writes random subsets of an all-float binary PLY: subsample.py in.ply out_prefix frac1 frac2 ...
import sys, re
import numpy as np
src, prefix, fracs = sys.argv[1], sys.argv[2], [float(f) for f in sys.argv[3:]]
data = open(src, 'rb').read()
end = data.index(b'end_header\n') + len(b'end_header\n')
header = data[:end].decode()
n = int(re.search(r'element vertex (\d+)', header).group(1))
props = len(re.findall(r'^property float', header, re.M))
verts = np.frombuffer(data, dtype='<f4', offset=end, count=n * props).reshape(n, props)
rng = np.random.default_rng(0)
for f in fracs:
    k = int(n * f)
    sel = np.sort(rng.choice(n, k, replace=False))
    out = f'{prefix}_{int(f*100)}.ply'
    with open(out, 'wb') as fo:
        fo.write(re.sub(r'element vertex \d+', f'element vertex {k}', header).encode())
        fo.write(verts[sel].tobytes())
    print(out, k)
