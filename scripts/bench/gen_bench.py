# generates the action list for a benchmark session: gen_bench.py full|quick shots_dir
import json, sys
full = sys.argv[1] == 'full'
O = sys.argv[2] if len(sys.argv) > 2 else None
A = [{"ui": False}]
def rec(label):
    A.extend([{"wait": 1500}, {"record": label, "samples": 6, "interval": 400}])
def sel(i, v): A.append({"select": [i, v]})
# selects: 0 renderer, 1 gaussian_mode, 2 debug_view, 3 workgroup_size, 4 precision
# checkboxes: 0 frustum_culling, 1 tight_bounds
rec("pointcloud")
sel(0, "gaussian")
if full and O:
    A += [{"wait": 1000}, {"shot": f"{O}/gaussian.png"}]
    for v in ["quads", "quad_size", "quad_color", "pixel_offset"]:
        sel(2, v); A += [{"wait": 800}, {"shot": f"{O}/dbg_{v}.png"}]
    sel(2, "final")
    for m, name in [(0.25, "025"), (0.5, "050"), (1.5, "150"), (1.0, "100")]:
        A += [{"number": [0, m]}, {"wait": 1000}, {"shot": f"{O}/mult_{name}.png"}]
    sel(1, "tile"); A += [{"wait": 1200}, {"shot": f"{O}/tile.png"}]
    sel(2, "tile_heatmap"); A += [{"wait": 1000}, {"shot": f"{O}/dbg_tile_heatmap.png"}]
    sel(2, "final"); sel(1, "quad")
for mode in ["quad", "tile"]:
    sel(1, mode)
    if full:
        for prec in ["f32", "f16"]:
            sel(4, prec)
            for wg in ["32", "64", "128", "256"]:
                sel(3, wg); rec(f"{mode}|{prec}|wg{wg}")
        sel(4, "f32")
        A.append({"checkbox": [1, False]}); rec(f"{mode}|tight_off")
        A.append({"checkbox": [1, True]})
    else:
        rec(f"{mode}|f32|wg256")
if full:
    for preset in range(10):
        A.append({"key": str(preset)})
        for mode in ["quad", "tile"]:
            sel(1, mode)
            A.append({"checkbox": [0, True]}); rec(f"{mode}|preset{preset}|cull_on")
            A.append({"checkbox": [0, False]}); rec(f"{mode}|preset{preset}|cull_off")
            A.append({"checkbox": [0, True]})
print(json.dumps(A))
