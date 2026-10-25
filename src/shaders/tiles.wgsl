alias SplatFloat = f32;
struct Splat {
    center: vec4<f32>, half_size: vec2<f32>, radius: f32,
    conic: vec4<SplatFloat>, color: vec4<SplatFloat>,
}
struct Grid { width: u32, height: u32, columns: u32, rows: u32 }
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> depths: array<u32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> entries: array<vec2<u32>>;
@group(0) @binding(7) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var<storage, read_write> sort_dispatch: array<u32>;
@group(0) @binding(9) var<storage, read_write> scratch: array<vec2<u32>>;
override stage: u32 = 0u;
override run_size: u32 = 1u;
override source_scratch: bool = false;
fn offset(tile: u32) -> u32 {
    return atomicLoad(&counts[grid.columns * grid.rows + tile]);
}
const tile_size = 16u;
fn pixel_center(s: Splat) -> vec2<f32> {
    return (s.center.xy * vec2<f32>(0.5, -0.5) + 0.5) * vec2<f32>(f32(grid.width), f32(grid.height));
}
// Count first, then scatter into packed per-tile ranges.
@compute @workgroup_size(256)
fn bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= info[0]) { return; }
    let index = indices[gid.x];
    let s = splats[index];
    if (s.center.z < 0. || s.center.z > 1.) { return; }
    let center = pixel_center(s);
    let lo = max(vec2<i32>(0), vec2<i32>(floor((center - s.radius) / 16.)));
    let hi = min(vec2<i32>(i32(grid.columns)-1, i32(grid.rows)-1), vec2<i32>(floor((center + s.radius) / 16.)));
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let tile = u32(y) * grid.columns + u32(x);
            let slot = atomicAdd(&counts[tile], 1u);
            if (stage == 1u) {
                let destination = offset(tile) + slot;
                if (destination < arrayLength(&entries)) {
                    entries[destination] = vec2<u32>(depths[gid.x], index);
                }
            }
        }
    }
}
@compute @workgroup_size(1)
fn prefix() {
    var total = 0u;
    let tiles = grid.columns * grid.rows;
    for (var tile = 0u; tile < tiles; tile++) {
        let count = atomicLoad(&counts[tile]);
        atomicStore(&counts[tiles + tile], total);
        // Saturation preserves the overflow marker without wrapping u32.
        total = min(arrayLength(&entries) + 1u, total + count);
        atomicStore(&counts[tile], 0u);
    }
    sort_dispatch[0] = select(0u, (info[0] + 3839u) / 3840u, total > arrayLength(&entries));
}
fn less(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}
fn key_at(i: u32) -> vec2<u32> {
    if (source_scratch) { return scratch[i]; }
    return entries[i];
}
// Each key finds its rank in the adjacent sorted run. Distinct depth/index
// pairs give every invocation a distinct destination, including tied depths.
@compute @workgroup_size(256)
fn merge_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let tile = group.y * grid.columns + group.x;
    let count = atomicLoad(&counts[tile]);
    let base = offset(tile);
    if (base + count > arrayLength(&entries)) { return; }
    for (var i = tid; i < count; i += 256u) {
        let key = key_at(base + i);
        let block = (i / (2u * run_size)) * (2u * run_size);
        let right = (i - block) >= run_size;
        let own_start = select(block, block + run_size, right);
        let other_start = min(count, select(block + run_size, block, right));
        let other_end = min(count, other_start + run_size);
        var lo = other_start;
        var hi = other_end;
        while (lo < hi) {
            let mid = (lo + hi) / 2u;
            if (less(key_at(base + mid), key)) { lo = mid + 1u; }
            else { hi = mid; }
        }
        let destination = base + block + (i - own_start) + (lo - other_start);
        if (source_scratch) { entries[destination] = key; }
        else { scratch[destination] = key; }
    }
}
@compute @workgroup_size(8,8)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x>=grid.width || gid.y>=grid.height) { return; }
    let tile=(gid.y/tile_size)*grid.columns+gid.x/tile_size;
    let count=atomicLoad(&counts[tile]);
    // An overflowing tile filters the complete depth-sorted list in compute.
    // This has no capacity limit and never drops a contributing splat.
    let dense = offset(tile) + count > arrayLength(&entries);
    let length = select(count, info[0], dense);
    let pixel=vec2<f32>(gid.xy)+0.5;
    var color=vec3<f32>(0.); var transmittance=1.;
    // Near to far allows fully covered pixels to stop early.
    for (var n=length; n>0u; n--) {
        var index: u32;
        if (dense) { index = indices[n-1u]; }
        else { index = entries[offset(tile)+n-1u].y; }
        let s=splats[index];
        if (s.center.z<0. || s.center.z>1.) { continue; }
        let d=(pixel-pixel_center(s))*vec2<f32>(1.,-1.);
        if (any(abs(d)>vec2<f32>(s.radius))) { continue; }
        let conic = vec4<f32>(s.conic);
        let rgba = vec4<f32>(s.color);
        let power=-0.5*(conic.x*d.x*d.x+2.*conic.y*d.x*d.y+conic.z*d.y*d.y);
        let alpha=min(0.99,rgba.a*exp(power));
        if (power>0. || alpha<1./255.) { continue; }
        color+=transmittance*alpha*rgba.rgb;
        transmittance*=1.-alpha;
        if (transmittance<0.0001) { break; }
    }
    textureStore(output,vec2<i32>(gid.xy),vec4<f32>(color,1.));
}
