struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @interpolate(flat) @location(0) splatNdc: vec2<f32>,
    @interpolate(flat) @location(1) conic: vec3<f32>,
    @interpolate(flat) @location(2) op: f32,
    @interpolate(flat) @location(3) color: vec3<f32>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    ndc: vec2<f32>,
    size: vec2<f32>,
    conic: vec3<f32>,
    op: f32,
    color: vec3<f32>,
};

@group(0) @binding (0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VertexOutput {
    _ = camera.view;
    var out: VertexOutput;

    let splat = splats[sort_indices[iIdx]];

    var corners: array<vec2f, 6> = array<vec2f, 6>(
        vec2f(-1, -1),
        vec2f(1, -1),
        vec2f(1, 1),
        vec2f(-1, -1),
        vec2f(1, 1),
        vec2f(-1, 1),
    );
    let corner = corners[vIdx];

    out.position = vec4f(corner * splat.size + splat.ndc, 0.0, 1.0);
    out.splatNdc = splat.ndc;
    out.conic = splat.conic;
    out.op = splat.op;
    out.color = splat.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let splatPixelPos = vec2<f32>(
        (in.splatNdc.x * 0.5 + 0.5) * camera.viewport.x, 
        (-in.splatNdc.y * 0.5 + 0.5) * camera.viewport.y
    );

    let offset = splatPixelPos - in.position.xy;

    let conic_mat = mat2x2f(
        vec2f(in.conic.x, in.conic.y),
        vec2f(in.conic.y, in.conic.z)
    );

    let g = exp(-0.5 * dot(offset, conic_mat * offset));

    let alpha = min(0.99, in.op * g);
    if (alpha < (1.0 / 255.0)) { discard; }

    return vec4<f32>(in.color * alpha, alpha);
}