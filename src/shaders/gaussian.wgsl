struct Splat {
    center: vec2<f32>,
    quad: vec2<f32>,
    color: vec4<f32>,
    conic: vec3<f32>,
    opacity: f32
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct SortInfos {
    keys_size: u32,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0) 
var<uniform> camera: CameraUniforms;
@group(0) @binding(1) 
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<storage, read> sort_infos: SortInfos;
@group(0) @binding(3)
var<storage, read> sort_indices : array<u32>;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) conic: vec3<f32>,
    @location(3) opacity: f32
    };

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32, // quad corners:0,1,2,3
    @builtin(instance_index) instance_index: u32 // splat index 
) -> VertexOutput {
    
    // from back to front (z-depth)
    let n = sort_infos.keys_size;
    let splat = splats[sort_indices[n - 1u - instance_index]];

    // which corner of quad? 
    var offset: vec2<f32>;
if (vertex_index == 0u) {offset = vec2<f32>(-1., -1.);}
    else if (vertex_index == 1u) {offset = vec2<f32>(1., -1.);}
    else if (vertex_index == 2u) {offset = vec2<f32>(-1., 1.);}
    else {offset = vec2<f32>(1., 1.);}

    var out: VertexOutput;
    out.position = vec4<f32>(splat.center + offset * splat.quad, 0., 1.);

    let offset_pix = offset * splat.quad * camera.viewport * 0.5;
    out.offset = vec2<f32>(offset_pix.x, -offset_pix.y);

    out.color = splat.color.rgb;
    out.conic = splat.conic;
    out.opacity = splat.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.offset;
    // 0 at center, more negative outwards
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y)
                - in.conic.y * d.x * d.y;
    if (power > 0.) { discard; }

    let alpha = min(0.99, in.opacity * exp(power));
    if (alpha < 1.0 / 255.0) { discard; }

    return vec4<f32>(in.color * alpha, alpha);
}