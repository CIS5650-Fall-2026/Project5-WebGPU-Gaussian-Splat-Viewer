
struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

// struct Gaussian {
//     pos_opacity: array<u32,2>,
//     rot: array<u32,2>,
//     scale: array<u32,2>
// };

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos: vec2<f32>, // NDC center
    size: vec2<f32>, // NDC half width
    color: vec4<f32>,
    conic_opacity: vec4<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
    @location(1) conic_opacity: vec4<f32>,
    @location(2) center: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(@builtin(instance_index) idx: u32, @builtin(vertex_index) vid: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    const QUAD_VERTS = array<vec2<f32>, 6>(
        vec2( 1.f,  1.f),
        vec2(-1.f,  1.f),
        vec2( 1.f, -1.f),
        vec2( 1.f, -1.f),
        vec2(-1.f,  1.f),
        vec2(-1.f, -1.f),
    );

    let splat = splats[sort_indices[idx]];

    let ndcVertPos = QUAD_VERTS[vid];
    let ndcCenter = splat.pos;

    var out: VertexOutput;
    var ndcXY = ndcCenter + ndcVertPos * splat.size;
    out.position = vec4(ndcXY, 0.f, 1.f);
    out.color = splat.color;
    out.conic_opacity = splat.conic_opacity;
    // NDC -> pixels, y flipped since NDC +y is up but pixel +y is down
    out.center = (ndcCenter * vec2<f32>(0.5f, -0.5f) + 0.5f) * camera.viewport;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Pixel offset from center
    let d = in.center - in.position.xy;

    // Falloff
    let conic = in.conic_opacity.xyz;
    let opacity = in.conic_opacity.w;
    let power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.f) {
        discard;
    }
    
    let alpha = min(0.99, opacity * exp(power));
    if (alpha < 1.f / 255.f) {
        discard;
    }

    return vec4(in.color.rgb * alpha, alpha);
}