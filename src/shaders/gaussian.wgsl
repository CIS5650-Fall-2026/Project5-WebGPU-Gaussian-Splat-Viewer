struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) center_px: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,  // rgb, opacity
};

// same layout as in preprocess.wgsl
struct Splat {
    center: vec2<f32>,      // NDC
    size: u32,              // f16x2: quad half extent in NDC
    conic: array<f32, 3>,   // inverse 2D covariance (xx, xy, yy) in pixels
    color: array<u32, 2>,   // f16x4: rgb, opacity
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    // instances are drawn in sorted (back to front) order
    let splat = splats[sort_indices[instance_index]];
    let size = unpack2x16float(splat.size);

    // triangle strip: (-1,-1), (1,-1), (-1,1), (1,1)
    let corner = vec2<f32>(f32(vertex_index & 1u), f32(vertex_index >> 1u)) * 2.0 - 1.0;

    var out: VertexOutput;
    out.position = vec4<f32>(splat.center + corner * size, 0.0, 1.0);
    // NDC -> framebuffer pixels (y down), matches @builtin(position) in the fragment shader
    out.center_px = vec2<f32>(0.5 * splat.center.x + 0.5, 0.5 - 0.5 * splat.center.y) * camera.viewport;
    out.conic = vec3<f32>(splat.conic[0], splat.conic[1], splat.conic[2]);
    out.color = vec4<f32>(unpack2x16float(splat.color[0]), unpack2x16float(splat.color[1]));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.center_px;
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    if power > 0.0 {
        discard;
    }
    let alpha = min(0.99, in.color.a * exp(power));
    if alpha < 1.0 / 255.0 {
        discard;
    }
    // premultiplied alpha, blended back to front
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
