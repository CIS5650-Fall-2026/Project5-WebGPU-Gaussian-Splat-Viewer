struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
}

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot: array<u32, 2>,
    scale: array<u32, 2>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<storage, read> splats: array<Splat>;
@group(1) @binding(1) var<storage, read> sorted_indices: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) @interpolate(flat) center_pixels: vec2<f32>,
    @location(1) @interpolate(flat) conic_opacity: vec4<f32>,
    @location(2) @interpolate(flat) color: vec3<f32>,
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    clip_position: vec4<f32>,
    cov_radius: vec4<f32>,
    conic_opacity: vec4<f32>,
    color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    // Quad corner offset
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),

        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    // Draw Quad
    let splat_index = sorted_indices[instance_index];
    let splat = splats[splat_index];
    let center_clip = splat.clip_position;

    let half_size_pixels = vec2<f32>(splat.cov_radius.w);

    let offset_ndc = corners[vertex_index] * (2.0 * half_size_pixels / camera.viewport);

    out.position = vec4<f32>(
        center_clip.xy + offset_ndc * center_clip.w,
        center_clip.z,
        center_clip.w,
    );

    let center_ndc = center_clip.xy / center_clip.w;

    out.center_pixels = vec2<f32>(
        (center_ndc.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - center_ndc.y * 0.5) * camera.viewport.y,
    );

    out.conic_opacity = splat.conic_opacity;

    out.color = splat.color.xyz;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.center_pixels;

    let A = in.conic_opacity.x;
    let B = in.conic_opacity.y;
    let C = in.conic_opacity.z;
    let opacity = in.conic_opacity.w;

    let r2 = max(
        A * d.x * d.x
        + 2.0 * B * d.x * d.y
        + C * d.y * d.y,
        0.0,
    );
    if (r2 > 9.0) {
        discard;
    }

    let weight = exp(-0.5 * r2);
    let alpha = min(0.99, opacity * weight);

    if (alpha < 1.0 / 255.0) {
        discard;
    }

    let color = in.color;

    return vec4<f32>(color * alpha, alpha);
}