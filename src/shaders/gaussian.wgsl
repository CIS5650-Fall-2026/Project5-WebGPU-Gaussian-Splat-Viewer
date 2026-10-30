struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;

@group(2) @binding(0)
var<storage, read> sort_indices : array<u32>;

@group(3) @binding(0)
var<storage,read> splats : array<Splat>;

struct VertexInput {
    @location(0) pos: vec2f, // Local quad coordinate
    @builtin(instance_index) instance_id: u32
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_pos: vec2f,
    //TODO: information passed from vertex shader to fragment shader
    @location(1) @interpolate(flat) instance_id: u32
};

struct Splat {
    screen_pos: vec4f,
    cov_inv: vec3f, // Symmetric mat2x2
    mean_2D: vec2<f32>,
    pixel_radius: f32,
    opacity: f32,
    col: vec4f
};

fn screenToPixel(scr: vec2f) -> vec2f {
    return 0.5*scr*vec2f(camera.viewport);
}

fn pixelToScreen(pix: vec2f) -> vec2f {
    return 2.0 * pix / vec2f(camera.viewport);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let splat = splats[sort_indices[in.instance_id]];
    let splat_screen_pos = splat.screen_pos;

    var pix_gauss_pos = screenToPixel(splat_screen_pos.xy);
    pix_gauss_pos += in.pos*splat.pixel_radius;
    let corner = vec4(pixelToScreen(pix_gauss_pos), splat_screen_pos.zw);

    out.position = corner;

    out.pixel_pos = vec2f(0.5*out.position.xy/out.position.w*vec2f(camera.viewport));

    out.instance_id = in.instance_id;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let splat = splats[sort_indices[in.instance_id]];
    let uv = splat.mean_2D/vec2f(camera.viewport)+0.5;

    let d = in.pixel_pos - splat.mean_2D;

    // Expansion of r2 = d^T * cov_inv * d in gauss pdf = exp(-r2/2)
    let cov_inv = mat2x2f(splat.cov_inv.x, splat.cov_inv.y, splat.cov_inv.y, splat.cov_inv.z);
    let r2 = dot(d,cov_inv * d);//let r2 = splat.cov_inv.x * d.x * d.x + 2.0 * splat.cov_inv.y * d.x * d.y + splat.cov_inv.z * d.y * d.y;

    let alpha = exp(-r2/2.0)*splat.opacity;

    if(alpha < 0.001) { discard; }

    return vec4<f32>(splat.col.rgb, alpha);
}