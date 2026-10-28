// First, indirect drawing; similar to basic point cloud step, should be drawing
// quads at point positions

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // information passed from vertex shader to fragment shader
    @location(0) @interpolate(flat) sort_color: vec4<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) conic: vec3<f32>,
    @location(3) @interpolate(flat) center: vec2<f32>
};

struct Splat {
    // information defined in preprocess compute shader
    // relevant for culling and atomic count
    pos: u32, // ndc
    size: u32, // quad
    conic_opacity: array<u32, 2>, // conic xyz and opacity
    color: array<u32, 2> // rgb
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> sort_indices: array<u32>;
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

@vertex
fn vs_main( @builtin(vertex_index) iv: u32,
            @builtin(instance_index) ii: u32
) -> VertexOutput {
    // reconstruct 2D quad based on information from splat, pass 
    var quad = array<vec2<f32>, 6> (
        vec2(-1.0, -1.0), 
        vec2(1.0, -1.0), 
        vec2(1.0, 1.0),
        vec2(-1.0, -1.0), 
        vec2(1.0, 1.0), 
        vec2(-1.0, 1.0)
    );

    let s = splats[sort_indices[ii]];
    let center = unpack2x16float(s.pos);
    let size = unpack2x16float(s.size);
    let rg = unpack2x16float(s.color[0]);
    let b_ = unpack2x16float(s.color[1]);
    let conic_xy = unpack2x16float(s.conic_opacity[0]);
    let conic_z = unpack2x16float(s.conic_opacity[1]);

    var out: VertexOutput;
    out.position = vec4<f32>(center + quad[iv] * size , 0., 1.);
    out.sort_color = vec4<f32>(size.x, size.y, 0.0, 1.0);
    out.conic = vec3<f32>(conic_xy.x, conic_xy.y, conic_z.x);
    out.center = center;
    out.color = vec4<f32>(rg.x, rg.y, b_.x, conic_z.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // offset calculation (x coord reversed)
    // in.position.xy: y points down (framebuffer pixels)
    // converting to ndc means y up
    let frag_ndc = vec2<f32>(
        in.position.x / camera.viewport.x * 2.0 - 1.0,
        1.0 - in.position.y / camera.viewport.y * 2.0
    );
    var d = (frag_ndc - in.center) * camera.viewport * 0.5;
    d.x = -d.x;

    // return vec4<f32>(d.x, d.y, 0.0, 1.0); // can enable to see debug coloring
    
    let power = - 0.5 * 
        (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) -
        in.conic.y * d.x * d.y;

    if (power > 0.0) {
        discard;
    }

    let alpha = min(0.99, in.color.a * exp(power));
    
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    return vec4<f32>(in.color.rgb * alpha, alpha);
}