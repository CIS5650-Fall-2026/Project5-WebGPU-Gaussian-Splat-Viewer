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

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) pixel_offset: vec2<f32>,
    @location(2) conic: vec3<f32>,
    @location(3) opacity: f32
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    center: vec2<f32>,
    half_size_pixels: vec2<f32>,
    conic: vec3<f32>,
    opacity: f32,
    color: vec4<f32>,
};


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sort_indices: array<u32>;



@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {

    //TODO: reconstruct 2D quad based on information from splat, pass 

    let sorted_splat_index = sort_indices[instance_index];
    let splat = splats[sorted_splat_index];

    // two triangles around the center
    var corner = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    // pixel radius to NDC size
    let half_size_ndc = splat.half_size_pixels * 2.0 / camera.viewport; //multiply by 2 because it is the size of NDC

    var out: VertexOutput;

    out.position = vec4<f32>(
        splat.center + corner[vertex_index] * half_size_ndc,
        0.0,
        1.0
    );

    out.color = splat.color;

    // pixel offset from the center of the quad with x with minus for the conic convention
    out.pixel_offset = vec2<f32>(-corner[vertex_index].x, corner[vertex_index].y) * splat.half_size_pixels;

    out.conic = splat.conic;
    out.opacity = splat.opacity;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let pixel_difference = in.pixel_offset;

    let gaussian_power = -0.5 * (
        in.conic.x * pixel_difference.x * pixel_difference.x + //Ax^2
        2.0 * in.conic.y * pixel_difference.x * pixel_difference.y + //2Bxy
        in.conic.z * pixel_difference.y * pixel_difference.y //Cy^2
    );

    // for correct covariance should not happen
    if (gaussian_power > 0.0) {
        discard;
    }

    let gaussian_alpha = min(0.99, in.opacity * exp(gaussian_power)); // opacity gets smaller exponentially

    // throw away basically invisible splats
    if (gaussian_alpha < 1.0 / 255.0) {
        discard;
    }

    return vec4<f32>(in.color.rgb, gaussian_alpha);
}
