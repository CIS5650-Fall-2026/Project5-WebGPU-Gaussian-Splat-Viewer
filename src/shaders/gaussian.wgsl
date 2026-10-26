@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(1) var<storage, read> splats_list: array<Splat>;

@group(1) @binding(2) var<storage, read> sort_indices: array<u32>;

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

struct Splat {
    //TODO: information defined in preprocess compute shader
    mean_ndc: vec3<f32>,
    cov_inv: mat2x2<f32>,
    ssaabb_halfextent: vec2<f32>,
    color: vec4<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    //if we were not using the same structs as vertex output/fragment input, these locations would make more sense
    @location(0) cov_inv: vec3<f32>, //matrix can't be passed directlyas a location based vertex output
    @location(1) color: vec4<f32>, //tells the pipeline where the vertex shader writes the color and where the fragment shader receives it
    @location(2) pixelCenter: vec2<f32>
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32, @builtin(instance_index) in_instance_index: u32) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
   var out: VertexOutput;

   //overlapping fragments are blended in the back to front order, even if their shader calculations finish in a different order
    let splat = splats_list[sort_indices[in_instance_index]];
    let corners = array<vec2<f32> ,4>(
        vec2<f32>(-1.0,-1.0),
        vec2<f32>(1.0,-1.0),
        vec2<f32>(-1.0,1.0),
        vec2<f32>(1.0,1.0)
    );

    let corner = corners[in_vertex_index];

    out.position = vec4<f32>(splat.mean_ndc.x + corner.x * splat.ssaabb_halfextent.x,
                                splat.mean_ndc.y + corner.y * splat.ssaabb_halfextent.y, 
                                splat.mean_ndc.z, 
                                1.0);
    out.cov_inv = vec3<f32>(splat.cov_inv[0][0], splat.cov_inv[0][1], splat.cov_inv[1][1]);
    
    out.color = splat.color; //vec4<f32>(splat.ssaabb_halfextent.x*2, splat.ssaabb_halfextent.y*2, 0, 1);

    out.pixelCenter = vec2<f32>((splat.mean_ndc.x + 1.0)/2.0 * camera.viewport.x,
                                (1.0 - splat.mean_ndc.y)/2.0 * camera.viewport.y);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var pixelDist = in.pixelCenter - in.position.xy;
    //pixelDist.x = -pixelDist.x; //projection matrix already flips y? //covariance and framebuffer offsets need matching axis conventions, NDC has Y increasing upward, and framebuffer pixels have Y increasing downward. flipping y reconciles this but flipping x instead produces the same gaussian value, (x, -y) and (-x, y) but a gaussian is symmetric about its center

    //conic section: curve formed by the intersection of a plane and a double-napped cone
    let power = -(in.cov_inv.x * pixelDist.x * pixelDist.x
                + 2*in.cov_inv.y * pixelDist.x * pixelDist.y
                + in.cov_inv.z * pixelDist.y * pixelDist.y)/2;

    let alpha = in.color.w * exp(power);

    return vec4(in.color.xyz, alpha);
}