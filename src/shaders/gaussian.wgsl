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
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) center: vec2<f32>,

    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: store information for 2D splat rendering
    ndc: vec3f,
    conic: vec3f,
    radius: f32,
    color: vec4f
};


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0) // ccheck on this??
var<storage,read> splats : array<Splat>;

@group(2) @binding(0) // ccheck on this??
var<storage,read> sorted_indices : array<u32>;



// const QUAD_SIZE = 0.01;

const corners = array<vec2f, 4>(
    vec2f( 1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0, -1.0),
  );

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_index: u32) -> VertexOutput {
    var out: VertexOutput;    
    _ = camera.viewport;

    let i = sorted_indices[in_index]; // get sorted gaussian
    let s = splats[i];

    // take ratio of camera viewport dims
    let combo = corners[in_vertex_index] * s.radius / camera.viewport;

    out.position = vec4f(s.ndc.xy + combo, s.ndc.z, 1.0);
    out.color = s.color;
    out.conic = s.conic;
    out.center = vec2f((s.ndc.x * 0.5 + 0.5) * camera.viewport.x, (0.5 - s.ndc.y * 0.5) * camera.viewport.y); // in pixel space

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // use conic matrix to determine whethr a point is inside the splat, compute opacity to update each fragment's color

    // compute offset from frag pos to center of quad with x coord reversed
    // center to screen space

    var offset = in.position.xy - in.center; // vec2f

    let conicmat = mat2x2f(
        vec2f(in.conic.x, in.conic.y), vec2f(in.conic.y, in.conic.z)
    );
    let opacity = exp(-0.5 * dot(offset, conicmat * offset) );
    
    var outcol = in.color;

    if (opacity < 0.01) {
        outcol.a = 0.0; // opacity is 0
    } else {
        outcol.a = min(1.0, opacity * in.color.a);
    }

    // opacity decreases exponentially as it gets more distant from center
    return outcol;
}