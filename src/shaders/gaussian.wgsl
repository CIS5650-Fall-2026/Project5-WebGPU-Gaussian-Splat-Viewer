struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    center: u32,
    size: u32,
    color_a: u32,
    color_b: u32,
    conic_a: u32,
    conic_b: u32,
};

const minAlpha = 0.99;

@group(0) @binding(0)
var<storage,read> splatArr : array<Splat>;
@group(0) @binding(1)
var<storage,read> sortIndices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) vertex_color: vec4<f32>,
    // same for every pixel of a quad, so no interpolation
    @location(1) @interpolate(flat) centerPx: vec2<f32>,
    @location(2) @interpolate(flat) conic: vec3<f32>,
};


//create array representing the square corners
var<private> corners = array<vec2<f32>, 6>(vec2<f32>(-1., -1.),vec2<f32>( 1., -1.),vec2<f32>(-1.,  1.),vec2<f32>(-1.,  1.),vec2<f32>( 1., -1.),vec2<f32>( 1.,  1.),);
fn camToPix(cam: vec2<f32>) -> 
vec2<f32> {
    let x = 0.5 + cam.x *0.5;
    let y = 0.5 - cam.y*0.5;
    return vec2<f32>(
        x * camera.viewport.x, y * camera.viewport.y
    );
}

fn calcOffset(center: vec2<f32>, pix: vec2<f32>) ->
vec2<f32> {
    return center - pix;
}
fn calcGaussianPower(conic: vec3<f32>, a: vec2<f32>) -> f32 {
    return -0.5 * (conic.x * a.x * a.x + conic.z * a.y * a.y) 
        - conic.y * a.x * a.y;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vert: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;
    // splats are drawn in sorted (far to near) order
    let splat = splatArr[sortIndices[instance]];
    //convert to float
    let size = unpack2x16float(splat.size);
    let center = unpack2x16float(splat.center);
    let cornerOffset = size * corners[vert] * 0.5;
    let outX = center + cornerOffset;
    let out_y = 0.;
    let out_z = 1.;
    out.position = vec4<f32>(outX, out_y, out_z);
    out.vertex_color = vec4<f32>(unpack2x16float(splat.color_a), unpack2x16float(splat.color_b));
    let conicA = unpack2x16float(splat.conic_a);
    let conicB = unpack2x16float(splat.conic_b).x;
    out.conic = vec3<f32>(conicA.x, conicA.y, conicB);
    out.centerPx = camToPix(center);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    const showOffset = false;//change for debugging
    let p = calcOffset(in.centerPx, in.position.xy);
    if !showOffset {
        let pow = calcGaussianPower(in.conic, p);
        if pow <= 0. {
            let alpha = min(minAlpha, in.vertex_color.a * exp(pow));
            return vec4<f32>(in.vertex_color.rgb, alpha);
        } else {
            discard;
        }

        
    } 
       return vec4<f32>(p.x, p.y, 0., 1.); 
    
}
