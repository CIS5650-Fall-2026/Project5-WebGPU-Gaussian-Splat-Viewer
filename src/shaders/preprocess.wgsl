const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect { // how many gaussians to process per launch of compute shader, indirect draw call
    dispatch_x: atomic<u32>,
    dispatch_y: u32, // not reelvant
    dispatch_z: u32, // not reelvant
} // launch on the gpu

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>, 
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    ndc: vec3f,
    conic: vec3f,
    radius: f32,
    color: vec4f
};

struct SH_Coeff {
    data: array<u32, 24>,   // 3 * 16 f16s, so 24 f32s
}

//TODO: bind your data here

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<uniform> render_settings: RenderSettings;
@group(1) @binding(2)
var<storage,read> sh_buffer : array<SH_Coeff>; // idk what the array would be



@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos; // out, how many gaussians to process
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>; // out, for radix
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>; // out, for gaussian.wgsl
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect; // out, for radix dispatching necessary number of workgroups

// out
@group(3) @binding(0)
var<storage, read_write> splat_buffer: array<Splat>;


/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let i = c_idx * 3u;
    // if odd, we wanna extract the upper half, otherwise we extract lower half
    let is_odd = (i % 2u) == 1u; // 1 if odd, 0 if even
    // unpacking
    let sh_idx = i / 2;
    
    let first = unpack2x16float(sh_buffer[splat_idx].data[sh_idx]);
    let second = unpack2x16float(sh_buffer[splat_idx].data[sh_idx + 1]);

    if (is_odd) {
        return vec3f(first.y, second.x, second.y);
    } else {
        return vec3f(first.x, first.y, second.x);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn load_pos(g: Gaussian) -> vec3f {
    
    let xy = unpack2x16float(g.pos_opacity[0]);
    let z = unpack2x16float(g.pos_opacity[1]).x;
    return vec3f(xy, z);
}

fn load_opacity(g: Gaussian) -> f32 {
    let op = unpack2x16float(g.pos_opacity[1]).y;
    // apply sigmoid, since can have negative or over 1
    return 1/ (1 + exp(-op));
}

fn load_rot(g: Gaussian) -> vec4f {
    let xy = unpack2x16float(g.rot[0]);
    let zw = unpack2x16float(g.rot[1]);
    return vec4f(xy, zw);
}

fn load_scale(g: Gaussian) -> vec3f {
    let xy = unpack2x16float(g.scale[0]);
    let z = unpack2x16float(g.scale[1]).x;
    return vec3f(xy, z);
}

// referenced https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
fn compute_cov3D(g: Gaussian, scaling: f32) -> mat3x3f {
    let q = normalize(load_rot(g));

	let r = q.x;
	let x = q.y;
	let y = q.z;
	let z = q.w;

	// Compute rotation matrix from quaternion
	let rotmat =  mat3x3f(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    let s = exp(load_scale(g)) * scaling;
    let scalemat = mat3x3f(
        vec3f(s.x, 0.0, 0.0),
        vec3f(0.0, s.y, 0.0),
        vec3f(0.0, 0.0, s.z)
    );

    let sr = scalemat * rotmat;
    return transpose(sr) * sr;
}

// referenced https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
fn compute_cov2D(mean: vec3f, cov3d: mat3x3f) -> mat2x2f
{
    let tan_fovx = camera.viewport.x / (2.0 * camera.focal.x);
    let tan_fovy = camera.viewport.y / (2.0 * camera.focal.y);

    var t = (camera.view * vec4f(mean, 1.0)).xyz;

	let limx = 1.3 * tan_fovx;
	let limy = 1.3 * tan_fovy;
	t.x = min(limx, max(-limx, t.x / t.z)) * t.z;
	t.y = min(limy, max(-limy, t.y / t.z)) * t.z;

    // jacobian of perspective projection
	let J = mat3x3f(
		camera.focal.x / t.z, 0, -(camera.focal.x * t.x) / (t.z * t.z),
		0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
		0, 0, 0);

	let W = transpose(mat3x3f( camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz)); 

    let T = W * J;
    let cov = transpose(T) * transpose(cov3d) * T;
    var cov2d = mat2x2f(cov[0].xy, cov[1].xy);
    
    // low pass filter
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f; 
    return cov2d;
}

fn compute_conic(det: f32, cov2D: mat2x2f) -> vec3f
{
    let det_inv = 1.f / det;
    return vec3f(cov2D[1][1] * det_inv, -cov2D[0][1] * det_inv, cov2D[0][0] * det_inv);
}

fn compute_radius(det: f32, cov2D: mat2x2f) -> f32
{
    let mid = 0.5 * (cov2D[0][0] + cov2D[1][1]);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    return ceil(3 * sqrt(max(lambda1, lambda2)));
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {return;}
    //TODO: set up pipeline as described in instruction
    
    // world space to ndc space
    let g = gaussians[idx];

    let worldpos = load_pos(g);
    
    // world to clip
    var clip = camera.proj * camera.view * vec4f(worldpos, 1.0); 
    if (clip.w <= 0) { return; } // behind camera
    
    // clip to ndc
    let ndc = clip.xyz / clip.w;
    
     // out of camera bounds, cull
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z < 0 || ndc.z > 1) { return; }

    // computing cov
    let cov3d = compute_cov3D(g, render_settings.gaussian_scaling);
    let cov2D = compute_cov2D(worldpos.xyz, cov3d);

    // computing 2d conic, max radius, and max quad size
    let det = (cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1]);
    if (det <= 0.f) {return;}

    let i = atomicAdd(&sort_infos.keys_size, 1u); // add this splat to number of splats in render pass


    // get conic
    let conic = compute_conic(det, cov2D);
    // get radius
    let radius = compute_radius(det, cov2D);

    let opacity = load_opacity(g);

    let campos = camera.view_inv[3].xyz;
    let dir = normalize(worldpos - campos);
    let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));
    
    splat_buffer[idx] = Splat(ndc, conic, radius, vec4f(color, opacity));

    sort_depths[i] = bitcast<u32>(1 - ndc.z); // sort key, ndc z always between 0 and 1
    sort_indices[i] = idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (i % keys_per_dispatch == 0u)
    {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}