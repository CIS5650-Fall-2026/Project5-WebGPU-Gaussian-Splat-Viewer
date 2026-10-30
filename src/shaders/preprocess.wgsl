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
override numGaussians: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

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
    focal: vec2<f32> // Usually to project we have xProj = xCam / (z*tan(fov/2))
                     // Then to get to pixel space we also mult Dim/2, focal just encodes all that in a multiplication
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    dt: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    screen_pos: vec4f,
    cov_inv: vec3f,
    mean_2D: vec2<f32>,
    pixel_radius: f32,
    opacity: f32,
    col: vec4f
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_coeffs : array<u32>;
@group(1) @binding(2)
var<uniform> render_settings: RenderSettings;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<storage,read_write> splats : array<Splat>;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let start_index = splat_idx * 16 * 3 + c_idx * 3;
    
    let converted_index = start_index/2;
    let mod_index = start_index%2;

    let sh_piece_1 = unpack2x16float(sh_coeffs[converted_index]);
    let sh_piece_2 = unpack2x16float(sh_coeffs[converted_index+1]);
    let full_sh = vec4(sh_piece_1, sh_piece_2);
    let sh = select(full_sh.yzw, full_sh.xyz, mod_index == 0);

    return sh;
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

fn quatToRotationMatrix(q: vec4f) -> mat3x3f {
    let r = q.x;
	let x = q.y;
	let y = q.z;
	let z = q.w;

    return transpose(mat3x3f(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    ));
}

fn sigmoid(v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-v));
}

fn offset_pos(p: vec3f) -> vec3f {
    return p;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if(idx >= numGaussians) {
        return;
    }

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 

    // Get gaussian data
    let gauss = gaussians[idx];

    // Create splat
    var splat : Splat = Splat();
    let posXY = unpack2x16float(gauss.pos_opacity[0]);
    let posZOpacity = unpack2x16float(gauss.pos_opacity[1]);
    var gaussPos_world = vec4(posXY, posZOpacity.x, 1.0);
    let rawScale = vec4(unpack2x16float(gauss.scale[0]), unpack2x16float(gauss.scale[1]));
    let rotQuat = vec4(unpack2x16float(gauss.rot[0]), unpack2x16float(gauss.rot[1]));

    // Opacity
    let opacity = sigmoid(posZOpacity.y);
    splat.opacity = opacity;

    // Positions
    let gaussPos_view = camera.view * gaussPos_world;
    var gaussPos_screen = camera.proj * gaussPos_view;
    let proj_w = gaussPos_screen.w;
    gaussPos_screen /= proj_w;
    splat.mean_2D = 0.5*gaussPos_screen.xy*vec2f(camera.viewport);
    splat.screen_pos = gaussPos_screen;

    // Color
    let camera_pos = camera.view_inv[3].xyz;
    splat.col = vec4(computeColorFromSH(normalize(gaussPos_world.xyz - camera_pos), idx, 3u), 0.0);

    // Covar World
    let scale = render_settings.gaussian_scaling * exp(rawScale.xyz);
    let scaleMatrix = mat3x3f(
        scale.x, 0.0, 0.0, 
        0.0, scale.y, 0.0, 
        0.0, 0.0, scale.z
        );
    let rotMatrix = quatToRotationMatrix(normalize(rotQuat));
    let covar_3D_world =  rotMatrix * scaleMatrix * transpose(scaleMatrix) * transpose(rotMatrix);

    // Covar screen
    let camera_view_3x3 = mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let camZ = gaussPos_view.z;
    let camZ2 = camZ*camZ;
    let proj_jacobian = mat3x2(
        camera.focal.x / camZ, 0.0, 
        0.0, -camera.focal.y / camZ, 
        -camera.focal.x * gaussPos_view.x / camZ2, camera.focal.y * gaussPos_view.y / camZ2
        ); // Derivative of projection function f(x, y, z) -> (u, v) where projection matrix flips y
    let T = proj_jacobian * camera_view_3x3;

    var covar_2D_screen = T * covar_3D_world * transpose(T);

    // covar += 0.3I (make sure invertible by making it positive definite NOT SEMI)
    covar_2D_screen[0][0] += 0.3;
    covar_2D_screen[1][1] += 0.3;

    // Get det and encode cov as 3 floats since symmetric
    let cov = vec3(covar_2D_screen[0][0], covar_2D_screen[1][0], covar_2D_screen[1][1]); // Cuz symmetric
    let det = cov.x * cov.z - cov.y * cov.y;
    
    // Get eigenvalues for radius
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det)); // max to prevent -sqrt I think
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    // cov is PD so all eigens are positive
    splat.pixel_radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Check within frustum using pixel_radius as bounding box
    let diff = abs(splat.mean_2D) - 0.6*vec2f(camera.viewport);// - splat.pixel_radius;
    if(
        gaussPos_screen.z < 0. || gaussPos_screen.z > 1. || // Z Condition
        diff.x > 0. || diff.y > 0. || 
        proj_w <= 0. ||
        det <= 0.
    ) {
        return;
    }

    // Invert cov
    let cov_inv = vec3(cov.z, -cov.y, cov.x)/det;
    splat.cov_inv = cov_inv;

    // Add splat data
    let ind = atomicAdd(&sort_infos.keys_size, 1u);
    splats[ind] = splat;
    sort_indices[ind] = ind;
    sort_depths[ind] = bitcast<u32>(10000.-camZ); // Floats interpreted as u32 will maintain their ordering

    if(ind % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
    // todo: remember fill in indices and sort depth which I guess is just bitcast<float> since it maintains order
}