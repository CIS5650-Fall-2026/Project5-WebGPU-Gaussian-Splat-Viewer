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
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32, //spherical harmonics degree
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    mean_ndc: vec3<f32>,
    cov_inv: mat2x2<f32>,
    ssaabb_halfextent: vec2<f32>,
    color: vec4<f32>
};

//a way to represent a value that changes with direction, here this value is a Gaussian's RGB color
//Gaussian color changes depending on which position you're looking at it from
//uses a fixed set of basis functions (patterns defined over all directions on a sphere)
//each coeff controls how much one pattern contributes. degree: controls how detailed the directional variation can be (0-constant color, 1-broad directional variation 3-finer variation)
//like combining sine waves to approximate a sound, but SH combines patterns on a sphere to approximate how color varies with direction
//in gaussian splatting, these coefficients are already learned during training and saved to the PLY file, therefore we just read them
struct SphericalHarmonics{
    coefficients: array<u32, 24>, //will unpack to 48 half precision floats
}

//TODO: bind your data here
@group(0) @binding(0) 
var<uniform> camera: CameraUniforms;
@group(1) @binding(0) 
var<storage, read> gaussians_list: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_list: array<SphericalHarmonics>; 
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(3) @binding(0)
var<storage, read_write> splats: array<Splat>;
@group(3) @binding(1)
var<storage, read> render_settings: RenderSettings;

// reads the ith sh coef from the storage buffer 
//c_idx is the index of the SH coefficient that we want to read (0-15)
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let sphericalHarmonics = sh_list[splat_idx];

    //[r g][b r][g b][r g][b r][g b][r g][b r][g b][r g][b r][g b]
    //coefficient 3 -> rIdx 9 gIdx 10 bIdx 11
    let rIdx = 3 * c_idx;
    let gIdx = 3 * c_idx + 1;
    let bIdx = 3 * c_idx + 2;

    //r = [b r] g,b = [g b]
    let r = unpack2x16float(sphericalHarmonics.coefficients[rIdx/2]);
    let g = unpack2x16float(sphericalHarmonics.coefficients[gIdx/2]);
    let b = unpack2x16float(sphericalHarmonics.coefficients[bIdx/2]);
    
    //r[1] g[0] b[1]
    return vec3<f32>(r[rIdx % 2], g[gIdx % 2], b[bIdx % 2]);
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

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if(idx >= arrayLength(&gaussians_list)){
        return;
    }

    //TODO: set up pipeline as described in instruction
    var splat: Splat;

    let gaussian = gaussians_list[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);

    var clip = camera.proj * camera.view * pos;

    //view frustum culling (done before to prevent flipping of negative values)
    if(clip.x > 1.2 * clip.w || clip.x < -1.2 * clip.w ||
        clip.y > 1.2 * clip.w || clip.y < -1.2 * clip.w ||
        clip.z > clip.w || clip.z < 0.0){
        return;
    }
    
    if(clip.w <= 0){ //reject points behind the camera
        return;
    }
    let ndc = clip/clip.w; //we do this now which is ok because the pipeline will divide by 1 anyway

    // Covariance
    // Covariance R
    let rx = unpack2x16float(gaussian.rot[0]);
    let yz = unpack2x16float(gaussian.rot[1]);
    let quat = normalize(vec4<f32>(rx,yz));
    let r = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    let rotation = mat3x3<f32>(
        1.f - 2.f*(y*y + z*z), 2.f*(x*y + r*z), 2.f*(x*z - r*y),
        2.f*(x*y - r*z), 1.f - 2.f*(x*x + z*z), 2.f*(y*z + r*x),
        2.f*(x*z + r*y), 2.f*(y*z - r*x), 1.f - 2.f*(x*x + y*y)
    );

    // Covariance S
    let scale1 = unpack2x16float(gaussian.scale[0]);
    let scale2 = unpack2x16float(gaussian.scale[1]);

    let scale = mat3x3<f32>(
        exp(scale1.x) * render_settings.gaussian_scaling, 0, 0,
        0, exp(scale1.y) * render_settings.gaussian_scaling, 0,
        0, 0, exp(scale2.x) * render_settings.gaussian_scaling
    );

    // Covariance matrix
    let covariance = rotation * scale * transpose(scale) * transpose(rotation);

    // 2D Covariance
    var ssMean = camera.view * pos;

    let tan_fovx = camera.viewport.x / (2*camera.focal.x);
    let tan_fovy = camera.viewport.y / (2*camera.focal.y);

    let limx = 1.3f * tan_fovx;
    let limy = 1.3f * tan_fovy;
    let txtz = ssMean.x/ssMean.z;
    let tytz = ssMean.y/ssMean.z;
    ssMean.x = min(limx, max(-limx, txtz)) * ssMean.z;
    ssMean.y = min(limy, max(-limy, tytz)) * ssMean.z;

    let j = mat3x3<f32>(
        camera.focal.x / ssMean.z, 0.0f, 0.0f,
        0.0f, camera.focal.y / ssMean.z, 0.0f,
        (-camera.focal.x * ssMean.x) / (ssMean.z * ssMean.z), (-camera.focal.y * ssMean.y) / (ssMean.z * ssMean.z),  0.0
    );

    let w = mat3x3<f32>(
        camera.view[0][0], camera.view[0][1], camera.view[0][2],
        camera.view[1][0], camera.view[1][1], camera.view[1][2],
        camera.view[2][0], camera.view[2][1], camera.view[2][2]
    );

    let t = j * w;

    var cov2DTemp = t * covariance * transpose(t);
    //numeric stability - cov is only positive semi-definite, not sufficient to be invertible, adding 0.3 makes it strictly positive
    cov2DTemp[0][0] += 0.3f;
    cov2DTemp[1][1] += 0.3f;

    let cov2D = mat2x2<f32>(cov2DTemp[0][0], cov2DTemp[0][1],
                            cov2DTemp[1][0], cov2DTemp[1][1]);

    let det = (cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1]);

    let mid = 0.5f*(cov2D[0][0] + cov2D[1][1]);

    //eigenvalue gives the squared 1-standard deviation of the radius
    let lambda1 = mid + sqrt(max(0.1f, mid*mid - det));
    let lambda2 = mid - sqrt(max(0.1f, mid*mid - det));

    //using max radius of the two to create a x y screen space aligned bounding box that canc ontain the ellipsoid in any orientation (convservative way to gauarntee coverage)
    let maxRadius = ceil(3.f * sqrt(max(lambda1, lambda2))); //3 standard deviations (98.9%), ceil to round up radius to a whole pixel
    //let maxRadius = 2.;

    splat.ssaabb_halfextent.x = 2 * maxRadius / camera.viewport.x;
    splat.ssaabb_halfextent.y = 2 * maxRadius / camera.viewport.y;

    splat.cov_inv = (1.f/det) * mat2x2<f32>(cov2D[1][1], -cov2D[1][0], -cov2D[0][1], cov2D[0][0]);

    splat.mean_ndc = vec3(ndc.x,ndc.y,ndc.z);

    // Spherical harmonics
    let camPos = camera.view_inv[3].xyz;
    let camToGaus = normalize(pos.xyz - camPos);
    
    let opacity = 1/(1 + exp(-b.y)); //sigmoid - declaring as const (let) in 1 straight equation is faster?

    splat.color = vec4<f32>(computeColorFromSH(camToGaus, idx, u32(render_settings.sh_deg)), opacity);

    let splatIdx = atomicAdd(&sort_infos.keys_size, 1u); //must claim a slot with atomicAdd
    splats[idx] = splat;
    sort_indices[splatIdx] = idx;
    // splats[splatIdx] = splat; //for performance comparisons later (maybe)
    // sort_indices[splatIdx] = splatIdx;
    sort_depths[splatIdx] = ~bitcast<u32>(ssMean.z); //radix sort uses unsigned integer bit patterns (bitcast works because every depth that reaches this point is positive), farther splats get smaller keys and render first

    let keys_per_workgroup = workgroupSize * sortKeyPerThread; //every launchgroup sorts a certain number of keys
    // dispatch launches a compute shader across a specified number of workgroups(blocks)
    
    // increment DispatchIndirect.dispatchx each time you reach limit for one workgroup of keys
    if(splatIdx % keys_per_workgroup == 0){
        atomicAdd(&sort_dispatch.dispatch_x, 1);
    }
}