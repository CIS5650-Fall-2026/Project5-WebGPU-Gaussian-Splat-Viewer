const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const splatdim = 48u;
const dim = 3u;
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
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    center: u32,
    size: u32,
    color_a: u32,
    color_b: u32,
    conic_a: u32,
    conic_b: u32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> renderSettings: RenderSettings;
@group(1) @binding(0)
var<storage, read_write> gaussianArr : array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splatArr : array<Splat>;
@group(1) @binding(2)
var<storage, read> shArr : array<u32>;

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn readCoef(splat_idx: u32, c_idx: u32) ->
    vec3<f32> {
        let base = splat_idx * splatdim + c_idx * dim;
        let x = readShArr(base);
        let y = readShArr(base + 1u);
        let z = readShArr(base + 2u);
        return vec3<f32>(x,y ,z );
    }

fn norm(clip: vec4<f32>) -> 
    vec3<f32> {
        return clip.xyz / clip.w;
    }

fn readShArr(i: u32) -> 
    f32 {
        let pair = unpack2x16float(shArr[i >> 1u]);
        return select(pair.x, pair.y, (i & 1u) == 1u);
    }

fn readPos(g: Gaussian) -> 
    vec3<f32> {
        let t1 = unpack2x16float(g.pos_opacity[0]);
        let t2 = unpack2x16float(g.pos_opacity[1]);
        return vec3<f32>(t1.x, t1.y, t2.x);
    }

fn getRot(g: Gaussian) -> 
    vec4<f32> {
        let t1 = unpack2x16float(g.rot[0]);
        let t2 = unpack2x16float(g.rot[1]);
        return normalize(vec4<f32>(t1.x, t1.y, t2.x, t2.y));
    }

fn getOpacity(g: Gaussian) -> 
    f32 {
        return unpack2x16float(g.pos_opacity[1]).y;
    }
fn getScale(g: Gaussian) -> 
    vec3<f32> {
        let xy = unpack2x16float(g.scale[0]);
        let z = unpack2x16float(g.scale[1]).x;
        return exp(vec3<f32>(xy.x, xy.y, z)) * renderSettings.gaussian_scaling;
    }


//covariance
fn rotationMatrix(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    return mat3x3<f32>(
        1.-2.*(y*y + z * z), 
        2.*(x*y - r*z), 
        2.*(x*z + r*y),
        2.*(x*y + r*z), 
        1.- 2.* (x*x + z*z), 
        2.*(y*z - r*x),
        2.*(x*z - r*y), 
        2.*(y*z + r*x), 
        1.-2.*(x*x + y*y)
    );
}
fn scaleMatrix(scale: vec3<f32>) -> 
    mat3x3<f32> {
        return mat3x3<f32>(
            scale.x, 0., 0.,
            0., scale.y, 0.,
            0., 0., scale.z
        );
    }
fn calc3dcov(rotation: vec4<f32>, scale: vec3<f32>) -> 
    mat3x3<f32> {
        let M = scaleMatrix(scale) * rotationMatrix(rotation);
        return transpose(M) * M;
    }
fn viewRotation() -> 
    mat3x3<f32> {
        return mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    }
fn calcPerspectiveMat(viewPos: vec3<f32>) -> 
    mat3x3<f32> {
        let scaling = 0.5;
        let clampScale = 1.3;
        let tanx = scaling * camera.viewport.x / camera.focal.x;
        let tany = scaling * camera.viewport.y / camera.focal.y;
        let tz = viewPos.z;
        let tx = clamp(viewPos.x / tz, -clampScale * tanx, clampScale * tanx) * tz;
        let ty = clamp(viewPos.y / tz, -clampScale * tany, clampScale * tany) * tz;

        let r1 = camera.focal.x / tz;
        let r2 = camera.focal.y / tz;
        let r3 = -(camera.focal.x * tx) / (tz * tz);
        let r4 = -(camera.focal.y * ty) / (tz * tz);
        let rv = mat3x3<f32>(
            r1, 0., 0.,
            0., r2, 0.,
            r3, r4, 0.
        );
        return rv;
    }
fn calc2dcov(viewPos: vec3<f32>, cov3d: mat3x3<f32>) -> vec3<f32> {
    let T = calcPerspectiveMat(viewPos) * viewRotation();
    let cov = T * cov3d * transpose(T);
    return vec3<f32>(cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3);
}
fn splatRadius(cov: vec3<f32>) -> f32 {
    let determinant = cov.x * cov.z - cov.y * cov.y;
    let mid = 0.5 * (cov.x + cov.z);
    let largestEigenvalue = mid + sqrt(max(0.1, mid * mid - determinant));
    return ceil(3. * sqrt(largestEigenvalue));
}
fn calcConic(cov: vec3<f32>) -> 
    vec3<f32> {
        let determinant = conicDeterminant(cov);
        return vec3<f32>(cov.z / determinant, 
            -cov.y / determinant, 
            cov.x / determinant);
    }
fn conicDeterminant(cov: vec3<f32>) ->
    f32 {
        return cov.x * cov.z - cov.y * cov.y;
    }
fn calcSize(radius: f32) -> vec2<f32> {
    return 4. * radius / camera.viewport;
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * readCoef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - y * readCoef(v_idx, 1u) * SH_C1 
        + z * readCoef(v_idx, 2u) * SH_C1
        - x * readCoef(v_idx, 3u) * SH_C1;

        if sh_deg > 1u {
            //calculate all the combinations x-z
            let xx = dir.x * dir.x;
            let xy = dir.x * dir.y;
            let xz = dir.x * dir.z;
            let yy = dir.y * dir.y;
            let yz = dir.y * dir.z;
            let zz = dir.z * dir.z;
            
        
            result += 
                SH_C2[0]*readCoef(v_idx, 4u)*xy +
                yz*readCoef(v_idx, 5u)*SH_C2[1]+
                (2.0 * zz - xx - yy)*readCoef(v_idx, 6u)*SH_C2[2] +
                xz*readCoef(v_idx, 7u)*SH_C2[3] +
                (xx - yy)*readCoef(v_idx, 8u)*SH_C2[4];

            let s0 = SH_C3[0];
            let s1 = SH_C3[1];
            let s2 = SH_C3[2];
            let s3 = SH_C3[3];
            let s4 = SH_C3[4];
            let s5 = SH_C3[5];
            let s6 = SH_C3[6];
            if sh_deg > 2u {
                result = result + y*s0*(3.0 * xx - yy) * readCoef(v_idx, 9u) + 
                xy*z*s1*readCoef(v_idx, 10u) + 
                y*s2*(4.0 * zz - xx - yy) * readCoef(v_idx, 11u) + 
                z*s3*(2.0 * zz - 3.0 * xx - 3.0 * yy) * readCoef(v_idx, 12u) + 
                x*s4*(4.0 * zz - xx - yy) * readCoef(v_idx, 13u) + 
                z*s5*(xx - yy) * readCoef(v_idx, 14u) + 
                x*s6*(xx - 3.0 * yy) * readCoef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

// check for view-frustum culling
fn should_be_culled(pos: vec4<f32>) 
-> bool {
    var toBeCulled = true;

    if pos.w > 0. {
        let bounds = pos.xyz / pos.w;
         let threshold = 1.2;
        toBeCulled = abs(bounds.x) > threshold 
        || abs(bounds.y) > threshold
        || bounds.z < 0
        || bounds.z > 1;
    }
    return toBeCulled;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&gaussianArr) { return; }
    let gaussian = gaussianArr[index];
    let worldPos = readPos(gaussian);
    let viewPos = camera.view * vec4<f32>(worldPos, 1.);
    let clipPos = camera.proj * viewPos;

    if should_be_culled(clipPos) == false {
        let cam = norm(clipPos);
        let cov_3 = calc3dcov(getRot(gaussian),
            getScale(gaussian));
        let cov_2 = calc2dcov(viewPos.xyz, cov_3);
        if cov_2.x * cov_2.z - cov_2.y * cov_2.y <= 0. {
            return; 
        }
        let quadSize = calcSize(splatRadius(cov_2));
        let conic = calcConic(cov_2);
        let camPos = camera.view_inv[3].xyz;
        let viewDir = normalize(worldPos - camPos);
        let color = computeColorFromSH(viewDir, index, u32(renderSettings.sh_deg));
        let opacity = 1. / (1. + exp(-getOpacity(gaussian)));
        let slot = atomicAdd(&sort_infos.keys_size, 1u);
        sort_depths[slot] = 0xFFFFFFFFu - bitcast<u32>(viewPos.z);
        splatArr[slot].conic_b = pack2x16float(vec2<f32>(conic.z, 0.));
        splatArr[slot].conic_a = pack2x16float(conic.xy);
        splatArr[slot].center = pack2x16float(cam.xy);
        splatArr[slot].size = pack2x16float(quadSize);
        splatArr[slot].color_a = pack2x16float(color.rg);
        splatArr[slot].color_b = pack2x16float(vec2<f32>(color.b, opacity));

        
        
        sort_indices[slot] = slot;
        let keysToDispatch = workgroupSize * sortKeyPerThread;
        if slot % keysToDispatch == 0u {
            atomicAdd(&sort_dispatch.dispatch_x, 1u);
        }
    }
}