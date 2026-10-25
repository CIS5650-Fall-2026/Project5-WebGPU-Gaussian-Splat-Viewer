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


// Maximum spherical harmonics degree available in the loaded Gaussian data
override shDegree: u32;

// Benchmark toggle: 1 enables frustum culling, 0 disables it
override enableFrustumCulling: u32 = 1u;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: atomic<u32>, 
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


// User-adjustable Gaussian rendering parameters
@group(0) @binding(4)
var<uniform> render_settings: RenderSettings;




struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering

    // Projected Gaussian center in normalized device coordinates
    center: vec2<f32>,

    // Inverse projected covariance, stored as (A, B, C)
    // Represents the symmetric matrix:
    // [A B]
    // [B C]
    conic: vec3<f32>,

    // Decoded per-Gaussian opacity in the range [0, 1]
    opacity: f32,

    // RGB color evaluated from spherical harmonics
    color: vec3<f32>,

};

//TODO: bind your data here


// Spherical harmonics coefficients packed as half-precision RGB values
// Each Gaussian stores 16 RGB coefficients (96 bytes total); Each u32 contains two consecutive f16 coefficient values
@group(0) @binding(3)
var<storage, read> sh_coeffs: array<u32>;




// Input Gaussians packed as 16-bit floating-point values
// Each Gaussian occupies six u32 words (24 bytes)
@group(0) @binding(0)
var<storage, read> gaussians: array<Gaussian>;



// Camera matrices used to transform world-space Gaussian centers
@group(0) @binding(1)
var<uniform> camera: CameraUniforms;




// Projected 2D splats (center, conic, opacity, color) written by the compute shader
@group(0) @binding(2)
var<storage, read_write> splats: array<Splat>;
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored



    // Each Gaussian contains 16 RGB coefficients, or 48 f16 values
    // Coefficients are stored in RGB-interleaved order
    let base = splat_idx * 48u + c_idx * 3u;




    // Each u32 contains two consecutive f16 values
    let packed_0 = unpack2x16float(sh_coeffs[base / 2u]);
    let packed_1 = unpack2x16float(sh_coeffs[(base + 2u) / 2u]);






    // Every coefficient begins at an alternating f16 alignment
    if ((base % 2u) == 0u) {


        return vec3<f32>(packed_0.x, packed_0.y, packed_1.x);
    }






    return vec3<f32>(
        packed_0.y,
        packed_1.x,
        packed_1.y
    );


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
    //TODO: set up pipeline as described in instruction


    // Prevent invocations in the final workgroup from reading beyond the allocated Gaussian input buffer
    if (idx >= arrayLength(&gaussians)) {
        return;
    }


    // Unpack the three half-precision position coordinates
    // pos_opacity contains [x, y] and [z, opacity]
    let xy = unpack2x16float(gaussians[idx].pos_opacity[0]);
    let zo = unpack2x16float(gaussians[idx].pos_opacity[1]);

    let world_pos = vec4<f32>(xy.x, xy.y, zo.x, 1.0);


    // Extract the camera position in world coordinates from the inverse view matrix
    let camera_world_pos = camera.view_inv[3].xyz;




    // Compute the viewing direction from the camera toward this Gaussian
    let view_direction = normalize(world_pos.xyz - camera_world_pos);




    // Extract the packed opacity logit from the fourth half-precision component
    let opacity_raw = zo.y;

    // Convert the opacity logit into a value between 0 and 1
    // Standard 3D Gaussian Splatting stores opacity in logit space
    let opacity = 1.0 / (1.0 + exp(-opacity_raw));








    // Unpack the Gaussian's quaternion rotation
    // Each u32 contains two packed half-precision floating-point values
    // The quaternion is stored in (w, x, y, z) order
    let rot_01 = unpack2x16float(gaussians[idx].rot[0]);
    let rot_23 = unpack2x16float(gaussians[idx].rot[1]);



    let rotation = normalize(vec4<f32>(rot_01.x, rot_01.y, rot_23.x, rot_23.y));





    // Unpack the Gaussian's three log-scale components
    // load.ts stores the original PLY values without exponentiation
    let scale_01 = unpack2x16float(gaussians[idx].scale[0]);
    let scale_2 = unpack2x16float(gaussians[idx].scale[1]);

    // Standard 3D Gaussian Splatting stores logarithmic scales
    // Convert them into positive standard deviations for covariance
    let scale = exp(vec3<f32>(
        scale_01.x,
        scale_01.y,
        scale_2.x
    )) * render_settings.gaussian_scaling;



    // Convert the normalized (w, x, y, z) quaternion into a 3D rotation matrix
    let w = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;

    // WGSL matrices are constructed column by column
    let R = mat3x3<f32>(
        vec3<f32>(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y)
        ),
        vec3<f32>(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x)
        ),
        vec3<f32>(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y)
        )
    );


    // Construct the 3D covariance matrix from rotation and scale
    // Each scale represents a standard deviation along a local Gaussian axis
    // Sigma_3D = R * diag(scale^2) * transpose(R)

    // Scale the columns of R to form the transformation M = R * S
    let M = mat3x3<f32>(
        R[0] * scale.x,
        R[1] * scale.y,
        R[2] * scale.z
    );

    // Construct the symmetric positive-semidefinite covariance matrix
    let covariance_3d = M * transpose(M);



    // Extract the linear transformation from the camera view matrix
    // WGSL matrices are column-major, so the first three columns contain the 3D linear transformation without translation
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );






    // Transform the world-space Gaussian covariance into camera space
    // Sigma_camera = W * Sigma_3D * transpose(W)
    let covariance_camera = W * covariance_3d * transpose(W);


    // Transform the Gaussian center into camera coordinates
    // The perspective-projection Jacobian depends on camera-space position
    let camera_pos = camera.view * world_pos;



    // Construct the Jacobian of perspective projection in normalized device coordinates
    // The camera looks along positive Z, and its projection matrix flips the Y axis
    // Focal lengths are expressed in pixels, so divide by the viewport dimensions
    // x_ndc = (2 * fx / width) * X / Z
    // y_ndc = -(2 * fy / height) * Y / Z
    // J is a 2x3 matrix that maps small camera-space displacements into normalized device coordinate displacements

    let X = camera_pos.x;
    let Y = camera_pos.y;

    // Avoid division by zero for Gaussians near the camera plane
    // The frustum test below will reject centers behind the camera
    let Z = max(camera_pos.z, 0.00001);




    let fx_ndc = 2.0 * camera.focal.x / camera.viewport.x;
    let fy_ndc = 2.0 * camera.focal.y / camera.viewport.y;





    let inv_Z = 1.0 / Z;
    let inv_Z2 = inv_Z * inv_Z;





    // WGSL matrices are specified column by column
    // This mat3x2 has three columns and two rows
    let J = mat3x2<f32>(
        vec2<f32>(fx_ndc * inv_Z, 0.0),
        vec2<f32>(0.0, -fy_ndc * inv_Z),
        vec2<f32>(
            -fx_ndc * X * inv_Z2,
            fy_ndc * Y * inv_Z2
        )
    );


    // Project the camera-space 3D covariance into normalized device coordinates
    // Sigma_2D = J * Sigma_camera * transpose(J)
    // Matrix dimensions:
    // J: 2x3
    // covariance_camera: 3x3
    // transpose(J): 3x2
    // covariance_2d: 2x2
    let covariance_2d = J * covariance_camera * transpose(J);






    // Regularize the projected covariance to improve numerical stability
    // A half-pixel standard deviation is converted into NDC units because covariance_2d is expressed in NDC squared
    let min_sigma_ndc = vec2<f32>(1.0 / camera.viewport.x, 1.0 / camera.viewport.y);





    // Add a small positive variance to each diagonal entry
    // This helps prevent extremely small projected Gaussians from producing nearly singular covariance matrices
    let covariance_2d_stable = covariance_2d + mat2x2<f32>(
        vec2<f32>(min_sigma_ndc.x * min_sigma_ndc.x, 0.0),
        vec2<f32>(0.0, min_sigma_ndc.y * min_sigma_ndc.y)
    );






    // Extract the three independent entries of the symmetric 2D covariance
    // WGSL matrix indexing uses [column][row]
    let a = covariance_2d_stable[0][0];

    let b = 0.5 * (covariance_2d_stable[0][1] + covariance_2d_stable[1][0]);

    let c = covariance_2d_stable[1][1];





    // Compute the determinant of the symmetric 2D covariance
    // det(Sigma) = a*c - b*b
    let det = a * c - b * b;


    // Reject non-positive determinants before inversion
    // The positive diagonal regularization already provides a minimum variance
    // Avoid a fixed NDC determinant threshold that depends on resolution
    let valid_covariance = det > 0.0;



    // Compute the inverse covariance coefficients (the conic)
    // Invalid covariance matrices keep a zero conic; such Gaussians are rejected by valid_projection below
    var conic = vec3<f32>(0.0, 0.0, 0.0);

    if (valid_covariance) {
        let inv_det = 1.0 / det;

        // Inverse covariance:
        // [conic.x  conic.y]
        // [conic.y  conic.z]
        conic = vec3<f32>(
            c * inv_det,
            -b * inv_det,
            a * inv_det
        );
    }







    // Compute the eigenvalues of the symmetric projected covariance
    // The largest eigenvalue determines the maximum principal-axis variance
    let half_trace = 0.5 * (a + c);


    let eigen_discriminant = sqrt(max(0.0, 0.25 * (a - c) * (a - c) + b * b));



    let lambda_max = max(0.0, half_trace + eigen_discriminant);



    // Use a three-standard-deviation bounding radius in NDC units
    // This is a conservative circular bound for the projected Gaussian
    let radius_ndc = 3.0 * sqrt(lambda_max);








    // Transform the Gaussian center into clip space
    let clip_pos = camera.proj * camera.view * world_pos;





    // Compute the projected Gaussian center in NDC
    // This will be used with radius_ndc for extent-aware visibility testing
    // The positive-Z camera requires clip_pos.w > 0 for valid projection
    var center_ndc = vec2<f32>(10.0, 10.0);

    if (clip_pos.w > 0.00001) {
        center_ndc = clip_pos.xy / clip_pos.w;
    }






    // Test whether the projected Gaussian footprint overlaps the viewport
    // A three-sigma circular bound conservatively encloses the Gaussian ellipse
    // Keep the clip-space depth test to reject centers outside the near/far range
    let gaussian_in_frustum =
        clip_pos.w > 0.00001 &&
        clip_pos.z >= 0.0 &&
        clip_pos.z <= clip_pos.w &&
        valid_covariance &&
        abs(center_ndc.x) <= 1.0 + radius_ndc &&
        abs(center_ndc.y) <= 1.0 + radius_ndc;




    // Validity checks that always apply, independent of the frustum-culling toggle: the center must be in front of the camera, within the near/far depth range, and have an invertible covariance
    // When enableFrustumCulling is 1, gaussian_in_frustum additionally requires the three-sigma footprint to overlap the viewport
    let valid_projection =
    clip_pos.w > 0.00001 &&
    clip_pos.z >= 0.0 &&
    clip_pos.z <= clip_pos.w &&
    valid_covariance;

    if (valid_projection && (enableFrustumCulling == 0u || gaussian_in_frustum)) {


        // Encode positive camera-space depth as an unsigned sorting key
        // Reversing its bits makes farther Gaussians sort before nearer ones
        let depth_key = ~bitcast<u32>(camera_pos.z);


        // Reserve a unique position in the compacted visible-Gaussian list
        let sort_idx = atomicAdd(&sort_infos.keys_size, 1u);




        // Store the depth key at its compacted position
        sort_depths[sort_idx] = depth_key;




        // Preserve the original Gaussian index for rendering after sorting
        sort_indices[sort_idx] = idx;



        // Increase the sorting dispatch count once per block of visible Gaussians
        let keys_per_dispatch = 256u * sortKeyPerThread;

        if (sort_idx % keys_per_dispatch == 0u) {
            atomicAdd(&sort_dispatch.dispatch_x, 1u);
        }



        let required_padded_size = ((sort_idx / keys_per_dispatch) + 1u) * keys_per_dispatch;



        atomicMax(&sort_infos.padded_size, required_padded_size);



        // Store the projected center of each potentially visible Gaussian
        splats[idx].center = center_ndc;

        // Store the inverse projected covariance for Gaussian evaluation
        splats[idx].conic = conic;

        // Store the decoded opacity for fragment-stage Gaussian blending
        splats[idx].opacity = opacity;

        // Evaluate view-dependent RGB color using the spherical harmonics degree supplied by the loaded Gaussian data through the pipeline override
        splats[idx].color = computeColorFromSH(view_direction, idx, shDegree);




    } else {
        // Rejected Gaussians are not appended to the sort list, so they are not drawn
        // Clear their splat entry so the buffer never holds stale data
        splats[idx].center = vec2<f32>(10.0, 10.0);



        // Initialize the conic for invisible Gaussians
        splats[idx].conic = vec3<f32>(0.0, 0.0, 0.0);

        // Invisible Gaussians contribute no opacity
        splats[idx].opacity = 0.0;


        // Initialize the color of invisible Gaussians
        splats[idx].color = vec3<f32>(0.0, 0.0, 0.0);
    }















    
}