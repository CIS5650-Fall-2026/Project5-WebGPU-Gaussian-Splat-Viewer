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
override sortWorkgroupSize: u32;

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
    //TODO: store information for 2D splat rendering
    center: vec2<f32>,
    half_size_pixels: vec2<f32>,
    conic: vec3<f32>,
    opacity: f32,
    color: vec4<f32>,
};

//TODO: bind your data here

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;


@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> spherical_harmonics: array<u32>;


@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;


// reads one color component from a pair packed inside a u32
fn read_sh_component(component_index: u32) -> f32 {

    // split in 2 because it contains 2 f16
    let packed_components = unpack2x16float(spherical_harmonics[component_index / 2u]);

    // decide if we need .x or .y
    if (component_index % 2u == 0u) {
        return packed_components.x;
    }
    else{
        return packed_components.y;
    }
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(gaussian_index: u32, coefficient_index: u32) -> vec3<f32> { // cganged parameter names for easier understanding
    //TODO: access your binded sh_coeff, see load.ts for how it is stored

    let coefficient_start = gaussian_index * 48u + coefficient_index * 3u;

    return vec3<f32>(read_sh_component(coefficient_start), read_sh_component(coefficient_start + 1u), read_sh_component(coefficient_start + 2u));
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


fn compute_3d_covariance(gaussian: Gaussian) -> mat3x3<f32> {

    // rotation is stored as quaternion w, x, y, z
    let packed_rotation_wx = unpack2x16float(gaussian.rot[0]);
    let packed_rotation_yz = unpack2x16float(gaussian.rot[1]);

    let normalized_rotation = normalize(vec4<f32>(
        packed_rotation_wx.x,
        packed_rotation_wx.y,
        packed_rotation_yz.x,
        packed_rotation_yz.y
    ));

    // just so its easier to understand and write
    let quaternion_w = normalized_rotation.x;
    let quaternion_x = normalized_rotation.y;
    let quaternion_y = normalized_rotation.z;
    let quaternion_z = normalized_rotation.w;

    // columns of the rotation matrix from quaternion
    let rotation_matrix = mat3x3<f32>(
        vec3<f32>(
            1.0 - 2.0 * (quaternion_y * quaternion_y + quaternion_z * quaternion_z),
            2.0 * (quaternion_x * quaternion_y + quaternion_w * quaternion_z),
            2.0 * (quaternion_x * quaternion_z - quaternion_w * quaternion_y)
        ),

        vec3<f32>(
            2.0 * (quaternion_x * quaternion_y - quaternion_w * quaternion_z),
            1.0 - 2.0 * (quaternion_x * quaternion_x + quaternion_z * quaternion_z),
            2.0 * (quaternion_y * quaternion_z + quaternion_w * quaternion_x)
        ),

        vec3<f32>(
            2.0 * (quaternion_x * quaternion_z + quaternion_w * quaternion_y),
            2.0 * (quaternion_y * quaternion_z - quaternion_w * quaternion_x),
            1.0 - 2.0 * (quaternion_x * quaternion_x + quaternion_y * quaternion_y)
        )
    );

    // scale stores in log space, so exp gives actual scale
    let packed_scale_xy = unpack2x16float(gaussian.scale[0]);
    let packed_scale_z = unpack2x16float(gaussian.scale[1]);

    let gaussian_scale = exp(vec3<f32>(packed_scale_xy.x, packed_scale_xy.y, packed_scale_z.x)) * render_settings.gaussian_scaling;

    // diagonal scale matrix
    let scale_matrix = mat3x3<f32>(
        vec3<f32>(gaussian_scale.x, 0.0, 0.0),
        vec3<f32>(0.0, gaussian_scale.y, 0.0),
        vec3<f32>(0.0, 0.0, gaussian_scale.z)
    );

    let rotation_scale_matrix = rotation_matrix * scale_matrix;
    return rotation_scale_matrix * transpose(rotation_scale_matrix);
}


fn compute_2d_covariance(
    world_position: vec4<f32>,
    covariance_3d: mat3x3<f32>
) -> vec3<f32> {


    let view_position = (camera.view * world_position).xyz;

    // so its less messy in the Jacobian
    let inverse_depth = 1.0 / view_position.z;
    let inverse_depth_squared = inverse_depth * inverse_depth;

    // derivative of perspective projection from camera space to pixel space
    let projection_jacobian = mat3x3<f32>(
        vec3<f32>(camera.focal.x * inverse_depth, 0.0, 0.0),
        vec3<f32>(0.0, camera.focal.y * inverse_depth, 0.0),
        vec3<f32>(-camera.focal.x * view_position.x * inverse_depth_squared, -camera.focal.y * view_position.y * inverse_depth_squared, 0.0)
    );

    // get the rotation part
    let view_rotation = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );

    let world_to_pixel_matrix = projection_jacobian * view_rotation;

    let covariance_in_pixels = world_to_pixel_matrix * covariance_3d * transpose(world_to_pixel_matrix);



    // minimum pixel footprint prevents very small gaussians from just disappearing
    return vec3<f32>(
        covariance_in_pixels[0][0] + 0.3,
        covariance_in_pixels[1][0],
        covariance_in_pixels[1][1] + 0.3
    );
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    // extra threads leave
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // packed f16 values become xyz and opacity
    let gaussian = gaussians[idx];
    let xy = unpack2x16float(gaussian.pos_opacity[0]);
    let z_opacity = unpack2x16float(gaussian.pos_opacity[1]);
    let world_pos = vec4<f32>(xy.x, xy.y, z_opacity.x, 1.0);

    // world position to screen position
    let clip_pos = camera.proj * camera.view * world_pos;
    if (clip_pos.w <= 0.0) {
        return;
    }
    let ndc_pos = clip_pos.xyz / clip_pos.w;

    // slightly larger box to keep splats near edges
    if (abs(ndc_pos.x) > 1.2 || abs(ndc_pos.y) > 1.2 || ndc_pos.z < 0.0 || ndc_pos.z > 1.0) {
        return;
    }


    let covariance_3d = compute_3d_covariance(gaussian);
    let covariance_2d = compute_2d_covariance(world_pos, covariance_3d);

    let covariance_determinant = covariance_2d.x * covariance_2d.z - covariance_2d.y * covariance_2d.y;

    // if our covariance is bad we throw it away
    if (covariance_determinant <= 0.0) {
        return;
    }

    // we will need for eigenvalue computations
    let half_trace = 0.5 * (covariance_2d.x + covariance_2d.z);

    let eigenvalue_difference = sqrt(max(0.0, half_trace * half_trace - covariance_determinant));


    let first_eigenvalue = half_trace + eigenvalue_difference;
    let second_eigenvalue = half_trace - eigenvalue_difference;
    let largest_eigenvalue = max(first_eigenvalue, second_eigenvalue);

    // three standard deviations contain almost the entire gaussian
    let gaussian_radius_pixels = max(1.0, ceil(3.0 * sqrt(max(largest_eigenvalue, 0.0))));


    // each visible gaussian gets the next free splat slot
    let visible_splat_index = atomicAdd(&sort_infos.keys_size, 1u);


    splats[visible_splat_index].center = ndc_pos.xy;

    splats[visible_splat_index].half_size_pixels = vec2<f32>(gaussian_radius_pixels, gaussian_radius_pixels);


    // SH color depends on the direction from the camera to this gaussian
    let camera_world_position = camera.view_inv[3].xyz;

    // see where camera is looking and normalize it
    let gaussian_view_direction = normalize(world_pos.xyz - camera_world_position);

    let gaussian_color = computeColorFromSH(
        gaussian_view_direction,
        idx,
        u32(render_settings.sh_deg)
    );


    splats[visible_splat_index].color = vec4<f32>(
        gaussian_color,
        1.0 // for now we will keep like this
    );


    // sigmoid for opacity
    splats[visible_splat_index].opacity = 1.0 / (1.0 + exp(-z_opacity.y));


    // inverse covariance is the conic used later by the fragment shader
    splats[visible_splat_index].conic = vec3<f32>(covariance_2d.z / covariance_determinant, -covariance_2d.y / covariance_determinant, covariance_2d.x / covariance_determinant);


    // because radix sort uses ascending keys far splats need smaller values
    let back_to_front_depth = 1.0 - ndc_pos.z;


    sort_depths[visible_splat_index] = bitcast<u32>(back_to_front_depth);
    sort_indices[visible_splat_index] = visible_splat_index;


    let keys_per_dispatch = sortWorkgroupSize * sortKeyPerThread;

    // each sorting workgroup can process keys_per_dispatch splats
    if (visible_splat_index % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}