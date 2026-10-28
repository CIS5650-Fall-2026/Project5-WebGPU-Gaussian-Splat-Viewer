enable f16;

// This shader uses f16 for the quaternion, covariance, and projection chains
// SH color, sort depth, and the Splat output remain f32 for stability

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
override sortKeysPerDispatch: u32;

// Must match the tile key layout in tile_composite.wgsl.
const TILE_ID_SHIFT: u32 = 17u;
const DEPTH_KEY_SHIFT: u32 = 32u - TILE_ID_SHIFT;

// Same data structures as in preprocess.wgsl
struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,
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
    culling_enabled: f32,
    point_fraction: f32,
    tile_cols: u32,
    tile_rows: u32,
    tile_size: u32,
    tile_pair_capacity: u32,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    center: vec2<f32>,
    conic: vec3<f32>,
    color: vec4<f32>,
    radius: f32,
    extent: vec2<f32>,
};

// Bindings for the compute shader
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<uniform> render_settings: RenderSettings;
@group(1) @binding(3)
var<storage, read_write> draw_indirect: DrawIndirect;
@group(1) @binding(4)
var<storage, read> sh_coeffs: array<u32>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let coeff_base = splat_idx * 24u;

    let half_index_x = c_idx * 3u;
    let pair_x = unpack2x16float(sh_coeffs[coeff_base + half_index_x / 2u]);
    let value_x = select(pair_x.x, pair_x.y, (half_index_x & 1u) == 1u);

    let half_index_y = c_idx * 3u + 1u;
    let pair_y = unpack2x16float(sh_coeffs[coeff_base + half_index_y / 2u]);
    let value_y = select(pair_y.x, pair_y.y, (half_index_y & 1u) == 1u);

    let half_index_z = c_idx * 3u + 2u;
    let pair_z = unpack2x16float(sh_coeffs[coeff_base + half_index_z / 2u]);
    let value_z = select(pair_z.x, pair_z.y, (half_index_z & 1u) == 1u);

    return vec3<f32>(value_x, value_y, value_z);
}

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

    return max(vec3<f32>(0.0), result);
}

fn hash_u32(value: u32) -> u32 {
    var h = value * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

// f16 version of the preprocess compute shader
@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let sample_value = f32(hash_u32(idx)) / 4294967295.0;
    if (sample_value > clamp(render_settings.point_fraction, 0.0, 1.0)) {
        return;
    }

    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let position = vec4<f32>(a.x, a.y, b.x, 1.0);
    let clip = camera.proj * camera.view * position;

    if (clip.w <= 0.0) {
        return;
    }

    let ndc = clip.xy / clip.w;
    if (render_settings.culling_enabled > 0.5 && (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2)) {
        return;
    }

    let rot01 = unpack2x16float(gaussian.rot[0]);
    let rot23 = unpack2x16float(gaussian.rot[1]);
    var q = vec4<f16>(f16(rot01.x), f16(rot01.y), f16(rot23.x), f16(rot23.y));
    let q_length = length(q);
    if (f32(q_length) < 1e-8) {
        return;
    }
    q /= q_length;

    let scale01 = unpack2x16float(gaussian.scale[0]);
    let scale23 = unpack2x16float(gaussian.scale[1]);
    let scale_modifier = max(render_settings.gaussian_scaling, 0.0);
    let scale_half = vec3<f16>(
        f16(exp(scale01.x)),
        f16(exp(scale01.y)),
        f16(exp(scale23.x)),
    ) * f16(scale_modifier);
    let scale = scale_half;

    let x = q.y;
    let y = q.z;
    let z = q.w;
    let w = q.x;
    let rotation_half = mat3x3<f16>(
        vec3<f16>(
            f16(1.0) - f16(2.0) * (y * y + z * z),
            f16(2.0) * (x * y + w * z),
            f16(2.0) * (x * z - w * y),
        ),
        vec3<f16>(
            f16(2.0) * (x * y - w * z),
            f16(1.0) - f16(2.0) * (x * x + z * z),
            f16(2.0) * (y * z + w * x),
        ),
        vec3<f16>(
            f16(2.0) * (x * z + w * y),
            f16(2.0) * (y * z - w * x),
            f16(1.0) - f16(2.0) * (x * x + y * y),
        ),
    );
    let rotation = rotation_half;
    let scale_squared = mat3x3<f16>(
        vec3<f16>(scale.x * scale.x, f16(0.0), f16(0.0)),
        vec3<f16>(f16(0.0), scale.y * scale.y, f16(0.0)),
        vec3<f16>(f16(0.0), f16(0.0), scale.z * scale.z),
    );
    let covariance_3d = rotation * scale_squared * transpose(rotation);

    let view_linear = mat3x3<f16>(
        vec3<f16>(camera.view[0].xyz),
        vec3<f16>(camera.view[1].xyz),
        vec3<f16>(camera.view[2].xyz),
    );
    let covariance_view = view_linear * covariance_3d * transpose(view_linear);
    let view_position = (camera.view * position).xyz;
    let view_position_half = vec3<f16>(view_position);
    let inv_z = f16(1.0) / view_position_half.z;
    let focal = vec2<f16>(camera.focal);
    let jacobian_x_half = vec3<f16>(
        focal.x * inv_z,
        f16(0.0),
        -focal.x * view_position_half.x * inv_z * inv_z,
    );
    let jacobian_y_half = vec3<f16>(
        f16(0.0),
        focal.y * inv_z,
        -focal.y * view_position_half.y * inv_z * inv_z,
    );
    let cov_a = f32(dot(jacobian_x_half, covariance_view * jacobian_x_half)) + 0.3;
    let cov_b = f32(dot(jacobian_x_half, covariance_view * jacobian_y_half));
    let cov_c = f32(dot(jacobian_y_half, covariance_view * jacobian_y_half)) + 0.3;
    let determinant = cov_a * cov_c - cov_b * cov_b;
    if (determinant <= 1e-8) {
        return;
    }

    let determinant_inv = 1.0 / determinant;
    let conic = vec3<f32>(
        cov_c * determinant_inv,
        -cov_b * determinant_inv,
        cov_a * determinant_inv,
    );

    let mid = 0.5 * (cov_a + cov_c);
    let eigen_delta = sqrt(max(0.25 * (cov_a - cov_c) * (cov_a - cov_c) + cov_b * cov_b, 0.0));
    let lambda_max = mid + eigen_delta;
    let radius = min(
        max(ceil(3.0 * sqrt(max(lambda_max, 0.0))), 1.0),
        max(camera.viewport.x, camera.viewport.y),
    );
    let half_extent = radius * 2.0 / camera.viewport;
    let opacity = 1.0 / (1.0 + exp(-b.y));
    let camera_position = camera.view_inv[3].xyz;
    let view_direction = normalize(position.xyz - camera_position);
    let sh_deg = min(u32(render_settings.sh_deg + 0.5), 3u);
    let color = clamp(computeColorFromSH(view_direction, idx, sh_deg), vec3<f32>(0.0), vec3<f32>(1.0));

    let visible_index = atomicAdd(&draw_indirect.instance_count, 1u);

    splats[visible_index].center = ndc;
    splats[visible_index].conic = conic;
    splats[visible_index].color = vec4<f32>(color, opacity);
    splats[visible_index].radius = radius;
    splats[visible_index].extent = half_extent;

    // Tile based sorting
    let depth_key = bitcast<u32>(clip.w);
    if (render_settings.tile_cols == 0u) {
        let sorted_index = atomicAdd(&sort_infos.keys_size, 1u);
        sort_depths[sorted_index] = 0xFFFFFFFFu - depth_key;
        sort_indices[sorted_index] = visible_index;
        if (sorted_index % sortKeysPerDispatch == 0u) {
            atomicAdd(&sort_dispatch.dispatch_x, 1u);
        }
    } else {
        let center_pixel = vec2<f32>(
            (ndc.x + 1.0) * 0.5 * camera.viewport.x,
            (1.0 - ndc.y) * 0.5 * camera.viewport.y,
        );
        let tile_size = f32(render_settings.tile_size);
        let min_tile_x = u32(clamp(
            floor((center_pixel.x - radius) / tile_size),
            0.0,
            f32(render_settings.tile_cols - 1u),
        ));
        let max_tile_x = u32(clamp(
            floor((center_pixel.x + radius) / tile_size),
            0.0,
            f32(render_settings.tile_cols - 1u),
        ));
        let min_tile_y = u32(clamp(
            floor((center_pixel.y - radius) / tile_size),
            0.0,
            f32(render_settings.tile_rows - 1u),
        ));
        let max_tile_y = u32(clamp(
            floor((center_pixel.y + radius) / tile_size),
            0.0,
            f32(render_settings.tile_rows - 1u),
        ));

        for (var tile_y = min_tile_y; tile_y <= max_tile_y; tile_y += 1u) {
            for (var tile_x = min_tile_x; tile_x <= max_tile_x; tile_x += 1u) {
                // A CAS retry loop serializes millions of reservations on this
                // single counter. Use the native atomic append instead.
                let pair_index = atomicAdd(&sort_infos.keys_size, 1u);
                if (pair_index < render_settings.tile_pair_capacity) {
                    let tile_id = tile_y * render_settings.tile_cols + tile_x;
                    sort_depths[pair_index] =
                        (tile_id << TILE_ID_SHIFT) | (depth_key >> DEPTH_KEY_SHIFT);
                    sort_indices[pair_index] = visible_index;
                    if (pair_index % sortKeysPerDispatch == 0u) {
                        atomicAdd(&sort_dispatch.dispatch_x, 1u);
                    }
                } else {
                    // Publish only the initialized prefix to the radix sorter.
                    atomicMin(&sort_infos.keys_size, render_settings.tile_pair_capacity);
                }
            }
        }
    }
}
