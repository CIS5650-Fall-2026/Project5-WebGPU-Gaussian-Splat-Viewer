// Entry points of the preprocess step. Relies on `project_gaussian` from the selected math backend.

// Large scenes need more than 65535 workgroups for small workgroup sizes, so the host may
// dispatch a 2D grid of workgroups.
fn gaussian_index(gid: vec3<u32>, wgs: vec3<u32>) -> u32 {
    return gid.x + gid.y * wgs.x * workgroupSize;
}

fn tile_grid() -> vec2<u32> {
    return (vec2<u32>(camera.viewport) + tileSize - 1u) / tileSize;
}

// number of high key bits used for the tile id, the remaining bits hold the depth
fn tile_key_bits(tiles: vec2<u32>) -> u32 {
    let num_tiles = tiles.x * tiles.y;
    if num_tiles <= 1u {
        return 1u;
    }
    return firstLeadingBit(num_tiles - 1u) + 1u;
}

// ---------------------------------------------------------------------------------------------
// Quad renderer: one sort key (inverted view depth => back to front) per visible gaussian
// ---------------------------------------------------------------------------------------------
@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gaussian_index(gid, wgs);
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let p = project_gaussian(idx);
    if !p.visible {
        return;
    }

    let i = atomicAdd(&sort_infos.keys_size, 1u);
    splats[i] = make_splat(p);
    // positive floats keep their order when reinterpreted as u32; inverting sorts far to near
    sort_depths[i] = ~bitcast<u32>(p.view_z);
    sort_indices[i] = i;

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if i % sortKeysPerDispatch == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}

// ---------------------------------------------------------------------------------------------
// Tile renderer: one key per (tile, gaussian) pair. key = [tile id | view depth], so after the
// sort every tile owns a contiguous, front to back ordered range of gaussians.
// ---------------------------------------------------------------------------------------------
@compute @workgroup_size(workgroupSize,1,1)
fn preprocess_tiles(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gaussian_index(gid, wgs);
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let p = project_gaussian(idx);
    if !p.visible {
        return;
    }

    let tiles = tile_grid();
    let center = ndc_to_pixel(p.ndc);
    let tile_f = f32(tileSize);
    let tiles_f = vec2<f32>(tiles);
    let rect_min = vec2<u32>(clamp(floor((center - p.radius) / tile_f), vec2<f32>(0.0), tiles_f));
    let rect_max = vec2<u32>(clamp(ceil((center + p.radius) / tile_f), vec2<f32>(0.0), tiles_f));
    let extent = rect_max - rect_min;
    let num_keys = extent.x * extent.y;
    if num_keys == 0u {
        return;
    }

    let s = atomicAdd(&tile_counters.splat_count, 1u);
    splats[s] = make_splat(p);

    // reserve a contiguous block of keys; their order does not matter since they get sorted
    let capacity = u32(render_settings.key_capacity);
    var k = atomicAdd(&sort_infos.keys_size, num_keys);

    let tile_bits = tile_key_bits(tiles);
    // drop the (always zero) sign bit and keep the top bits of the float => log-like depth quantization
    let depth_key = (bitcast<u32>(p.view_z) << 1u) >> tile_bits;
    for (var y = rect_min.y; y < rect_max.y; y++) {
        for (var x = rect_min.x; x < rect_max.x; x++) {
            if k >= capacity {
                return;
            }
            sort_depths[k] = ((y * tiles.x + x) << (32u - tile_bits)) | depth_key;
            sort_indices[k] = s;
            k++;
        }
    }
}

// clamps the key count to the sort buffer capacity and sets up the indirect dispatches of the
// sort and of the tile range identification (256 keys per workgroup, grid-stride beyond 65535 groups)
@compute @workgroup_size(1)
fn finalize_tiles() {
    let capacity = u32(render_settings.key_capacity);
    let requested = atomicLoad(&sort_infos.keys_size);
    let n = min(requested, capacity);
    atomicStore(&sort_infos.keys_size, n);
    atomicStore(&sort_dispatch.dispatch_x, (n + sortKeysPerDispatch - 1u) / sortKeysPerDispatch);
    tile_counters.range_dispatch_x = min((n + 255u) / 256u, 65535u);
    tile_counters.range_dispatch_y = 1u;
    tile_counters.range_dispatch_z = 1u;
    tile_counters.requested_keys = requested;
}
