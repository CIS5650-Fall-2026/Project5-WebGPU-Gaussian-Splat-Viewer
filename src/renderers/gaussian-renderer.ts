import { PointCloud } from '../utils/load';
import preprocessCommonWGSL from '../shaders/preprocess_common.wgsl';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import preprocessF16WGSL from '../shaders/preprocess_f16.wgsl';
import preprocessEntriesWGSL from '../shaders/preprocess_entries.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import tileRasterWGSL from '../shaders/tile_raster.wgsl';
import blitWGSL from '../shaders/blit.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { GpuStats } from '../utils/gpu-stats';

export type GaussianRenderMode = 'quad' | 'tile';
export type PreprocessPrecision = 'f32' | 'f16';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
  set_frustum_culling: (enabled: boolean) => void,
  set_tight_bounds: (enabled: boolean) => void,
  set_workgroup_size: (size: number) => void,
  set_mode: (mode: GaussianRenderMode) => void,
  // returns false when f16 is requested but the device lacks `shader-f16`
  set_precision: (precision: PreprocessPrecision) => boolean,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

// keep in sync with `Splat` in the shaders
const c_size_splat = 8 * 4;
const c_size_render_settings = 8 * 4;
const c_max_workgroups_per_dim = 65535;
const c_tile_size = 16;
const c_sort_keys_per_dispatch = C.histogram_wg_size * c_histogram_block_rows;
// upper bound of (tile, gaussian) keys for the tile renderer: 16M keys ~ 256 MB of sort buffers
const c_max_tile_keys = 1 << 24;
const c_tile_keys_per_gaussian = 8;

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  canvas: HTMLCanvasElement,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  let mode: GaussianRenderMode = 'quad';

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const splat_buffer = createBuffer(
    device, 'splats', pc.num_points * c_size_splat, GPUBufferUsage.STORAGE,
  );

  // vertex_count, instance_count, first_vertex, first_instance
  const indirect_draw_buffer = createBuffer(
    device, 'indirect draw', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([4, pc.num_points, 0, 0]),
  );

  // gaussian_scaling, sh_deg, frustum_culling, key_capacity, tight_bounds, padding
  const render_settings = new Float32Array([1.0, pc.sh_deg, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
  const render_settings_buffer = createBuffer(
    device, 'render settings', c_size_render_settings,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings,
  );
  const update_render_settings = () => device.queue.writeBuffer(render_settings_buffer, 0, render_settings);

  // only used by the tile renderer, but part of the shared preprocess layout
  // splat_count, range dispatch (x, y, z), requested_keys
  const tile_counters_buffer = createBuffer(
    device, 'tile counters', 5 * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const buffer_entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
  });

  // explicit layouts so the bind groups can be shared by all preprocess entry points
  const preprocess_layouts = [
    device.createBindGroupLayout({
      label: 'preprocess camera + settings',
      entries: [buffer_entry(0, 'uniform'), buffer_entry(1, 'uniform')],
    }),
    device.createBindGroupLayout({
      label: 'preprocess gaussians',
      entries: [buffer_entry(0, 'read-only-storage'), buffer_entry(1, 'read-only-storage'), buffer_entry(2, 'storage')],
    }),
    device.createBindGroupLayout({
      label: 'preprocess sort',
      entries: [0, 1, 2, 3, 4].map(i => buffer_entry(i, 'storage')),
    }),
  ];
  const preprocess_pipeline_layout = device.createPipelineLayout({
    label: 'preprocess',
    bindGroupLayouts: preprocess_layouts,
  });

  const create_preprocess_module = (precision: PreprocessPrecision) => device.createShaderModule({
    label: `preprocess ${precision}`,
    code: precision === 'f16'
      ? 'enable f16;\n' + preprocessCommonWGSL + preprocessF16WGSL + preprocessEntriesWGSL
      : preprocessCommonWGSL + preprocessWGSL + preprocessEntriesWGSL,
  });
  let preprocess_module = create_preprocess_module('f32');

  let workgroup_size = C.histogram_wg_size;
  const create_preprocess_pipeline = (entryPoint: string) => device.createComputePipeline({
    label: `preprocess ${entryPoint}`,
    layout: preprocess_pipeline_layout,
    compute: {
      module: preprocess_module,
      entryPoint,
      constants: {
        workgroupSize: workgroup_size,
        sortKeysPerDispatch: c_sort_keys_per_dispatch,
        tileSize: c_tile_size,
      },
    },
  });
  let preprocess_pipeline = create_preprocess_pipeline('preprocess');

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_layouts[0],
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_layouts[1],
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
    ],
  });

  const create_sort_bind_group = (s: ReturnType<typeof get_sorter>) => device.createBindGroup({
    label: 'sort',
    layout: preprocess_layouts[2],
    entries: [
      { binding: 0, resource: { buffer: s.sort_info_buffer } },
      { binding: 1, resource: { buffer: s.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: s.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: s.sort_dispatch_indirect_buffer } },
      { binding: 4, resource: { buffer: tile_counters_buffer } },
    ],
  });
  const sort_bind_group = create_sort_bind_group(sorter);


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ label: 'gaussian render', code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        // splats are drawn back to front => standard "over" blending
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      // sorting ends after an even number of passes => the result is back in ping_pong[0]
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Tile Renderer Resources (created on demand)
  // ===============================================
  type TileResources = ReturnType<typeof create_tile_resources>;
  let tile: TileResources | undefined;

  function create_tile_resources() {
    const max_keys_by_limit = Math.floor(device.limits.maxStorageBufferBindingSize / 4) - 2 * c_sort_keys_per_dispatch;
    const capacity = Math.max(1, Math.min(pc.num_points * c_tile_keys_per_gaussian, c_max_tile_keys, max_keys_by_limit));
    const tile_sorter = get_sorter(capacity, device);
    render_settings[3] = capacity;
    update_render_settings();

    const max_tiles = Math.ceil(device.limits.maxTextureDimension2D / c_tile_size) ** 2;
    const tile_ranges_buffer = createBuffer(
      device, 'tile ranges', max_tiles * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );

    const raster_module = device.createShaderModule({ label: 'tile raster', code: tileRasterWGSL });
    const ranges_pipeline = device.createComputePipeline({
      label: 'tile ranges',
      layout: 'auto',
      compute: { module: raster_module, entryPoint: 'identify_ranges' },
    });
    const raster_pipeline = device.createComputePipeline({
      label: 'tile rasterize',
      layout: 'auto',
      compute: { module: raster_module, entryPoint: 'rasterize' },
    });
    const ranges_bind_group = device.createBindGroup({
      label: 'tile ranges',
      layout: ranges_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: tile_sorter.sort_info_buffer } },
        { binding: 2, resource: { buffer: tile_sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 3, resource: { buffer: tile_ranges_buffer } },
      ],
    });

    const blit_module = device.createShaderModule({ label: 'blit', code: blitWGSL });
    const blit_pipeline = device.createRenderPipeline({
      label: 'blit',
      layout: 'auto',
      vertex: { module: blit_module, entryPoint: 'vs_main' },
      fragment: { module: blit_module, entryPoint: 'fs_main', targets: [{ format: presentation_format }] },
      primitive: { topology: 'triangle-list' },
    });

    // output texture and the bind groups using it follow the canvas size
    let output: GPUTexture | undefined;
    let raster_bind_group: GPUBindGroup;
    let blit_bind_group: GPUBindGroup;
    const ensure_output = (width: number, height: number) => {
      if (output && output.width === width && output.height === height) return;
      output?.destroy();
      output = device.createTexture({
        label: 'tile raster output',
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      const view = output.createView();
      raster_bind_group = device.createBindGroup({
        label: 'tile rasterize',
        layout: raster_pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: camera_buffer } },
          { binding: 3, resource: { buffer: tile_ranges_buffer } },
          { binding: 4, resource: { buffer: splat_buffer } },
          { binding: 5, resource: { buffer: tile_sorter.ping_pong[0].sort_indices_buffer } },
          { binding: 6, resource: view },
        ],
      });
      blit_bind_group = device.createBindGroup({
        label: 'blit',
        layout: blit_pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: view }],
      });
    };

    return {
      capacity,
      sorter: tile_sorter,
      sort_bind_group: create_sort_bind_group(tile_sorter),
      preprocess_pipeline: create_preprocess_pipeline('preprocess_tiles'),
      finalize_pipeline: create_preprocess_pipeline('finalize_tiles'),
      tile_ranges_buffer,
      ranges_pipeline,
      ranges_bind_group,
      raster_pipeline,
      blit_pipeline,
      ensure_output,
      raster_bind_group: () => raster_bind_group,
      blit_bind_group: () => blit_bind_group,
    };
  }

  // ===============================================
  //    Profiling
  // ===============================================
  // timestamps: 0 preprocess begin, 1 preprocess end, 2 raster begin, 3 raster end, 4 blit begin, 5 blit end
  const stats: Record<GaussianRenderMode, GpuStats> = {
    // counters: visible gaussians
    quad: new GpuStats(device, 4, [[0, 1], [1, 2], [2, 3], [0, 3]], 1),
    // counters: visible gaussians, requested keys, sorted keys
    tile: new GpuStats(device, 6, [[0, 1], [1, 2], [2, 3], [4, 5], [0, 5]], 3),
  };

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const dispatch_preprocess = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, sort_group: GPUBindGroup) => {
    const num_workgroups = Math.ceil(pc.num_points / workgroup_size);
    const x = Math.min(num_workgroups, c_max_workgroups_per_dim);
    const y = Math.ceil(num_workgroups / x);

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_group);
    pass.dispatchWorkgroups(x, y);
  };

  const frame_quad = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const s = stats.quad;

    // reset the visible gaussian count and the sort dispatch size
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

    const pre = encoder.beginComputePass({ label: 'preprocess', timestampWrites: s.writes(0, 1) });
    dispatch_preprocess(pre, preprocess_pipeline, sort_bind_group);
    pre.end();
    s.copy_counter(encoder, 0, sorter.sort_info_buffer);

    sorter.sort(encoder);

    // instance count = number of visible gaussians written by the preprocess step
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store',
        }
      ],
      timestampWrites: s.writes(2, 3),
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();

    s.resolve(encoder);
  };

  const frame_tile = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    tile ??= create_tile_resources();
    const t = tile;
    const s = stats.tile;
    const width = canvas.width;
    const height = canvas.height;
    t.ensure_output(width, height);

    device.queue.writeBuffer(t.sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(t.sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
    device.queue.writeBuffer(tile_counters_buffer, 0, nulling_data);
    encoder.clearBuffer(t.tile_ranges_buffer, 0, Math.ceil(width / c_tile_size) * Math.ceil(height / c_tile_size) * 8);

    // duplicate every gaussian into one key per overlapped tile
    const pre = encoder.beginComputePass({ label: 'preprocess tiles', timestampWrites: s.writes(0, 1) });
    dispatch_preprocess(pre, t.preprocess_pipeline, t.sort_bind_group);
    pre.setPipeline(t.finalize_pipeline);
    pre.dispatchWorkgroups(1);
    pre.end();
    s.copy_counter(encoder, 0, tile_counters_buffer);
    s.copy_counter(encoder, 1, tile_counters_buffer, 16);
    s.copy_counter(encoder, 2, t.sorter.sort_info_buffer);

    t.sorter.sort(encoder);

    const raster = encoder.beginComputePass({ label: 'tile raster', timestampWrites: s.writes(2, 3) });
    raster.setPipeline(t.ranges_pipeline);
    raster.setBindGroup(0, t.ranges_bind_group);
    raster.dispatchWorkgroupsIndirect(tile_counters_buffer, 4);
    raster.setPipeline(t.raster_pipeline);
    raster.setBindGroup(0, t.raster_bind_group());
    raster.dispatchWorkgroups(Math.ceil(width / c_tile_size), Math.ceil(height / c_tile_size));
    raster.end();

    const blit = encoder.beginRenderPass({
      label: 'blit',
      colorAttachments: [{ view: texture_view, loadOp: 'clear', storeOp: 'store' }],
      timestampWrites: s.writes(4, 5),
    });
    blit.setPipeline(t.blit_pipeline);
    blit.setBindGroup(0, t.blit_bind_group());
    blit.draw(3);
    blit.end();

    s.resolve(encoder);
  };

  const rebuild_preprocess_pipelines = () => {
    preprocess_pipeline = create_preprocess_pipeline('preprocess');
    if (tile) {
      tile.preprocess_pipeline = create_preprocess_pipeline('preprocess_tiles');
      tile.finalize_pipeline = create_preprocess_pipeline('finalize_tiles');
    }
  };

  const fmt = (ms: number) => ms.toFixed(3).padStart(7);

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      if (mode === 'tile') {
        frame_tile(encoder, texture_view);
      } else {
        frame_quad(encoder, texture_view);
      }
    },
    camera_buffer,

    after_submit: () => stats[mode].after_submit(),
    stats_text: () => {
      const s = stats[mode];
      const d = s.durations_ms;
      if (mode === 'quad') {
        return [
          `visible     ${s.counters[0]} / ${pc.num_points}`,
          `preprocess ${fmt(d[0])} ms`,
          `sort       ${fmt(d[1])} ms`,
          `render     ${fmt(d[2])} ms`,
          `gpu total  ${fmt(d[3])} ms`,
        ].join('\n');
      }
      const overflow = s.counters[1] > s.counters[2] ? ' (overflow!)' : '';
      return [
        `visible     ${s.counters[0]} / ${pc.num_points}`,
        `tile keys   ${s.counters[2]} / ${tile?.capacity ?? 0}${overflow}`,
        `preprocess ${fmt(d[0])} ms`,
        `sort       ${fmt(d[1])} ms`,
        `rasterize  ${fmt(d[2])} ms`,
        `blit       ${fmt(d[3])} ms`,
        `gpu total  ${fmt(d[4])} ms`,
      ].join('\n');
    },

    set_gaussian_multiplier: (value: number) => {
      render_settings[0] = value;
      update_render_settings();
    },
    set_frustum_culling: (enabled: boolean) => {
      render_settings[2] = enabled ? 1.0 : 0.0;
      update_render_settings();
    },
    set_tight_bounds: (enabled: boolean) => {
      render_settings[4] = enabled ? 1.0 : 0.0;
      update_render_settings();
    },
    set_workgroup_size: (size: number) => {
      workgroup_size = size;
      rebuild_preprocess_pipelines();
    },
    set_precision: (precision: PreprocessPrecision) => {
      if (precision === 'f16' && !device.features.has('shader-f16')) return false;
      preprocess_module = create_preprocess_module(precision);
      rebuild_preprocess_pipelines();
      return true;
    },
    set_mode: (m: GaussianRenderMode) => {
      mode = m;
    },
  };
}
