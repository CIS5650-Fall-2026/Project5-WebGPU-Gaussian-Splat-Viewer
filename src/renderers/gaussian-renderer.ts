import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import preprocessF16WGSL from '../shaders/preprocess_f16.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import tileCompositeWGSL from '../shaders/tile_composite.wgsl';
import tileBlitWGSL from '../shaders/tile_blit.wgsl';
import { get_sorter,C } from '../sort/sort';
import { Renderer } from './renderer';

// Must match the storage layout of Splat in the Gaussian shaders.
const c_size_splat = 64;
const c_tile_size = 16;
const c_max_tile_count = 1 << 15;
const c_tile_pairs_per_point = 16;
const c_max_tile_pairs = 24_000_000;

export interface GaussianRenderer extends Renderer {
  set_gaussian_scaling: (value: number) => void;
  set_culling: (enabled: boolean) => void;
  set_point_fraction: (fraction: number) => void;
}

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

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  canvas: HTMLCanvasElement,
  camera_buffer: GPUBuffer,
  initial_gaussian_scaling = 1,
  initial_culling = true,
  initial_point_fraction = 1,
): GaussianRenderer {
  const query_params = new URLSearchParams(window.location.search);
  const tile_sort_requested = query_params.get('sort') === 'tile';
  const workgroup_parameter = query_params.get('wg');
  const requested_workgroup_size = workgroup_parameter === null ? Number.NaN : Number(workgroup_parameter);
  const preprocess_workgroup_size = Number.isFinite(requested_workgroup_size)
    ? Math.min(
        Math.max(Math.round(requested_workgroup_size), 32),
        Math.min(512, device.limits.maxComputeInvocationsPerWorkgroup),
      )
    : C.histogram_wg_size;
  const requested_precision = query_params.get('precision');
  const use_f16_preprocess = requested_precision === 'f16' && device.features.has('shader-f16');
  if (requested_precision === 'f16' && !use_f16_preprocess) {
    console.warn('shader-f16 is unavailable; falling back to f32 preprocessing');
  }

  const tile_pair_capacity = tile_sort_requested
    ? Math.max(1, Math.min(pc.num_points * c_tile_pairs_per_point, c_max_tile_pairs))
    : 1;
  const sorter = get_sorter(
    tile_sort_requested ? tile_pair_capacity : pc.num_points,
    device,
  );

  const nulling_data = new Uint32Array([0]);
  const reset_draw_indirect_data = new Uint32Array([4, 0, 0, 0]);
  const splat_buffer = createBuffer(
    device,
    'splat buffer',
    pc.num_points * c_size_splat,
    GPUBufferUsage.STORAGE,
  );
  const render_settings_buffer = device.createBuffer({
    label: 'render settings',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const render_settings_data = new ArrayBuffer(32);
  const render_settings_float = new Float32Array(render_settings_data);
  const render_settings_uint = new Uint32Array(render_settings_data);
  render_settings_float[0] = Math.max(initial_gaussian_scaling, 0);
  render_settings_float[1] = pc.sh_deg;
  render_settings_float[2] = initial_culling ? 1 : 0;
  render_settings_float[3] = Math.min(Math.max(initial_point_fraction, 0), 1);
  render_settings_uint[7] = tile_pair_capacity;
  const write_render_settings = () => {
    device.queue.writeBuffer(render_settings_buffer, 0, render_settings_data);
  };
  write_render_settings();

  const draw_indirect_buffer = device.createBuffer({
    label: 'gaussian draw indirect',
    size: reset_draw_indirect_data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(draw_indirect_buffer, 0, reset_draw_indirect_data);

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: use_f16_preprocess ? preprocessF16WGSL : preprocessWGSL,
      }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: preprocess_workgroup_size,
        sortKeysPerDispatch: C.histogram_wg_size * C.rs_histogram_block_rows,
      },
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'gaussian preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussian preprocess data',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: render_settings_buffer } },
      { binding: 3, resource: { buffer: draw_indirect_buffer } },
      { binding: 4, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const render_shader = device.createShaderModule({ code: renderWGSL });
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
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  const splat_bind_group = device.createBindGroup({
    label: 'gaussian splats',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  let tile_compute_pipeline: GPUComputePipeline | undefined;
  let tile_blit_pipeline: GPURenderPipeline | undefined;
  let tile_target:
    | {
        width: number;
        height: number;
        texture: GPUTexture;
        compute_bind_group: GPUBindGroup;
        blit_bind_group: GPUBindGroup;
      }
    | undefined;

  if (tile_sort_requested) {
    const tile_module = device.createShaderModule({ code: tileCompositeWGSL });
    tile_compute_pipeline = device.createComputePipeline({
      label: 'tile composite',
      layout: 'auto',
      compute: {
        module: tile_module,
        entryPoint: 'composite',
      },
    });

    const blit_module = device.createShaderModule({ code: tileBlitWGSL });
    tile_blit_pipeline = device.createRenderPipeline({
      label: 'tile blit',
      layout: 'auto',
      vertex: {
        module: blit_module,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: blit_module,
        entryPoint: 'fs_main',
        targets: [{ format: presentation_format }],
      },
      primitive: {
        topology: 'triangle-strip',
      },
    });

  }

  const ensure_tile_target = (width: number, height: number) => {
    if (!tile_compute_pipeline || !tile_blit_pipeline) {
      throw new Error('Tile pipelines are not initialized');
    }
    if (tile_target?.width === width && tile_target.height === height) {
      return tile_target;
    }

    tile_target?.texture.destroy();
    const texture = device.createTexture({
      label: 'tile composite output',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const texture_view = texture.createView();
    tile_target = {
      width,
      height,
      texture,
      compute_bind_group: device.createBindGroup({
        label: 'tile composite bind group',
        layout: tile_compute_pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
          { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
          { binding: 2, resource: { buffer: splat_buffer } },
          { binding: 3, resource: { buffer: camera_buffer } },
          { binding: 4, resource: { buffer: sorter.sort_info_buffer } },
          { binding: 5, resource: texture_view },
        ],
      }),
      blit_bind_group: device.createBindGroup({
        label: 'tile blit bind group',
        layout: tile_blit_pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture_view },
        ],
      }),
    };
    return tile_target;
  };

  let tile_disabled_logged = false;
  const update_tile_settings = () => {
    if (!tile_sort_requested) {
      return false;
    }
    const cols = Math.ceil(canvas.width / c_tile_size);
    const rows = Math.ceil(canvas.height / c_tile_size);
    const enabled = cols * rows <= c_max_tile_count;
    render_settings_uint[4] = enabled ? cols : 0;
    render_settings_uint[5] = enabled ? rows : 0;
    render_settings_uint[6] = c_tile_size;
    write_render_settings();

    if (!enabled && !tile_disabled_logged) {
      console.warn(`Tile sort requires at most ${c_max_tile_count} tiles; using global sorting`);
      tile_disabled_logged = true;
    }
    return enabled;
  };

  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({ label: 'gaussian preprocess' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / preprocess_workgroup_size));
    pass.end();
  };

  const render_global = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, splat_bind_group);
    pass.drawIndirect(draw_indirect_buffer, 0);
    pass.end();
  };

  const render_tile = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    if (!tile_compute_pipeline || !tile_blit_pipeline) {
      throw new Error('Tile pipelines are not initialized');
    }
    const width = Math.max(canvas.width, 1);
    const height = Math.max(canvas.height, 1);
    const target = ensure_tile_target(width, height);
    const tile_cols = Math.ceil(width / c_tile_size);
    const tile_rows = Math.ceil(height / c_tile_size);

    const compute_pass = encoder.beginComputePass({ label: 'tile composite' });
    compute_pass.setPipeline(tile_compute_pipeline);
    compute_pass.setBindGroup(0, target.compute_bind_group);
    compute_pass.dispatchWorkgroups(tile_cols, tile_rows);
    compute_pass.end();

    const blit_pass = encoder.beginRenderPass({
      label: 'tile blit',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    blit_pass.setPipeline(tile_blit_pipeline);
    blit_pass.setBindGroup(0, target.blit_bind_group);
    blit_pass.draw(4);
    blit_pass.end();
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      const tile_enabled = update_tile_settings();
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
      device.queue.writeBuffer(draw_indirect_buffer, 0, reset_draw_indirect_data);
      preprocess(encoder);
      sorter.sort(encoder);
      if (tile_enabled) {
        render_tile(encoder, texture_view);
      } else {
        render_global(encoder, texture_view);
      }
    },
    camera_buffer,
    set_gaussian_scaling: (value: number) => {
      render_settings_float[0] = Math.max(value, 0);
      write_render_settings();
    },
    set_culling: (enabled: boolean) => {
      render_settings_float[2] = enabled ? 1 : 0;
      write_render_settings();
    },
    set_point_fraction: (fraction: number) => {
      render_settings_float[3] = Math.min(Math.max(fraction, 0), 1);
      write_render_settings();
    },
  };
}
