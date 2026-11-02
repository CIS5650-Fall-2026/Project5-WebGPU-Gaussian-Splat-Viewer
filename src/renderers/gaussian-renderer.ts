import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import preprocessF16WGSL from '../shaders/preprocess_f16.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import compositeWGSL from '../shaders/composite.wgsl';
import { get_sorter,c_histogram_block_rows,C,SortStuff } from '../sort/sort';
import { align } from '../utils/util';
import { Renderer } from './renderer';

export interface GaussianParams {
  gaussian_multiplier: number,
  f16_preprocess: boolean,
  tile_based: boolean,
}

export interface GaussianRenderer extends Renderer {
  set_params: (params: GaussianParams) => void,
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

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const keys_per_dispatch = C.histogram_wg_size * c_histogram_block_rows;
  const sorter = get_sorter(pc.num_points, device);
  const tile_sorter = get_sorter(align(pc.num_points * 4, keys_per_dispatch), device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  const splat_buffer = createBuffer(device, 'splats', pc.num_points * 24, GPUBufferUsage.STORAGE);
  const settings_buffer = createBuffer(device, 'render settings', 8, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const indirect_buffer = createBuffer(device, 'draw indirect', 16, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, new Uint32Array([4, 0, 0, 0]));

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const buffer_layout = (types: GPUBufferBindingType[]) => device.createBindGroupLayout({
    entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
  });
  const camera_layout = buffer_layout(['uniform', 'uniform']);
  const gaussian_layout = buffer_layout(['read-only-storage', 'read-only-storage', 'storage']);
  const sort_layout = buffer_layout(['storage', 'storage', 'storage', 'storage']);
  const preprocess_layout = device.createPipelineLayout({ bindGroupLayouts: [camera_layout, gaussian_layout, sort_layout] });

  const preprocess_pipelines = (code: string) => {
    const module = device.createShaderModule({ code });
    const pipeline = (entryPoint: string) => device.createComputePipeline({
      label: entryPoint,
      layout: preprocess_layout,
      compute: {
        module,
        entryPoint,
        constants: {
          workgroupSize: C.histogram_wg_size,
          sortKeyPerThread: c_histogram_block_rows,
        },
      },
    });
    return { quad: pipeline('preprocess'), tile: pipeline('preprocess_tile') };
  };
  const f32_pipelines = preprocess_pipelines(preprocessWGSL);
  const f16_pipelines = device.features.has('shader-f16') ? preprocess_pipelines(preprocessF16WGSL) : f32_pipelines;

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: camera_layout,
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: settings_buffer } },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: gaussian_layout,
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
    ],
  });

  const sort_bind_group = (s: SortStuff) => device.createBindGroup({
    label: 'sort',
    layout: sort_layout,
    entries: [
      { binding: 0, resource: { buffer: s.sort_info_buffer } },
      { binding: 1, resource: { buffer: s.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: s.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: s.sort_dispatch_indirect_buffer } },
    ],
  });
  const sort_bind_groups = { quad: sort_bind_group(sorter), tile: sort_bind_group(tile_sorter) };

  const composite_pipeline = device.createComputePipeline({
    label: 'composite',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: compositeWGSL }),
      entryPoint: 'composite',
    },
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
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
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  const render_camera_bind_group = device.createBindGroup({
    label: 'gaussian camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const splat_bind_group = device.createBindGroup({
    label: 'gaussian splats',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, s: SortStuff, sort_group: GPUBindGroup) => {
    device.queue.writeBuffer(s.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(s.sort_dispatch_indirect_buffer, 0, nulling_data);
    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture: GPUTexture) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_camera_bind_group);
    pass.setBindGroup(1, splat_bind_group);
    pass.drawIndirect(indirect_buffer, 0);
    pass.end();
  };

  const composite = (encoder: GPUCommandEncoder, texture: GPUTexture) => {
    const bind_group = device.createBindGroup({
      label: 'composite',
      layout: composite_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: splat_buffer } },
        { binding: 2, resource: { buffer: tile_sorter.sort_info_buffer } },
        { binding: 3, resource: { buffer: tile_sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 4, resource: { buffer: tile_sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 5, resource: texture.createView() },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'composite' });
    pass.setPipeline(composite_pipeline);
    pass.setBindGroup(0, bind_group);
    pass.dispatchWorkgroups(Math.ceil(texture.width / 16), Math.ceil(texture.height / 16));
    pass.end();
  };

  let f16_preprocess = false;
  let tile_based = false;

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture: GPUTexture) => {
      const pipelines = f16_preprocess ? f16_pipelines : f32_pipelines;
      if (tile_based) {
        preprocess(encoder, pipelines.tile, tile_sorter, sort_bind_groups.tile);
        tile_sorter.sort(encoder);
        composite(encoder, texture);
      } else {
        preprocess(encoder, pipelines.quad, sorter, sort_bind_groups.quad);
        encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);
        sorter.sort(encoder);
        render(encoder, texture);
      }
    },
    camera_buffer,
    set_params: (params: GaussianParams) => {
      f16_preprocess = params.f16_preprocess;
      tile_based = params.tile_based;
      device.queue.writeBuffer(settings_buffer, 0, new Float32Array([params.gaussian_multiplier, pc.sh_deg]));
    },
  };
}
