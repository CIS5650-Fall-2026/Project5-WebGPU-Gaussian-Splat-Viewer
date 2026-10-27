import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { create_tile_renderer } from './tile-renderer';

export interface GaussianRenderer extends Renderer {
  update_settings: (params: { gaussian_multiplier: number, culling: boolean, half_precision: boolean, tile_sort: boolean }) => void,
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

// byte size of Splat in the shaders
const c_size_splat = 8 * 4;

// the same preprocess shader with real = f16 and f16 values read with bitcast
const preprocessF16WGSL = 'enable f16;\n' + preprocessWGSL
  .replace('alias real = f32;', 'alias real = f16;')
  .replace('return unpack2x16float(w);', 'return bitcast<vec2<f16>>(w);');

// preprocess workgroup size, can be changed with ?workgroup=N in the url
const workgroup_size = Number(new URLSearchParams(location.search).get('workgroup') ?? C.histogram_wg_size);

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  // 6 vertices per quad, the instance count is copied from the sort infos every frame
  const indirect_buffer = createBuffer(device, 'gaussian draw indirect', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, new Uint32Array([6, 0, 0, 0]));
  const splat_buffer = createBuffer(device, 'gaussian splats', pc.num_points * c_size_splat, GPUBufferUsage.STORAGE);
  // gaussian_scaling, sh_deg, culling
  const settings_buffer = createBuffer(device, 'gaussian render settings', 3 * 4,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([1.0, pc.sh_deg, 1.0]));

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const create_preprocess = (code: string) => {
    const pipeline = device.createComputePipeline({
      label: 'preprocess',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code }),
        entryPoint: 'preprocess',
        constants: {
          workgroupSize: workgroup_size,
          sortKeyPerThread: c_histogram_block_rows,
          sortWorkgroupSize: C.histogram_wg_size,
        },
      },
    });

    const camera_bind_group = device.createBindGroup({
      label: 'preprocess camera',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: settings_buffer } },
      ],
    });

    const gaussian_bind_group = device.createBindGroup({
      label: 'preprocess gaussians',
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
        { binding: 1, resource: { buffer: splat_buffer } },
        { binding: 2, resource: { buffer: pc.sh_buffer } },
      ],
    });

    const sort_bind_group = device.createBindGroup({
      label: 'sort',
      layout: pipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      ],
    });
    return { pipeline, bind_groups: [camera_bind_group, gaussian_bind_group, sort_bind_group] };
  };

  const preprocess_f32 = create_preprocess(preprocessWGSL);
  const preprocess_f16 = device.features.has('shader-f16') ? create_preprocess(preprocessF16WGSL) : undefined;
  let half_precision = false;
  let tile_sort = false;
  let tiles: ReturnType<typeof create_tile_renderer> | undefined;


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
        // back to front "over" blending with premultiplied color
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    const active = (half_precision && preprocess_f16) || preprocess_f32;
    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(active.pipeline);
    active.bind_groups.forEach((group, i) => pass.setBindGroup(i, group));
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / workgroup_size));
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          clearValue: [0, 0, 0, 1],
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView, texture: GPUTexture) => {
      // reset the visible count and the sort dispatch size
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
      preprocess(encoder);
      if (tile_sort) {
        // made when it is first used, its sort buffers are big
        tiles ??= create_tile_renderer(device, pc.num_points, presentation_format, camera_buffer, splat_buffer, sorter);
        tiles.frame(encoder, texture, texture_view);
        return;
      }
      sorter.sort(encoder);
      // instance count = number of splats left after culling
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);
      render(encoder, texture_view);
    },
    camera_buffer,
    update_settings: (params) => {
      device.queue.writeBuffer(settings_buffer, 0, new Float32Array([params.gaussian_multiplier, pc.sh_deg, params.culling ? 1 : 0]));
      half_precision = params.half_precision;
      tile_sort = params.tile_sort;
    },
  };
}
