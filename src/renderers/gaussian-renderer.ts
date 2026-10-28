import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
}

export interface GaussianRendererOptions {
  gaussian_multiplier?: number,
  preprocess_workgroup_size?: number,
  frustum_culling?: boolean,
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

const c_size_splat = 32;  // see Splat in preprocess.wgsl

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  options: GaussianRendererOptions = {},
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  const preprocess_wg_size = options.preprocess_workgroup_size ?? C.histogram_wg_size;
  // 2D dispatch so large scenes / small workgroups stay under maxComputeWorkgroupsPerDimension
  const num_wg = Math.ceil(pc.num_points / preprocess_wg_size);
  const num_wg_x = Math.min(num_wg, device.limits.maxComputeWorkgroupsPerDimension);
  const num_wg_y = Math.ceil(num_wg / num_wg_x);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const splat_buffer = createBuffer(
    device, 'splats', pc.num_points * c_size_splat,
    GPUBufferUsage.STORAGE,
  );

  // gaussian_scaling, sh_deg
  const render_settings = new Float32Array([options.gaussian_multiplier ?? 1.0, pc.sh_deg]);
  const render_settings_buffer = createBuffer(
    device, 'render settings', render_settings.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, render_settings,
  );

  // vertex_count, instance_count, first_vertex, first_instance
  // instance_count is overwritten every frame with the number of visible splats
  const indirect_draw_buffer = createBuffer(
    device, 'gaussian indirect draw', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([4, 0, 0, 0]),
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: preprocess_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
        sortWorkgroupSize: C.histogram_wg_size,
        frustumCulling: (options.frustum_culling ?? true) ? 1 : 0,
      },
    },
  });

  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const preprocess_gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
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
        // premultiplied "over", splats are drawn back to front
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  // after 4 radix passes the sorted indices end up back in ping_pong[0]
  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    // reset visible splat count and sort dispatch size
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(num_wg_x, num_wg_y);
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    // instance_count = number of visible splats
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [{
        view: texture_view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    set_gaussian_multiplier: (value: number) => {
      render_settings[0] = value;
      device.queue.writeBuffer(render_settings_buffer, 0, render_settings);
    },
    camera_buffer,
  };
}
