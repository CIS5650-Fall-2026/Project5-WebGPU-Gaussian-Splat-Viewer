import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianScaling: (v: number) => void;
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

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const splat_buffer = createBuffer(device, 'splats', pc.num_points * 24, GPUBufferUsage.STORAGE);
const indirect_buffer = createBuffer(device, 'indirect draw', 16,
  GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, new Uint32Array([6, 0, 0, 0]));
const settings_buffer = createBuffer(device, 'render settings', 8,
  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([1.0, pc.sh_deg]));

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
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
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

  const camera_bind_group = device.createBindGroup({
  label: 'preprocess camera',
  layout: preprocess_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: camera_buffer } },
    { binding: 1, resource: { buffer: settings_buffer } },
  ],
});
const gaussian_bind_group = device.createBindGroup({
  label: 'preprocess gaussians',
  layout: preprocess_pipeline.getBindGroupLayout(1),
  entries: [
    { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
    { binding: 1, resource: { buffer: splat_buffer } },
    { binding: 2, resource: { buffer: pc.sh_buffer } },
  ],
});


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_module = device.createShaderModule({ code: renderWGSL });
const render_pipeline = device.createRenderPipeline({
  label: 'gaussian render',
  layout: 'auto',
  vertex: { module: render_module, entryPoint: 'vs_main' },
  fragment: { module: render_module, entryPoint: 'fs_main',
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
}] },
  primitive: { topology: 'triangle-list' },
});
const render_bind_group = device.createBindGroup({
  label: 'gaussian render',
  layout: render_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: splat_buffer } },
    { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
  ],
});

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
  device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
  device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

  const cpass = encoder.beginComputePass({ label: 'preprocess' });
  cpass.setPipeline(preprocess_pipeline);
  cpass.setBindGroup(0, camera_bind_group);
  cpass.setBindGroup(1, gaussian_bind_group);
  cpass.setBindGroup(2, sort_bind_group);
  cpass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
  cpass.end();

  sorter.sort(encoder);
  encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);

  const rpass = encoder.beginRenderPass({
    label: 'gaussian render',
    colorAttachments: [{ view: texture_view, loadOp: 'clear', storeOp: 'store',
                         clearValue: [0, 0, 0, 1] }],
  });
  rpass.setPipeline(render_pipeline);
  rpass.setBindGroup(0, render_bind_group);
  rpass.drawIndirect(indirect_buffer, 0);
  rpass.end();
},
setGaussianScaling: (v: number) => {
  device.queue.writeBuffer(settings_buffer, 0, new Float32Array([v]));
},
camera_buffer,
  };
}
