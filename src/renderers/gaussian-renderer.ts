import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianScaling: (scale: number) => void;
}

// perfomance analysis
const WORKGROUP_SIZE = 256;
const ENABLE_FRUSTUM_CULLING = 1;

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

  const splat_buffer = createBuffer(
    device,
    'splats',
    pc.num_points * 48,
    GPUBufferUsage.STORAGE
  );

  const settings_buffer = createBuffer(
    device,
    'render settings',
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1, pc.sh_deg, 0, 0])
  );

  const draw_indirect_buffer = createBuffer(
    device,
    'indirect draw',
    16,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([6, 0, 0, 0])
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
        workgroupSize: WORKGROUP_SIZE,
        sortKeyPerThread: c_histogram_block_rows,
        enableFrustumCulling: ENABLE_FRUSTUM_CULLING,
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

  const preprocess_camera_bind_group = device.createBindGroup({
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
    ],
  });

  const preprocess_data_bind_group = device.createBindGroup({
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: settings_buffer } },
      { binding: 3, resource: { buffer: pc.sh_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const render_module = device.createShaderModule({
    code: renderWGSL
  });

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',

    vertex: {
      module: render_module,
      entryPoint: 'vs_main',
    },

    fragment: {
      module: render_module,
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
      topology: 'triangle-list',
    },
  });

  const render_camera_bind_group = device.createBindGroup({
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
    ],
  });

  const render_data_bind_group = device.createBindGroup({
    layout: render_pipeline.getBindGroupLayout(1),
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
      // reset counters
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

      // preprocess gaussians
      const compute_pass = encoder.beginComputePass();

      compute_pass.setPipeline(preprocess_pipeline);
      compute_pass.setBindGroup(0, preprocess_camera_bind_group);
      compute_pass.setBindGroup(1, preprocess_data_bind_group);
      compute_pass.setBindGroup(2, sort_bind_group);

      compute_pass.dispatchWorkgroups(
        Math.ceil(pc.num_points / WORKGROUP_SIZE)
      );

      compute_pass.end();

      // copy visible count to indirect draw instance count
      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 0,
        draw_indirect_buffer, 4,
        4
      );

      // sort visible gaussians
      sorter.sort(encoder);

      // render
      const render_pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture_view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });

      render_pass.setPipeline(render_pipeline);
      render_pass.setBindGroup(0, render_camera_bind_group);
      render_pass.setBindGroup(1, render_data_bind_group);
      render_pass.drawIndirect(draw_indirect_buffer, 0);

      render_pass.end();
    },
    camera_buffer,
    setGaussianScaling: (scale: number) => {
      device.queue.writeBuffer(settings_buffer, 0, new Float32Array([scale]));
    },
  };
}
