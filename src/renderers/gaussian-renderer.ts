import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_scale: (scale: number) => void,
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
  const preprocess_size = C.histogram_wg_size;
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  
  const splat_buffer = createBuffer(
    device, 'Projected 2D Gaussian splats (data used by the render shader)', pc.num_points * 48,
    GPUBufferUsage.STORAGE
  );
  
  const settings_buffer = createBuffer(
    device, 'Gaussian renderer settings (scale multiplier and SH degree)', 16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1, pc.sh_deg, 0, 0])
  );
  
  const draw_buffer = createBuffer(
    device, 'Indirect draw args (quad vertices and visible splat count)', 16,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([6, 0, 0, 0])
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'Gaussian preprocessing pipeline',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: preprocess_size,
        sortKeyPerThread: c_histogram_block_rows,
        sortWorkgroupSize: C.histogram_wg_size,
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


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const camera_bind_group = device.createBindGroup({
    label: 'Preprocess camera and settings',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: settings_buffer } },
    ],
  });

  const data_bind_group = device.createBindGroup({
    label: 'Preprocess Gaussian data (3D values, SH coefficients, and 2D output)',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
    ],
  });

  const render_shader = device.createShaderModule({ code: renderWGSL });

  const render_pipeline = device.createRenderPipeline({
    label: 'Gaussian rendering pipeline (sorted quads with alpha blending)',
    layout: 'auto',
    vertex: { module: render_shader, entryPoint: 'vs_main' },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const render_camera_group = device.createBindGroup({
    label: 'Gaussian render camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const render_data_group = device.createBindGroup({
    label: 'Gaussian render data',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });


  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({ label: 'Gaussian preprocess pass' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, data_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / preprocess_size));
    pass.end();

    // First u32 in sorter.sort_info_buffer is the number of visible splats.
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, draw_buffer, 4, 4);
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'Gaussian render pass',
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_camera_group);
    pass.setBindGroup(1, render_data_group);
    pass.drawIndirect(draw_buffer, 0);
    pass.end();
  };


  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // Each frame starts with zero visible splats and zero sort workgroups.
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

      preprocess(encoder);
      sorter.sort(encoder); // Provided radix sort; sorted indices end in ping_pong[0].
      render(encoder, texture_view);
    },
    set_scale: (scale: number) => {
      device.queue.writeBuffer(settings_buffer, 0, new Float32Array([scale]));
    },
    camera_buffer,
  };
}
