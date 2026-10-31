import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_scaling: (scaling: number) => void,
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

  const splat_buffer = createBuffer(
    device, 
    "splats", 
    pc.num_points * 12 * 4, // N * size(vec2<f32>)
    GPUBufferUsage.STORAGE,
  );

  const draw_buffer = createBuffer(
    device, 
    "indirect draw", 
    4 * 4, 
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  );
  device.queue.writeBuffer(draw_buffer, 0, 
    new Uint32Array([6, pc.num_points, 0, 0]));

  const render_settings_buffer = createBuffer(
    device,
    'render settings',
    2 * 4, // 2 floats
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg]),
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
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: { buffer: camera_buffer }},
      {binding: 1, resource: { buffer: render_settings_buffer }}
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussian',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
      {binding: 1, resource: { buffer: splat_buffer}},
      {binding: 2, resource: { buffer: pc.sh_buffer}},
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'preprocess sort',
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
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'render',
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
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const splat_bind_group = device.createBindGroup({
    label: 'render splats',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: { buffer: splat_buffer}},
      {binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer}},
      {binding: 2, resource: { buffer: camera_buffer}},
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  function record_preprocess(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass({
      label: 'preprocess pass',
    });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    pass.end();
  }

  const record_render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'render pass',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, splat_bind_group);
    pass.drawIndirect(draw_buffer, 0);
    pass.end();
  };
  
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // Reset sorting buffers!
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

      // Preprocess pass
      record_preprocess(encoder);
      
      // Sort splat by depth
      sorter.sort(encoder);
      
      // Save sorted info
      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 0,
        draw_buffer, 4, 4
      );
      
      // Render splats
      record_render(encoder, texture_view);
    },
    camera_buffer,
    set_gaussian_scaling: (scaling: number) => {
      device.queue.writeBuffer(render_settings_buffer, 0, new Float32Array([scaling]));
    }
  };
}
