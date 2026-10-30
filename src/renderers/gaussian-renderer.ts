import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setScale(value: number) : void;
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
  // ===========================
  // ====================

  const nulling_data = new Uint32Array([0]);

  const drawArgs = new Uint32Array(4);
  drawArgs[0] = 4;
  drawArgs[1] = pc.num_points;
  drawArgs[2] = 0; // First Vertex
  drawArgs[3] = 0; // First Instance

  const indirect_data = createBuffer(device, "indirect", drawArgs.byteLength, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, drawArgs);

  const splat_buffer = device.createBuffer({
      label: "splats", 
      size: pc.num_points * 48,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }); // read write in preprocess shader, read in gaussian shader


  const renderSettings = new Float32Array(2);
  renderSettings[0] = 1.0;
  renderSettings[1] = pc.sh_deg;

  const render_settings =  createBuffer(device, "render settings", renderSettings.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, renderSettings);

  const setScale = (scale: number) =>
  {
    renderSettings[0] = scale;
    device.queue.writeBuffer(render_settings, 0, renderSettings);
  }

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
        sortKeyPerThread: c_histogram_block_rows, // for each thread in sorting workgroup, how many keys it is repsonible for sorting
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

     const camera_bind_group_compute = device.createBindGroup({
      label: 'gaussian camera',
      layout: preprocess_pipeline.getBindGroupLayout(0),
      entries: [{binding: 0, resource: { buffer: camera_buffer }}],
    });

    const gaussian_bind_group = device.createBindGroup({
      label: 'gaussians',
      layout: preprocess_pipeline.getBindGroupLayout(1),
      entries: [
        {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
        {binding: 1, resource: {buffer: render_settings}},
        {binding: 2, resource: {buffer: pc.sh_buffer}}
      ],
    });

    const splat_bind_group_compute = device.createBindGroup({
    label: 'use in gaussian wgsl',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      {binding: 0, resource: { buffer: splat_buffer }},
    ],
  })

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'quad render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main'
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format,
        blend: { // to see opacity values]
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
       }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

    const camera_bind_group = device.createBindGroup({
      label: 'gaussian camera',
      layout: render_pipeline.getBindGroupLayout(0),
      entries: [{binding: 0, resource: { buffer: camera_buffer }}],
    });


    const splat_bind_group = device.createBindGroup({
    label: 'splats',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: splat_buffer }},
    ],
  })

   const sort_indices_bind_group = device.createBindGroup({
    label: 'sort indices',
    layout: render_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } }
    ],
  });
  
  // ===============================================
  //    Command Encoder Functions
  // ===============================================

    const preprocess = (encoder: GPUCommandEncoder) => {
      const computePass = encoder.beginComputePass();

      computePass.setPipeline(preprocess_pipeline);
      computePass.setBindGroup(0, camera_bind_group_compute);
      computePass.setBindGroup(1, gaussian_bind_group);
      computePass.setBindGroup(2, sort_bind_group);
      computePass.setBindGroup(3, splat_bind_group_compute); // write to this empty buffer

      computePass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
        
      computePass.end();
  }
  
  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    // somehow use 
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, splat_bind_group); // NDC position filled buffer to be rendered
    pass.setBindGroup(2, sort_indices_bind_group) // sorted indices
    pass.drawIndirect(indirect_data, 0);

    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // reset counter 
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

      preprocess(encoder);

      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_data, 4, 4); // copy over indirect to sort info for render
      sorter.sort(encoder); // sort based upon our depth

      render(encoder, texture_view);
    },
    setScale,
    camera_buffer,
  };
}
