import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
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
  const deg = pc.sh_deg;
  const numPointsSize = pc.num_points * 24;
  const renderSettings = createBuffer(device,'renderSettings', 8, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([1.0, deg]));
  const splatBuffer = createBuffer( device, 'splatBuffer', numPointsSize, GPUBufferUsage.STORAGE);

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



  const pre_gaussian_bind = device.createBindGroup({
    label: 'pre_gaussian_bind',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } }, { binding: 1, resource: { buffer: splatBuffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

    const pre_camera_bind = device.createBindGroup({
    label: 'pre_camera_bind',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: renderSettings } },
    ],
  });

  const culling_bind = device.createBindGroup({
    label: 'culling_bind',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource:
        { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource:
        { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource:
        { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource:
        { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const indirect_draw_buff = createBuffer(device, 'indirect_draw_buffer',16, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array([6, pc.num_points, 0, 0]));

  const shader = device.createShaderModule({ code: renderWGSL });
  //create the render pipeline
  const pipeline_render = device.createRenderPipeline({
    fragment: {
      module: shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          alpha: {operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha'},
          color: {operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha'},
        },
      }],
    },
    label: 'pipeline_render',
    layout: 'auto',
    primitive: { topology: 'triangle-list',},
    vertex: { module: shader, entryPoint: 'vs_main',},
  });

  const splatBind = device.createBindGroup({
    label: 'splatBind',
    layout: pipeline_render.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({ label: 'preprocess' });
    //run the preprocess pipeline, bind the preprocess camera and gaussain bind groups and dispatch
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, pre_camera_bind);
    pass.setBindGroup(1, pre_gaussian_bind);
    pass.setBindGroup(2, culling_bind);
    //calculate the size of each work group
    const workgroupSize = Math.ceil(pc.num_points / C.histogram_wg_size);

    //dispatch
    pass.dispatchWorkgroups(workgroupSize);
    pass.end();
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buff, 4, 4);
  };


  const render = (encoder: GPUCommandEncoder,
    texture_view: GPUTextureView) => {
      //setup the render pass
    const pass = encoder.beginRenderPass({ label: 'pass',
      colorAttachments: [ {
          storeOp: 'store',
          loadOp: 'clear',
          view: texture_view,
        }
      ],
    });

    //render pass should use the render pipeline, camera and guassian binds and the indirect draw buffer to draw
    pass.setPipeline(pipeline_render);
    pass.setBindGroup(0, splatBind);
    pass.drawIndirect(indirect_draw_buff, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
        set_gaussian_multiplier: (value: number) => {
      device.queue.writeBuffer(renderSettings, 0, new Float32Array([value]));
    },
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
      preprocess(encoder);
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    camera_buffer,

  };
}
