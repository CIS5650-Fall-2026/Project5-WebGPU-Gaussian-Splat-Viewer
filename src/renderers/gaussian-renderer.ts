import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_Gaussian_scaling: (scaling: number) => void,
}

const PREPROCESS_WG = 256;

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

  //analysis
  const readback_buffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  let reading = false;
  setInterval(async () => {
    if (reading) return;
    reading = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sorter.sort_info_buffer, 0, readback_buffer, 0, 4);
    device.queue.submit([enc.finish()]);
    await readback_buffer.mapAsync(GPUMapMode.READ);
    const visible = new Uint32Array(readback_buffer.getMappedRange())[0];
    readback_buffer.unmap();
    console.log(`visible: ${visible} / ${pc.num_points}`);
    reading = false;
  }, 2000);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const tempData = new Uint32Array([6, pc.num_points, 0, 0]);
  const indirect_buffer = createBuffer(
    device,
    "indirect drawing",
    4 * 4, // 4 uint32 values
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    tempData
  );

  const splat_buffer = createBuffer(
    device,
    "splat buffer",
    pc.num_points * 48, // 4 float32 values per point
    GPUBufferUsage.STORAGE
  );

  const scale_data = new Float32Array([1.0, pc.sh_deg]);

  const scale_buffer = createBuffer(
    device,
    "scale buffer",
    4 * 4, // 4 float32 values
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    scale_data
  );



  const nulling_data = new Uint32Array([0]);

  

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
        workgroupSize: PREPROCESS_WG,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } },
              { binding: 1, resource: { buffer: scale_buffer } }
    ],
  });

  const preprocess_gaussion_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      {binding: 2, resource: { buffer: pc.sh_buffer }}
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
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });
  
 
  const splat_bind_group = device.createBindGroup({
    label: 'gaussian gaussians',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: { buffer: splat_buffer }},
      {binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer }},
      {binding: 2, resource: { buffer: camera_buffer }},
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_gaussion_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / PREPROCESS_WG));
    pass.end();
  }


   const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'Gaussian render',
      colorAttachments: [
        {
          view:texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, splat_bind_group);

    pass.drawIndirect(indirect_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);
      render(encoder, texture_view);
    },
    camera_buffer,
    set_Gaussian_scaling: (scaling: number) => {
      device.queue.writeBuffer(scale_buffer, 0, new Float32Array([scaling, pc.sh_deg]));
    }
  };
}
