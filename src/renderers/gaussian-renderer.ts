import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianMultiplier: (value: number) => void,
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

  const render_settings_buffer = createBuffer(
    device,
    'render settings',
    4 * Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([
      1.0,        // gaussian_scaling
      pc.sh_deg,  // sh_deg
      0.0,
      0.0,
    ])
  );

  const splat_buffer = createBuffer(
    device,
    '2D splat buffer',
    pc.num_points * 16 * Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );

  const draw_indirect_buffer = createBuffer(
    device,
    'gaussian draw indirect',
    4 * Uint32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([
      6,              // vertexCount: 6 vertices per quad
      pc.num_points,  // instanceCount: one quad per Gaussian
      0,              // firstVertex
      0,              // firstInstance
    ])
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

  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const preprocess_data_bind_group = device.createBindGroup({
    label: 'preprocess gaussian data',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
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
  
  const render_shader = device.createShaderModule({
    label: 'gaussian render shader',
    code: renderWGSL,
  });

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render pipeline',
    layout: 'auto',

    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },

    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [
        {
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
        }
      ],
    },

    primitive: {
      topology: 'triangle-list',
    },
  });

  const render_camera_bind_group = device.createBindGroup({
    label: 'gaussian render camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: camera_buffer },
      }
    ],
  });

  const render_gaussian_bind_group = device.createBindGroup({
    label: 'gaussian render input gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: 1,
        resource: { buffer: splat_buffer },
      },
      {
        binding: 2,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
      }
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  
  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({
      label: 'gaussian preprocess',
    });

    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_data_bind_group);
    pass.setBindGroup(2, sort_bind_group);

    const workgroup_count = Math.ceil(pc.num_points / C.histogram_wg_size);
    pass.dispatchWorkgroups(workgroup_count);

    pass.end();
  };

  const render = (
    encoder: GPUCommandEncoder,
    texture_view: GPUTextureView,
  ) => {
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

  pass.setBindGroup(
    0,
    render_camera_bind_group
  );

  pass.setBindGroup(
    1,
    render_gaussian_bind_group
  );

    pass.drawIndirect(
      draw_indirect_buffer,
      0
    );

    pass.end();
  };

  const setGaussianMultiplier = (value: number) => {
    device.queue.writeBuffer(
      render_settings_buffer,
      0,
      new Float32Array([value])
    );
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (
      encoder: GPUCommandEncoder,
      texture_view: GPUTextureView
    ) => {
      device.queue.writeBuffer(
        sorter.sort_info_buffer,
        0,
        nulling_data
      );

      device.queue.writeBuffer(
        sorter.sort_dispatch_indirect_buffer,
        0,
        nulling_data
      );

      preprocess(encoder);

      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer,
        0,
        draw_indirect_buffer,
        Uint32Array.BYTES_PER_ELEMENT,
        Uint32Array.BYTES_PER_ELEMENT
      );

      sorter.sort(encoder);

      render(
        encoder,
        texture_view
      );
    },

    camera_buffer,
    setGaussianMultiplier,
  };
}