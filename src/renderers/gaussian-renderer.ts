import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void
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

  const preprocess_workgroup_size = 128;

  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  // vertices per quad, quad count, first vertex, first instance
  const indirect_draw_buffer = createBuffer(
    device,
    'indirect draw',
    16,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,

    // six vertices make two triangles
    new Uint32Array([6, pc.num_points, 0, 0]),
  );

  // 48 bytes per splat for each gaussain
  const splat_buffer = createBuffer(
    device,
    'screen splats',
    pc.num_points * 48,
    GPUBufferUsage.STORAGE,
  );

  // default value at first, then sh degree
  const render_settings_data = new Float32Array([
    1.0,
    pc.sh_deg
  ]);


  const render_settings_buffer = createBuffer(
    device,
    'render settings',
    render_settings_data.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings_data
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
        workgroupSize: preprocess_workgroup_size,
        sortWorkgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows
      },
    },
  });


  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } }
    ],
  });


  const preprocess_data_bind_group = device.createBindGroup({
    label: 'preprocess data',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } }
    ],
  });


  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } }, // visiable keys
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } }, // depth keys
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } }
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const gaussian_shader = device.createShaderModule({ code: renderWGSL });
  const gaussian_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: gaussian_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: gaussian_shader,
      entryPoint: 'fs_main',
      targets: [
        {
          format: presentation_format,
          blend: {
            // mix this splat color with all the stuff behind it
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            // combine opacity of the splat with all the opacity behind it
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const render_camera_bind_group = device.createBindGroup({
    label: 'gaussian camera',
    layout: gaussian_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussian points',
    layout: gaussian_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } } // those sorted indices
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const preprocess = (encoder: GPUCommandEncoder) => {
    const compute_pass = encoder.beginComputePass({
      label: 'gaussian preprocess',
    });
  
    compute_pass.setPipeline(preprocess_pipeline);
    compute_pass.setBindGroup(0, preprocess_camera_bind_group);
    compute_pass.setBindGroup(1, preprocess_data_bind_group);
    compute_pass.setBindGroup(2, sort_bind_group);
  
    const workgroup_count = Math.ceil(pc.num_points / preprocess_workgroup_size);

    compute_pass.dispatchWorkgroups(workgroup_count);
    compute_pass.end();
  };
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      // we make visible splat count starts from zero every frame
      device.queue.writeBuffer(
        sorter.sort_info_buffer,
        0,
        nulling_data
      );

      // we reset sorting dispatch too so we do not allocate mroe workgroups
      device.queue.writeBuffer(
        sorter.sort_dispatch_indirect_buffer,
        0,
        nulling_data
      );

      preprocess(encoder);

      // copy how many we kept into the draw call
      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer,
        0,
        indirect_draw_buffer,
        4,
        4
      );

      sorter.sort(encoder);

      const pass = encoder.beginRenderPass({
        label: 'gaussian render',
        colorAttachments: [
          {
            view: texture_view,
            clearValue: [0, 0, 0, 1],
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      pass.setPipeline(gaussian_pipeline);
      pass.setBindGroup(0, render_camera_bind_group);
      pass.setBindGroup(1, gaussian_bind_group);
      pass.drawIndirect(indirect_draw_buffer, 0);
      pass.end();
    },

    set_gaussian_multiplier: (value: number) => {
      device.queue.writeBuffer(
        render_settings_buffer,
        0,
        new Float32Array([value])
      );
    },
    
    camera_buffer,
  };
}
