import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_scaling: (value: number) => void;
  set_sh_degree: (value: number) => void;
  set_frustum_culling: (enabled: boolean) => void;
  set_gaussian_stride: (value: number) => number;
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

//  const sorter = get_sorter(pc.num_points, device);
//  
//  // ===============================================
//  //            Initialize GPU Buffers
//  // ===============================================
//
//  const nulling_data = new Uint32Array([0]);
//
//  // ===============================================
//  //    Create Compute Pipeline and Bind Groups
//  // ===============================================
//  const preprocess_pipeline = device.createComputePipeline({
//    label: 'preprocess',
//    layout: 'auto',
//    compute: {
//      module: device.createShaderModule({ code: preprocessWGSL }),
//      entryPoint: 'preprocess',
//      constants: {
//        workgroupSize: C.histogram_wg_size,
//        sortKeyPerThread: c_histogram_block_rows,
//      },
//    },
//  });
//
//  const sort_bind_group = device.createBindGroup({
//    label: 'sort',
//    layout: preprocess_pipeline.getBindGroupLayout(2),
//    entries: [
//      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
//      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
//      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
//      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
//    ],
//  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const preprocess_workgroup_size = 256;

  const sort_keys_per_workgroup = C.histogram_wg_size * c_histogram_block_rows;

  const sorter = get_sorter(pc.num_points, device);

  let active_gaussian_count = pc.num_points;

  const render_settings_data = new Float32Array([
    1.0, // gaussian_scaling
    0.0, // sh_deg
    1.0, // frustum_culling
    1.0, // gaussian_stride
  ]);

  const max_sh_degree = Math.max(
    0,
    Math.min(3, Math.floor(pc.sh_deg)),
  );

  const render_settings_buffer = createBuffer(
    device,
    'gaussian render settings',
    render_settings_data.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings_data,
  );

  const set_gaussian_scaling = (value: number): void => {
    render_settings_data[0] = value;

    device.queue.writeBuffer(
      render_settings_buffer,
      0,
      render_settings_data,
    );
  };

  const set_sh_degree = (value: number): void => {
    render_settings_data[1] = Math.max(
      0,
      Math.min(max_sh_degree, Math.floor(value)),
    );

    device.queue.writeBuffer(
      render_settings_buffer,
      0,
      render_settings_data,
    );
  };

  const set_frustum_culling = (enabled: boolean): void => {
    render_settings_data[2] = enabled ? 1.0 : 0.0;

    device.queue.writeBuffer(
      render_settings_buffer,
      0,
      render_settings_data,
    );
  };

  const set_gaussian_stride = (value: number): number => {
    const stride = Math.max(1, Math.floor(value));

    active_gaussian_count = Math.ceil(pc.num_points / stride);
    render_settings_data[3] = stride;

    device.queue.writeBuffer(
      render_settings_buffer,
      0,
      render_settings_data,
    );

    const padded_count = Math.ceil(
      active_gaussian_count / sort_keys_per_workgroup,
    ) * sort_keys_per_workgroup;

    device.queue.writeBuffer(
      sorter.sort_info_buffer,
      4,
      new Uint32Array([padded_count]),
    );

    return active_gaussian_count;
  };

  const splat_buffer = createBuffer(
    device,
    'projected splats',
    pc.num_points * 64,
    GPUBufferUsage.STORAGE,
  );

  const sort_info_buffer = sorter.sort_info_buffer;

  const preprocess_shader = device.createShaderModule({
    label: 'project gaussian centers',
    code: preprocessWGSL,
  });

  const preprocess_pipeline = device.createComputePipeline({
    label: 'project gaussian centers',
    layout: 'auto',
    compute: {
      module: preprocess_shader,
      entryPoint: 'preprocess',
      constants: {
        preprocessWorkgroupSize: preprocess_workgroup_size,
        sortKeysPerWorkgroup: sort_keys_per_workgroup,
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
    label: 'preprocess gaussian and splat data',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const preprocess_count_bind_group = device.createBindGroup({
    label: 'preprocess sorting data',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sort_info_buffer }, },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer }, },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer }, },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer }, },
    ],
  });

  const render_shader = device.createShaderModule({
    label: 'white quad shader',
    code: renderWGSL,
  });

  const render_pipeline = device.createRenderPipeline({
    label: 'white quad pipeline',
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
          color: {
            operation: 'add',
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
          },
          alpha: {
            operation: 'add',
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
          },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'white quad camera and settings',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
    ],
  });

  const splat_bind_group = device.createBindGroup({
    label: 'render sorted splats',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer }, },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer }, },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const draw_args = new Uint32Array([6, 0, 0, 0]);

  const indirect_buffer = createBuffer(
    device,
    'white quad draw arguments',
    draw_args.byteLength,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    draw_args,
  );

  const render = (
    encoder: GPUCommandEncoder,
    texture_view: GPUTextureView,
  ) => {
    encoder.clearBuffer(sort_info_buffer, 0, 4);
    encoder.clearBuffer(sorter.sort_dispatch_indirect_buffer, 0, 4);

    const compute_pass = encoder.beginComputePass({
      label: 'cull and compact gaussians',
    });

    compute_pass.setPipeline(preprocess_pipeline);
    compute_pass.setBindGroup(0, preprocess_camera_bind_group);
    compute_pass.setBindGroup(1, preprocess_data_bind_group);
    compute_pass.setBindGroup(2, preprocess_count_bind_group);

    compute_pass.dispatchWorkgroups(
      Math.ceil(active_gaussian_count / preprocess_workgroup_size),
    );

    compute_pass.end();

    sorter.sort(encoder);

    encoder.copyBufferToBuffer(
      sort_info_buffer, 0,
      indirect_buffer, 4,
      4
    );

    const pass = encoder.beginRenderPass({
      label: 'white quad render',
      colorAttachments: [{
        view: texture_view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, splat_bind_group);
    pass.drawIndirect(indirect_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: render,
    camera_buffer,
    set_gaussian_scaling,
    set_sh_degree,
    set_frustum_culling,
    set_gaussian_stride
  };
}
