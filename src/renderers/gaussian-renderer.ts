import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_scaling: (value: number) => void,
  set_f16_preprocessing: (enabled: boolean) => void,
}

export interface GaussianRendererOptions {
  preprocess_workgroup_size?: number,
  frustum_culling?: boolean,
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
  options: GaussianRendererOptions = {},
): GaussianRenderer {

  const preprocess_workgroup_size = options.preprocess_workgroup_size ?? C.histogram_wg_size;
  if (!Number.isInteger(preprocess_workgroup_size) || preprocess_workgroup_size < 1 ||
      preprocess_workgroup_size > Math.min(device.limits.maxComputeWorkgroupSizeX, device.limits.maxComputeInvocationsPerWorkgroup)) {
    throw new Error('Unsupported preprocessing workgroup size');
  }
  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  // Packed sizes/color/opacity and f32 centers/conics, matching both shaders.
  const splat_buffer = createBuffer(device, '2d splats', pc.num_points * 32,
    GPUBufferUsage.STORAGE);
  const settings = new Float32Array([1, pc.sh_deg]);
  const settings_buffer = createBuffer(device, 'gaussian render settings', settings.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, settings);
  // DrawIndirect: vertex count, instance count, first vertex, first instance.
  const draw_indirect_buffer = createBuffer(device, 'splat draw indirect', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, new Uint32Array([4, 0, 0, 0]));

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const create_preprocess = (use_f16: boolean) => {
    const precision = use_f16 ? 'enable f16;\nalias Scalar = f16;\n' : 'alias Scalar = f32;\n';
    const preprocess_pipeline = device.createComputePipeline({
      label: use_f16 ? 'preprocess f16' : 'preprocess f32',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'gaussian preprocess', code: precision + preprocessWGSL }),
        entryPoint: 'preprocess',
        constants: {
          workgroupSize: preprocess_workgroup_size,
          sortWorkgroupSize: C.histogram_wg_size,
          sortKeyPerThread: c_histogram_block_rows,
          frustumCulling: options.frustum_culling === false ? 0 : 1,
        },
      },
    });

    const preprocess_camera_bind_group = device.createBindGroup({
      label: 'preprocess camera and settings',
      layout: preprocess_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: settings_buffer } },
      ],
    });

    const preprocess_gaussian_bind_group = device.createBindGroup({
      label: 'preprocess gaussian data',
      layout: preprocess_pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
        { binding: 1, resource: { buffer: pc.sh_buffer } },
        { binding: 2, resource: { buffer: splat_buffer } },
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

    return { preprocess_pipeline, preprocess_camera_bind_group, preprocess_gaussian_bind_group, sort_bind_group };
  };

  const preprocess_f32 = create_preprocess(false);
  const preprocess_f16 = device.features.has('shader-f16') ? create_preprocess(true) : undefined;
  let active_preprocess = preprocess_f16 ?? preprocess_f32;

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ label: 'gaussian render', code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
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
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  const render_camera_bind_group = device.createBindGroup({
    label: 'gaussian render camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });
  const render_splat_bind_group = device.createBindGroup({
    label: 'gaussian render splats',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      // Four radix passes leave their final output in ping_pong[0].
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    // Preserve the sorter's padded size and pass count; only visibility changes.
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
    const { preprocess_pipeline, preprocess_camera_bind_group, preprocess_gaussian_bind_group, sort_bind_group } = active_preprocess;
    const pass = encoder.beginComputePass({ label: 'preprocess gaussians' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / preprocess_workgroup_size));
    pass.end();

    // Copy the compacted visible count directly into DrawIndirect.instanceCount.
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, draw_indirect_buffer, 4, 4);
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [{
        view: texture_view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_camera_bind_group);
    pass.setBindGroup(1, render_splat_bind_group);
    pass.drawIndirect(draw_indirect_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    set_gaussian_scaling: (value: number) => {
      settings[0] = value;
      device.queue.writeBuffer(settings_buffer, 0, settings);
    },
    set_f16_preprocessing: (enabled: boolean) => {
      active_preprocess = enabled && preprocess_f16 ? preprocess_f16 : preprocess_f32;
    },
    camera_buffer,
  };
}
