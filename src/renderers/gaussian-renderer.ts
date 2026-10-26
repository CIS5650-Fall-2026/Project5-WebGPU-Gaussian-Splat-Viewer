import { PointCloud } from '../utils/load';
import preprocessCommonWGSL from '../shaders/preprocess_common.wgsl';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import preprocessEntriesWGSL from '../shaders/preprocess_entries.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
  set_frustum_culling: (enabled: boolean) => void,
  set_workgroup_size: (size: number) => void,
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

// keep in sync with `Splat` in the shaders
const c_size_splat = 8 * 4;
const c_size_render_settings = 4 * 4;
const c_max_workgroups_per_dim = 65535;

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
    device, 'splats', pc.num_points * c_size_splat, GPUBufferUsage.STORAGE,
  );

  // vertex_count, instance_count, first_vertex, first_instance
  const indirect_draw_buffer = createBuffer(
    device, 'indirect draw', 4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([4, pc.num_points, 0, 0]),
  );

  // gaussian_scaling, sh_deg, frustum_culling, key_capacity
  const render_settings = new Float32Array([1.0, pc.sh_deg, 1.0, 0.0]);
  const render_settings_buffer = createBuffer(
    device, 'render settings', c_size_render_settings,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings,
  );
  const update_render_settings = () => device.queue.writeBuffer(render_settings_buffer, 0, render_settings);

  // only used by the tile renderer, but part of the shared preprocess layout
  const splat_counter_buffer = createBuffer(
    device, 'splat counter', 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const buffer_entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type },
  });

  // explicit layouts so the bind groups can be shared by all preprocess entry points
  const preprocess_layouts = [
    device.createBindGroupLayout({
      label: 'preprocess camera + settings',
      entries: [buffer_entry(0, 'uniform'), buffer_entry(1, 'uniform')],
    }),
    device.createBindGroupLayout({
      label: 'preprocess gaussians',
      entries: [buffer_entry(0, 'read-only-storage'), buffer_entry(1, 'read-only-storage'), buffer_entry(2, 'storage')],
    }),
    device.createBindGroupLayout({
      label: 'preprocess sort',
      entries: [0, 1, 2, 3, 4].map(i => buffer_entry(i, 'storage')),
    }),
  ];
  const preprocess_pipeline_layout = device.createPipelineLayout({
    label: 'preprocess',
    bindGroupLayouts: preprocess_layouts,
  });

  const preprocess_module = device.createShaderModule({
    label: 'preprocess',
    code: preprocessCommonWGSL + preprocessWGSL + preprocessEntriesWGSL,
  });

  let workgroup_size = C.histogram_wg_size;
  const create_preprocess_pipeline = (entryPoint: string) => device.createComputePipeline({
    label: `preprocess ${entryPoint}`,
    layout: preprocess_pipeline_layout,
    compute: {
      module: preprocess_module,
      entryPoint,
      constants: {
        workgroupSize: workgroup_size,
        sortKeysPerDispatch: C.histogram_wg_size * c_histogram_block_rows,
      },
    },
  });
  let preprocess_pipeline = create_preprocess_pipeline('preprocess');

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_layouts[0],
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_layouts[1],
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: pc.sh_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_layouts[2],
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      { binding: 4, resource: { buffer: splat_counter_buffer } },
    ],
  });


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
        // splats are drawn back to front => standard "over" blending
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

  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      // sorting ends after an even number of passes => the result is back in ping_pong[0]
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const dispatch_preprocess = (encoder: GPUCommandEncoder, pipeline: GPUComputePipeline) => {
    const num_workgroups = Math.ceil(pc.num_points / workgroup_size);
    const x = Math.min(num_workgroups, c_max_workgroups_per_dim);
    const y = Math.ceil(num_workgroups / x);

    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  };

  const preprocess = (encoder: GPUCommandEncoder) => {
    // reset the visible gaussian count and the sort dispatch size
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
    dispatch_preprocess(encoder, preprocess_pipeline);
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    // instance count = number of visible gaussians written by the preprocess step
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
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
    camera_buffer,

    set_gaussian_multiplier: (value: number) => {
      render_settings[0] = value;
      update_render_settings();
    },
    set_frustum_culling: (enabled: boolean) => {
      render_settings[2] = enabled ? 1.0 : 0.0;
      update_render_settings();
    },
    set_workgroup_size: (size: number) => {
      workgroup_size = size;
      preprocess_pipeline = create_preprocess_pipeline('preprocess');
    },
  };
}
