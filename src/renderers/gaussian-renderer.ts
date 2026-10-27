import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import preprocessF16WGSL from '../shaders/preprocess-f16.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  settings_buffer: GPUBuffer,
  count_buffer: GPUBuffer,
  set_options: (workgroup_size: number, frustum_culling: boolean, point_count?: number, precision?: 'f32' | 'f16') => void,
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {
  const settings_buffer = device.createBuffer({
    label: 'render settings',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(
    settings_buffer, 0,
    new Float32Array([1, pc.sh_deg, 0, 0])
  );

  device.queue.writeBuffer(settings_buffer, 8, new Uint32Array([1, pc.num_points]));
  let workgroup_size = C.histogram_wg_size;
  let point_count = pc.num_points;

  const splat_buffer = device.createBuffer({
    label: '2d splats',
    size: pc.num_points * 64,
    usage: GPUBufferUsage.STORAGE,
  });

  const sorter = get_sorter(pc.num_points, device);
  const sort_info_buffer = sorter.sort_info_buffer;

  const preprocess_layout = device.createPipelineLayout({
    bindGroupLayouts: [
      device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      }),
      device.createBindGroupLayout({ entries: [] }),
      device.createBindGroupLayout({
        entries: [0, 1, 2, 3].map(binding => ({
          binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as GPUBufferBindingType },
        })),
      }),
    ],
  });
  const preprocess_module = device.createShaderModule({ code: preprocessWGSL });
  let preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: preprocess_layout,
    compute: {
      module: preprocess_module,
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeysPerDispatch: C.histogram_wg_size * c_histogram_block_rows,
        shDegree: pc.sh_deg,
      },
    },
  });

  const preprocess_bind_group = device.createBindGroup({
    label: 'preprocess data',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 2, resource: { buffer: splat_buffer } },
      { binding: 3, resource: { buffer: pc.sh_buffer } },
      { binding: 4, resource: { buffer: settings_buffer } },
    ],
  });

  const empty_bind_group = device.createBindGroup({
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'preprocess count',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sort_info_buffer } },
      { binding: 1, resource: {
        buffer: sorter.ping_pong[0].sort_depths_buffer,
      } },
      { binding: 2, resource: {
        buffer: sorter.ping_pong[0].sort_indices_buffer,
      } },
      { binding: 3, resource: {
        buffer: sorter.sort_dispatch_indirect_buffer,
      } },
    ],
  });

  const preprocess_modules = new Map<string, GPUShaderModule>([['f32', preprocess_module]]);
  const preprocess_pipelines = new Map([[`f32:${workgroup_size}`, preprocess_pipeline]]);

  function set_options(size: number, culling: boolean, count = pc.num_points, precision: 'f32' | 'f16' = 'f32') {
    if (![64, 128, 256].includes(size)) {
      throw new Error('Unsupported preprocess workgroup size');
    }
    if (precision === 'f16' && !device.features.has('shader-f16')) {
      throw new Error('Half-precision shaders are unavailable on this device');
    }
    if (!preprocess_modules.has(precision)) {
      preprocess_modules.set(precision, device.createShaderModule({ code: preprocessF16WGSL }));
    }
    const key = `${precision}:${size}`;
    if (!preprocess_pipelines.has(key)) {
      preprocess_pipelines.set(key, device.createComputePipeline({
        label: 'preprocess',
        layout: preprocess_layout,
        compute: {
          module: preprocess_modules.get(precision),
          entryPoint: 'preprocess',
          constants: {
            workgroupSize: size,
            sortKeysPerDispatch: C.histogram_wg_size * c_histogram_block_rows,
            shDegree: pc.sh_deg,
          },
        },
      }));
    }
    workgroup_size = size;
    point_count = Math.max(0, Math.min(pc.num_points, Math.floor(count)));
    preprocess_pipeline = preprocess_pipelines.get(key);
    device.queue.writeBuffer(settings_buffer, 8, new Uint32Array([Number(culling), point_count]));
  }

  const shader = device.createShaderModule({
    label: 'gaussian quad shader',
    code: renderWGSL,
  });

  shader.getCompilationInfo().then((info) => {
    for (const message of info.messages) {
      console.log(
        message.type,
        `gaussian.wgsl:${message.lineNum}`,
        message.message
      );
    }
  });

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian quad pipeline',
    layout: 'auto',
    vertex: {
      module: shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shader,
      entryPoint: 'fs_main',
      targets: [
        {
          format: presentation_format,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
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
      topology: 'triangle-strip',
      cullMode: 'none',
    },
  });

  const splat_bind_group = device.createBindGroup({
    label: 'splat render data',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: {
        buffer: sorter.ping_pong[0].sort_indices_buffer,
      } },
    ],
  });

  // vertexCount, instanceCount, firstVertex, firstInstance
  const draw_args = new Uint32Array([
    4,
    0,
    0,
    0,
  ]);

  const indirect_buffer = device.createBuffer({
    label: 'gaussian indirect draw',
    size: draw_args.byteLength,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(indirect_buffer, 0, draw_args);

  return {
    frame: (
      encoder: GPUCommandEncoder,
      texture_view: GPUTextureView,
      query_set?: GPUQuerySet,
    ) => {
      encoder.clearBuffer(sort_info_buffer, 0, 4);
      encoder.clearBuffer(
        sorter.sort_dispatch_indirect_buffer, 0, 4
      );

      const compute_pass = encoder.beginComputePass({
        label: 'preprocess pass',
        timestampWrites: query_set ? {
          querySet: query_set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
        } : undefined,
      });
      compute_pass.setPipeline(preprocess_pipeline);
      compute_pass.setBindGroup(0, preprocess_bind_group);
      compute_pass.setBindGroup(1, empty_bind_group);
      compute_pass.setBindGroup(2, sort_bind_group);
      compute_pass.dispatchWorkgroups(
        Math.ceil(point_count / workgroup_size)
      );
      compute_pass.end();

      sorter.sort(encoder);

      encoder.copyBufferToBuffer(
        sort_info_buffer, 0,
        indirect_buffer, 4,
        4
      );

      const pass = encoder.beginRenderPass({
        label: 'gaussian quad pass',
        timestampWrites: query_set ? {
          querySet: query_set, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3,
        } : undefined,
        colorAttachments: [
          {
            view: texture_view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      pass.setPipeline(render_pipeline);
      pass.setBindGroup(0, splat_bind_group);
      pass.drawIndirect(indirect_buffer, 0);
      pass.end();
    },

    camera_buffer,
    settings_buffer,
    count_buffer: sort_info_buffer,
    set_options,
  };
}