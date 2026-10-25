import tilesWGSL from '../shaders/tiles.wgsl';
import presentWGSL from '../shaders/tile-present.wgsl';
import { SortStuff } from '../sort/sort';

export default function get_tile_renderer(device: GPUDevice, format: GPUTextureFormat,
  splats: GPUBuffer, sorter: SortStuff, point_count: number, half = false) {
  const module = device.createShaderModule({ code: half ? tilesWGSL.replace('alias SplatFloat = f32;', 'enable f16;\nalias SplatFloat = f16;') : tilesWGSL });
  const layout = device.createBindGroupLayout({ entries: [
    ...['read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage',
      'uniform', 'storage', 'storage'].map((type, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type as GPUBufferBindingType },
    })),
    { binding: 7, visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ] });
  const pipeline_layout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  function pipeline(entryPoint: string, constants: Record<string, number> = {}) {
    return device.createComputePipeline({ layout: pipeline_layout, compute: { module, entryPoint, constants } });
  }
  const pipelines = [pipeline('bin'), pipeline('prefix'), pipeline('bin', { stage: 1 }), pipeline('composite')];
  // An even number of merge passes leaves the final list in entries.
  const passes = Math.ceil(Math.ceil(Math.log2(Math.max(2, point_count))) / 2) * 2;
  const merges = Array.from({ length: passes }, (_, i) =>
    pipeline('merge_tiles', { run_size: 2 ** i, source_scratch: i % 2 }));
  const present_module = device.createShaderModule({ code: presentWGSL });
  const present = device.createRenderPipeline({ layout: 'auto',
    vertex: { module: present_module, entryPoint: 'vs_main' },
    fragment: { module: present_module, entryPoint: 'fs_main', targets: [{ format }] },
  });
  const grid = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let width = 0, height = 0, columns = 0, rows = 0;
  let counts: GPUBuffer, entries: GPUBuffer, scratch: GPUBuffer, output: GPUTexture;
  let bindings: GPUBindGroup, present_bindings: GPUBindGroup;
  function resize(w: number, h: number) {
    if (w === width && h === height) return true;
    const cols = Math.ceil(w / 16), tile_rows = Math.ceil(h / 16);
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const bytes = Math.min(limit, Math.max(1024, point_count * 32) * 8);
    counts?.destroy();
    entries?.destroy();
    scratch?.destroy();
    output?.destroy();
    width = w;
    height = h;
    columns = cols;
    rows = tile_rows;
    counts = device.createBuffer({ size: (columns * rows * 2) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    entries = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    scratch = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    output = device.createTexture({ size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    device.queue.writeBuffer(grid, 0, new Uint32Array([width, height, columns, rows]));
    bindings = device.createBindGroup({ layout, entries: [
      ...[splats, sorter.ping_pong[0].sort_indices_buffer, sorter.ping_pong[0].sort_depths_buffer,
        sorter.sort_info_buffer, grid, counts, entries].map((buffer, binding) => ({ binding, resource: { buffer } })),
      { binding: 7, resource: output.createView() },
      { binding: 8, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      { binding: 9, resource: { buffer: scratch } },
    ] });
    present_bindings = device.createBindGroup({ layout: present.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: output.createView() }] });
    return true;
  }
  function dispatch(encoder: GPUCommandEncoder, stage: number, x: number, y = 1) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines[stage]);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }
  return {
    resize,
    destroy: () => {
      counts?.destroy();
      entries?.destroy();
      scratch?.destroy();
      output?.destroy();
      grid.destroy();
    },
    bin: (encoder: GPUCommandEncoder) => {
      encoder.clearBuffer(counts);
      dispatch(encoder, 0, Math.ceil(point_count / 256));
      dispatch(encoder, 1, 1);
      dispatch(encoder, 2, Math.ceil(point_count / 256));
      for (const pipeline of merges) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(columns, rows);
        pass.end();
      }
    },
    frame: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
      dispatch(encoder, 3, Math.ceil(width / 8), Math.ceil(height / 8));
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear',
        storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(present);
      pass.setBindGroup(0, present_bindings);
      pass.draw(3);
      pass.end();
    },
  };
}
