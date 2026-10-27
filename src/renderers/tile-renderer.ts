import tilesWGSL from '../shaders/tiles.wgsl';
import { get_sorter, SortStuff } from '../sort/sort';

const c_tile_size = 16;

// bindings in tiles.wgsl that each entry point uses
const c_bindings = {
  emit: [0, 1, 2, 3, 4, 5, 6, 7, 8],
  finish: [1, 6, 9],
  find_ranges: [1, 6, 7, 10],
  render: [0, 1, 2, 8, 10, 11],
  blit: [1, 11],
};

// tile-based depth sorting: every splat gets a key for each 16x16 tile it touches,
// the keys are sorted by tile and depth, and a compute shader blends each tile
export function create_tile_renderer(
  device: GPUDevice,
  num_points: number,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  splat_buffer: GPUBuffer,
  visible: SortStuff,
) {
  // room for 32 tile keys per gaussian on average, 2^25 keys at most
  const capacity = Math.min(num_points * 32, 1 << 25);
  const sorter = get_sorter(capacity, device);
  const nulling_data = new Uint32Array([0]);
  // tiles_x, tiles_y, width, height, depth_bits, capacity
  const info_buffer = device.createBuffer({
    label: 'tile info',
    size: 8 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const module = device.createShaderModule({ code: tilesWGSL });
  const compute = (entryPoint: string) => device.createComputePipeline({
    label: entryPoint,
    layout: 'auto',
    compute: { module, entryPoint },
  });
  const pipelines: Record<keyof typeof c_bindings, GPUComputePipeline | GPURenderPipeline> = {
    emit: compute('emit'),
    finish: compute('finish'),
    find_ranges: compute('find_ranges'),
    render: compute('render'),
    blit: device.createRenderPipeline({
      label: 'tile blit',
      layout: 'auto',
      vertex: { module, entryPoint: 'blit_vs' },
      fragment: { module, entryPoint: 'blit_fs', targets: [{ format: presentation_format }] },
      primitive: { topology: 'triangle-list' },
    }),
  };

  // these depend on the canvas size and are made again when it changes
  let width = 0;
  let height = 0;
  let tiles_x = 0;
  let tiles_y = 0;
  let ranges_buffer: GPUBuffer | undefined;
  let image_buffer: GPUBuffer | undefined;
  let groups = {} as Record<keyof typeof c_bindings, GPUBindGroup>;

  const resize = (w: number, h: number) => {
    width = w;
    height = h;
    tiles_x = Math.ceil(w / c_tile_size);
    tiles_y = Math.ceil(h / c_tile_size);
    const tile_bits = Math.ceil(Math.log2(tiles_x * tiles_y));
    device.queue.writeBuffer(info_buffer, 0, new Uint32Array([tiles_x, tiles_y, w, h, 32 - tile_bits, capacity]));

    ranges_buffer?.destroy();
    image_buffer?.destroy();
    ranges_buffer = device.createBuffer({
      label: 'tile ranges',
      size: tiles_x * tiles_y * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    image_buffer = device.createBuffer({ label: 'tile image', size: w * h * 4, usage: GPUBufferUsage.STORAGE });

    const buffers = [
      camera_buffer, info_buffer, splat_buffer,
      visible.sort_info_buffer, visible.ping_pong[0].sort_indices_buffer, visible.ping_pong[0].sort_depths_buffer,
      sorter.sort_info_buffer, sorter.ping_pong[0].sort_depths_buffer, sorter.ping_pong[0].sort_indices_buffer,
      sorter.sort_dispatch_indirect_buffer, ranges_buffer, image_buffer,
    ];
    for (const name of Object.keys(c_bindings) as (keyof typeof c_bindings)[]) {
      groups[name] = device.createBindGroup({
        label: `tile ${name}`,
        layout: pipelines[name].getBindGroupLayout(0),
        entries: c_bindings[name].map((b) => ({ binding: b, resource: { buffer: buffers[b] } })),
      });
    }
  };

  const dispatch = (pass: GPUComputePassEncoder, name: keyof typeof c_bindings) => {
    pass.setPipeline(pipelines[name] as GPUComputePipeline);
    pass.setBindGroup(0, groups[name]);
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture: GPUTexture, texture_view: GPUTextureView) => {
      if (texture.width != width || texture.height != height) {
        resize(texture.width, texture.height);
      }
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      encoder.clearBuffer(ranges_buffer!);

      const keys_pass = encoder.beginComputePass({ label: 'tile keys' });
      dispatch(keys_pass, 'emit');
      keys_pass.dispatchWorkgroups(Math.ceil(num_points / 256));
      dispatch(keys_pass, 'finish');
      keys_pass.dispatchWorkgroups(1);
      keys_pass.end();

      sorter.sort(encoder);

      const ranges_pass = encoder.beginComputePass({ label: 'tile ranges' });
      dispatch(ranges_pass, 'find_ranges');
      ranges_pass.dispatchWorkgroupsIndirect(sorter.sort_dispatch_indirect_buffer, 0);
      ranges_pass.end();

      const render_pass = encoder.beginComputePass({ label: 'tile render' });
      dispatch(render_pass, 'render');
      render_pass.dispatchWorkgroups(tiles_x, tiles_y);
      render_pass.end();

      const blit_pass = encoder.beginRenderPass({
        label: 'tile blit',
        colorAttachments: [{ view: texture_view, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }],
      });
      blit_pass.setPipeline(pipelines.blit as GPURenderPipeline);
      blit_pass.setBindGroup(0, groups.blit);
      blit_pass.draw(3);
      blit_pass.end();
    },
  };
}
