import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

//interface: describes properties, types and methods of an object
export interface GaussianRenderer extends Renderer {
  rendersettings_buffer: GPUBuffer //
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

  const gaussian_render_shader = device.createShaderModule({code:renderWGSL});
  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================
  const indirect_draw_buffer = device.createBuffer( //indirect draw command points to this buffer  - can be populated in the GPU (points to where the draw numbers are stored, isntead of containing the nubmers itself)
    {label: "indirect draw buffer", 
      size: 4*4, //size is fixed
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST});

  const splat_buffer = device.createBuffer(
    {label: "splat buffer",
    size: pc.num_points * 64,
    usage: GPUBufferUsage.STORAGE});

  const rendersettings_buffer = device.createBuffer(
    {label: "render settings buffer",
    size: 4*2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  //[vertices per instance, number of instances, first vertex, first instance] (4x4)
  //this indirect rednering uses instanced rendering, but they're separate concepts
  device.queue.writeBuffer(indirect_draw_buffer, 0, new Uint32Array([4, pc.num_points, 0, 0]));
  device.queue.writeBuffer(rendersettings_buffer, 0, new Float32Array([1.0, pc.sh_deg]));

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
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const camera_compute_bind_group = device.createBindGroup({
    label:'compute camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [{binding:0, resource: {buffer: camera_buffer}}]
  });

  const gaussian_compute_bind_group = device.createBindGroup({
    label: 'compute gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries:[
      {binding: 0, resource: {buffer:pc.gaussian_3d_buffer}},
      {binding: 1, resource: {buffer:pc.sh_buffer}}
    ]
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

  const splat_bind_group = device.createBindGroup({
    label: 'splat bind group',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries:[
      {binding: 0, resource: {buffer: splat_buffer}},
      {binding: 1, resource: {buffer: rendersettings_buffer}}
    ]
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const quad_render_pipeline = device.createRenderPipeline({
      label: "gaussian render pipeline",
      layout: 'auto',
      vertex: {
        module: gaussian_render_shader,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: gaussian_render_shader,
        entryPoint: 'fs_main',
        targets: [
          {format: presentation_format,
            blend:{
              color:{
                srcFactor: 'src-alpha', //fragment we're drawing right now
                dstFactor: 'one-minus-src-alpha', //pixel already in the render target
                operation: 'add', //combine (colors*alhpas + colord*(1-alphas))
              },
              alpha: { //controls resulting opacity
                srcFactor: 'one', //keep the incoming alpha as is
                dstFactor: 'one-minus-src-alpha', //retain the existing alpha in proportion to transparency (1 - incoming alpha)
                operation: 'add', //combine (alphas + alphad(1 - alphas))
              },
            },
        }], //pixel format of the canvas texture being rendered to
      },
      primitive: {
        topology: 'triangle-strip'
      }
  });

  const camera_render_bind_group = device.createBindGroup({
    label: 'render camera',
    layout: quad_render_pipeline.getBindGroupLayout(0),
    entries:[
      {binding: 0, resource: {buffer: camera_buffer}}
    ]
  })

  const splat_render_bind_group = device.createBindGroup({
    label: 'gaussian render bind group',
    layout: quad_render_pipeline.getBindGroupLayout(1),
    entries:[
      {binding: 1, resource: {buffer: splat_buffer}},
      //4 passes, so it's gauranteed that ping_pong[0] is the final written buffer
      {binding: 2, resource: {buffer: sorter.ping_pong[0].sort_indices_buffer}} //each element has both sort_indices_buffer and sort_depths_buffers, there are 2 ping pong buffers
    ]
  });
  
  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const render = (encoder:GPUCommandEncoder, texture_view:GPUTextureView) => {
    const preprocess_pass = encoder.beginComputePass({
      label: 'preprocess gaussians compute',
    });
    preprocess_pass.setPipeline(preprocess_pipeline);
    preprocess_pass.setBindGroup(0, camera_compute_bind_group);
    preprocess_pass.setBindGroup(1, gaussian_compute_bind_group);
    preprocess_pass.setBindGroup(2, sort_bind_group)
    preprocess_pass.setBindGroup(3, splat_bind_group);

    let workgroupSize = Math.ceil(pc.num_points/C.histogram_wg_size);
    preprocess_pass.dispatchWorkgroups(workgroupSize, 1, 1); //like a draw call but for compute shaders (almost like CUDA)
    preprocess_pass.end();

    //mem source is first this time, offset(keys_size is the first field), destination, destination offset, memcpy size
    sorter.sort(encoder);
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian renderpass',
      colorAttachments:[
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(quad_render_pipeline);
    pass.setBindGroup(0, camera_render_bind_group);
    pass.setBindGroup(1, splat_render_bind_group);

    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  }

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      //reset atomics to 0 every frame to support dynamic splat counts
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, new Uint32Array([0, 1, 1]));
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, new Uint32Array([0]));
      
      //sorter.sort(encoder); //moved inbetween preprocess and render
      render(encoder, texture_view);
    },
    camera_buffer,
    rendersettings_buffer //javascript variables are passed by reference
  };
}
