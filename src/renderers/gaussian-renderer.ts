import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import preprocessF16WGSL from '../shaders/preprocess_f16.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { GPUTimer } from '../utils/gpu-timer';
import renderF16WGSL from '../shaders/gaussian_f16.wgsl';
const PREPROCESS_WG_SIZES = [64, 128, 256] as const;
type PreprocessWorkgroupSize = 64 | 128 | 256;

export interface GaussianRenderer extends Renderer {
  setGaussianMultiplier: (value: number) => void;
  setBenchmarkCapture: (enabled: boolean) => void;
  resolveBenchmark: (encoder: GPUCommandEncoder) => void;
  readBenchmarkMs: () => Promise<number>;
  readonly preprocessWorkgroupSize: number;
  setPreprocessWorkgroupSize: (size: number) => void;
  readonly frustumCullingEnabled: boolean;
  setFrustumCulling: (enabled: boolean) => void;
  readonly f16PreprocessingEnabled: boolean;
  setF16Preprocessing: (enabled: boolean) => void;
  readonly f16PreprocessingSupported: boolean;
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

  const gpuTimer = device.features.has('timestamp-query') ? new GPUTimer(device) : undefined;

  let benchmarkCapture = false;


  let frustumCullingEnabled = true;

  // Default to the f32 baseline; the f16 path is opt-in through setF16Preprocessing
  let useF16Preprocessing = false;
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================


  // Each intermediate splat occupies 48 bytes: vec2<f32> center at offset 0, vec3<f32> conic at offset 16, f32 opacity at offset 28, and vec3<f32> color at offset 32, including padding required by WGSL storage-buffer alignment
  const splat_buffer = createBuffer(
    device,
    'projected Gaussian splats',
    pc.num_points * 48,
    GPUBufferUsage.STORAGE
  );

  const splat_f16_buffer = createBuffer(
    device,
    'compressed projected Gaussian splats',
    pc.num_points * 32,
    GPUBufferUsage.STORAGE
  );

  const nulling_data = new Uint32Array([0]);





  // Gaussian rendering settings: gaussian_scaling and sh_deg
  // Allocate 16 bytes to satisfy uniform-buffer alignment requirements
  const render_settings_buffer = createBuffer(
    device,
    'gaussian render settings',
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg, 0.0, 0.0])
  );





  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipelines = new Map<string, GPUComputePipeline>();

  const preprocess_module = device.createShaderModule({
    label: 'gaussian preprocessing shader',
    code: preprocessWGSL,
  });


  const supportsShaderF16 = device.features.has('shader-f16');

  console.log('Gaussian preprocessing f16 available:', supportsShaderF16);

  const preprocess_f16_module = supportsShaderF16
  ? device.createShaderModule({
      label: 'gaussian f16 preprocessing shader',
      code: preprocessF16WGSL,
    })
  : undefined;



  if (preprocess_f16_module) {
    void preprocess_f16_module.getCompilationInfo().then((info) => {
      for (const message of info.messages) {
        console.log(
          `[f16 WGSL ${message.type}] line ${message.lineNum}: ${message.message}`
        );
      }

      if (info.messages.length === 0) {
        console.log('f16 WGSL compilation: no diagnostics');
      }
    });
  }



  const pipelineKey = (size: number, culling: boolean): string =>
  `${size}-${culling ? 'on' : 'off'}`;

  for (const size of PREPROCESS_WG_SIZES) {
    for (const culling of [true, false]) {
      const pipeline = device.createComputePipeline({
        label: `preprocess-${size}-culling-${culling ? 'on' : 'off'}`,
        layout: 'auto',
        compute: {
          module: preprocess_module,
          entryPoint: 'preprocess',
          constants: {
            workgroupSize: size,
            sortKeyPerThread: c_histogram_block_rows,
            shDegree: pc.sh_deg,
            enableFrustumCulling: culling ? 1 : 0,
          },
        },
      });

      preprocess_pipelines.set(pipelineKey(size, culling), pipeline);


      if (preprocess_f16_module) {
        const f16Pipeline = device.createComputePipeline({
          label: `preprocess-f16-${size}-culling-${culling ? 'on' : 'off'}`,
          layout: 'auto',
          compute: {
            module: preprocess_f16_module,
            entryPoint: 'preprocess',
            constants: {
              workgroupSize: size,
              sortKeyPerThread: c_histogram_block_rows,
              shDegree: pc.sh_deg,
              enableFrustumCulling: culling ? 1 : 0,
            },
          },
        });

        preprocess_pipelines.set(
          `${pipelineKey(size, culling)}-f16`,
          f16Pipeline
        );
      }
    }
  }



  let activePreprocessWorkgroupSize: PreprocessWorkgroupSize = 256;


  // Bind packed Gaussian attributes, camera matrices, projected splats, spherical harmonics coefficients, and render settings
  // These bindings correspond to @group(0) @binding(0..4) in preprocess.wgsl
  const preprocess_bind_groups = new Map<string, GPUBindGroup>();
  const sort_bind_groups = new Map<string, GPUBindGroup>();

  for (const size of PREPROCESS_WG_SIZES) {
    for (const culling of [true, false]) {
      const key = pipelineKey(size, culling);
      const pipeline = preprocess_pipelines.get(key)!;

      const preprocessBindGroup = device.createBindGroup({
        label: `gaussian preprocessing resources ${key}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
          { binding: 1, resource: { buffer: camera_buffer } },
          { binding: 2, resource: { buffer: splat_buffer } },
          { binding: 3, resource: { buffer: pc.sh_buffer } },
          { binding: 4, resource: { buffer: render_settings_buffer } },
        ],
      });

      const sortBindGroup = device.createBindGroup({
        label: `gaussian sort resources ${key}`,
        layout: pipeline.getBindGroupLayout(2),
        entries: [
          { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
          { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
          { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
          { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
        ],
      });

      preprocess_bind_groups.set(key, preprocessBindGroup);
      sort_bind_groups.set(key, sortBindGroup);



      if (supportsShaderF16) {
        const f16Key = `${key}-f16`;
        const f16Pipeline = preprocess_pipelines.get(f16Key)!;

        const f16PreprocessBindGroup = device.createBindGroup({
          label: `gaussian f16 preprocessing resources ${f16Key}`,
          layout: f16Pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
            { binding: 1, resource: { buffer: camera_buffer } },
            { binding: 2, resource: { buffer: splat_f16_buffer } },
            { binding: 3, resource: { buffer: pc.sh_buffer } },
            { binding: 4, resource: { buffer: render_settings_buffer } },
          ],
        });

        const f16SortBindGroup = device.createBindGroup({
          label: `gaussian f16 sort resources ${f16Key}`,
          layout: f16Pipeline.getBindGroupLayout(2),
          entries: [
            { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
            { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
            { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
          ],
        });

        preprocess_bind_groups.set(f16Key, f16PreprocessBindGroup);
        sort_bind_groups.set(f16Key, f16SortBindGroup);
      }
    }
  }


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  // Four u32 values required by WebGPU's drawIndirect command: vertexCount, instanceCount, firstVertex, firstInstance
  //
  // Draw six vertices per Gaussian (two triangles per quad)
  // The instance count is overwritten every frame with the visible-Gaussian count produced by preprocessing
  const indirect_draw_buffer = createBuffer(
    device,
    'gaussian indirect draw arguments',
    4 * Uint32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    new Uint32Array([6, pc.num_points, 0, 0])
  );

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
    targets: [{
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
    }],
  },
  primitive: {
    topology: 'triangle-list',
  },
});


// Create a rendering pipeline for compressed 32-byte splats
const render_f16_shader = supportsShaderF16
  ? device.createShaderModule({
      label: 'compressed gaussian render shader',
      code: renderF16WGSL,
    })
  : undefined;

const render_f16_pipeline = render_f16_shader
  ? device.createRenderPipeline({
      label: 'compressed gaussian render pipeline',
      layout: 'auto',

      vertex: {
        module: render_f16_shader,
        entryPoint: 'vs_main',
      },

      fragment: {
        module: render_f16_shader,
        entryPoint: 'fs_main',
        targets: [{
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
        }],
      },

      primitive: {
        topology: 'triangle-list',
      },
    })
  : undefined;



  // Bind the projected splats and depth-sorted indices to the vertex shader
  // These correspond to @group(0) @binding(0..1) in gaussian.wgsl
  const render_bind_group = device.createBindGroup({
    label: 'gaussian rendering resources',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: splat_buffer },
      },
      {
        binding: 1,
        resource: {
          buffer: sorter.ping_pong[0].sort_indices_buffer,
        },
      },
    ],
  });

  const render_f16_bind_group = render_f16_pipeline
  ? device.createBindGroup({
      label: 'compressed gaussian rendering resources',
      layout: render_f16_pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: splat_f16_buffer },
        },
        {
          binding: 1,
          resource: {
            buffer: sorter.ping_pong[0].sort_indices_buffer,
          },
        },
      ],
    })
  : undefined;
    

  // ===============================================
  //    Command Encoder Functions
  // ===============================================


  // Dispatch one preprocessing invocation per Gaussian
  // Preprocessing uses activePreprocessWorkgroupSize threads per workgroup, independently of the radix sort configuration
  // The shader performs a bounds check for excess invocations

  let loggedPreprocessMode = false;

  const preprocess = (encoder: GPUCommandEncoder) => {
    // Reset the visible-Gaussian counter before each preprocessing dispatch
    encoder.clearBuffer(sorter.sort_info_buffer, 0, Uint32Array.BYTES_PER_ELEMENT);


    // Reset the padded sorting length (second u32 in SortInfos)
    encoder.clearBuffer(sorter.sort_info_buffer, Uint32Array.BYTES_PER_ELEMENT, Uint32Array.BYTES_PER_ELEMENT);

    // Reset the indirect sorting workgroup count before each frame
    encoder.clearBuffer(sorter.sort_dispatch_indirect_buffer, 0, Uint32Array.BYTES_PER_ELEMENT);
    
    const pass = encoder.beginComputePass({
      label: 'gaussian preprocessing',
      ...(gpuTimer && benchmarkCapture ? {
        timestampWrites: {
          querySet: gpuTimer.querySet,
          beginningOfPassWriteIndex: 0,
        },
      } : {}),
    });



    // Select the f32 or f16 pipeline variant for this frame
    const useF16 = useF16Preprocessing;



    const key = `${pipelineKey(
      activePreprocessWorkgroupSize,
      frustumCullingEnabled
    )}${useF16 ? '-f16' : ''}`;


    if (!loggedPreprocessMode) {
      console.log('Active Gaussian preprocessing pipeline:', key);
      loggedPreprocessMode = true;
    }




    pass.setPipeline(preprocess_pipelines.get(key)!);
    pass.setBindGroup(0, preprocess_bind_groups.get(key)!);

    pass.setBindGroup(2, sort_bind_groups.get(key)!);



    // Round up to cover all Gaussians, including the final partial workgroup
    const workgroup_count = Math.ceil(pc.num_points / activePreprocessWorkgroupSize);



    pass.dispatchWorkgroups(workgroup_count);
    pass.end();


    // Copy the visible-Gaussian count into the indirect draw instance count
    encoder.copyBufferToBuffer(
      sorter.sort_info_buffer,
      0,
      indirect_draw_buffer,
      Uint32Array.BYTES_PER_ELEMENT,
      Uint32Array.BYTES_PER_ELEMENT
    );



  };

  // Record a render pass that draws Gaussian quads indirectly
  // The instance count is the visible-Gaussian count copied from preprocessing
  const render = (
    encoder: GPUCommandEncoder,
    texture_view: GPUTextureView
  ) => {
    
    const pass = encoder.beginRenderPass({
      label: 'gaussian quad render',
      ...(gpuTimer && benchmarkCapture ? {
        timestampWrites: {
          querySet: gpuTimer.querySet,
          endOfPassWriteIndex: 1,
        },
      } : {}),
      colorAttachments: [{
        view: texture_view,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });


    if (useF16Preprocessing && render_f16_pipeline && render_f16_bind_group) {
      
      pass.setPipeline(render_f16_pipeline);
      pass.setBindGroup(0, render_f16_bind_group);


    } else {
      
      pass.setPipeline(render_pipeline);
      pass.setBindGroup(0, render_bind_group);
    }



    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    

    get f16PreprocessingEnabled() {
      return useF16Preprocessing;
    },

    get f16PreprocessingSupported() {
      return supportsShaderF16;
    },

    setF16Preprocessing: (enabled: boolean) => {
      useF16Preprocessing = enabled && supportsShaderF16;
      loggedPreprocessMode = false;
    },

    get frustumCullingEnabled() {
      return frustumCullingEnabled;
    },

    setFrustumCulling: (enabled: boolean) => {
      frustumCullingEnabled = enabled;
    },

  
    
    
    get preprocessWorkgroupSize() {
      return activePreprocessWorkgroupSize;
    },



    setPreprocessWorkgroupSize: (size: number) => {
      if (!PREPROCESS_WG_SIZES.some((supported) => supported === size)) {
        throw new Error(`Unsupported preprocessing workgroup size: ${size}`);
      }

      activePreprocessWorkgroupSize = size as PreprocessWorkgroupSize;
    },

    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // Project Gaussians and generate depth keys
      preprocess(encoder);


      // Sort visible Gaussians by depth using the provided GPU radix sorter
      sorter.sort(encoder);

      

      // Render Gaussian splats
      render(encoder, texture_view);
    },
    camera_buffer,

    setBenchmarkCapture: (enabled: boolean) => {
      benchmarkCapture = enabled;
    },

    resolveBenchmark: (encoder: GPUCommandEncoder) => {
      if (!gpuTimer) {
        throw new Error('GPU timestamp queries are unavailable');
      }
      gpuTimer.resolve(encoder);
    },

    readBenchmarkMs: async () => {
      if (!gpuTimer) {
        throw new Error('GPU timestamp queries are unavailable');
      }

      const timestamps = await gpuTimer.read();
      return (timestamps[1] - timestamps[0]) / 1_000_000;
    },

    setGaussianMultiplier: (value: number) => {
      device.queue.writeBuffer(
        render_settings_buffer,
        0,
        new Float32Array([value])
      );
    }
  };
}
