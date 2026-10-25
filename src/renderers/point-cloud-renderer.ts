import { PointCloud } from '../utils/load';
import pointcloud_wgsl from '../shaders/point_cloud.wgsl';
import { Renderer } from './renderer';


export interface PointCloudRenderer extends Renderer {
  setBenchmarkCapture: (enabled: boolean) => void;
  resolveBenchmark: (encoder: GPUCommandEncoder) => void;
  readBenchmarkMs: () => Promise<number>;
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer): PointCloudRenderer {

    const timestampSupported = device.features.has('timestamp-query');

    const timestampQuerySet = timestampSupported
      ? device.createQuerySet({
          type: 'timestamp',
          count: 2,
        })
      : undefined;

    const timestampResolveBuffer = timestampSupported
      ? device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        })
      : undefined;

    const timestampReadBuffer = timestampSupported
      ? device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
      : undefined;

    let benchmarkCapture = false;
  const render_shader = device.createShaderModule({code: pointcloud_wgsl});
  const render_pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'point-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'point cloud gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'point cloud render',

      ...(benchmarkCapture && timestampQuerySet
        ? {
            timestampWrites: {
              querySet: timestampQuerySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          }
        : {}),

      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);

    pass.draw(pc.num_points);
    pass.end();
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      render(encoder, texture_view);
    },

    camera_buffer,

    setBenchmarkCapture: (enabled: boolean) => {
      benchmarkCapture = enabled;
    },

    resolveBenchmark: (encoder: GPUCommandEncoder) => {
      if (!timestampQuerySet || !timestampResolveBuffer || !timestampReadBuffer) {
        throw new Error('GPU timestamp queries are unavailable.');
      }

      encoder.resolveQuerySet(
        timestampQuerySet,
        0,
        2,
        timestampResolveBuffer,
        0
      );

      encoder.copyBufferToBuffer(
        timestampResolveBuffer,
        0,
        timestampReadBuffer,
        0,
        16
      );
    },

    readBenchmarkMs: async () => {
      if (!timestampReadBuffer) {
        throw new Error('GPU timestamp queries are unavailable.');
      }

      await timestampReadBuffer.mapAsync(GPUMapMode.READ);

      try {
        const timestamps = new BigUint64Array(
          timestampReadBuffer.getMappedRange()
        );

        const elapsedNs = timestamps[1] - timestamps[0];

        return Number(elapsedNs) / 1_000_000;
      } finally {
        timestampReadBuffer.unmap();
      }
    },
  };
}