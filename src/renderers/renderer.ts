import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';
import { benchmark } from '../benchmark';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: Renderer | undefined; 
  let renderer: Renderer | undefined; 
  let cameras;
  let benchmarking = false;
  let point_count = 0;
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    if (benchmarking) return;
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);
  
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });
  

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    culling: true,
    half_precision: false,
    tile_sorting: false,
    workgroup_size: 256,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
    });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
      }
    }).on('change', (e) => {
      renderer = renderers[e.value];
    });
  }
  {
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        const pc = await load(uploadedFile, device);
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer.set_scale(params.gaussian_multiplier);
        gaussian_renderer.set_half_precision(params.half_precision);
        gaussian_renderer.set_tile_sorting(params.tile_sorting);
        gaussian_renderer.set_viewport(canvas.width, canvas.height);
        gaussian_renderer.set_culling(params.culling);
        gaussian_renderer.set_workgroup_size(params.workgroup_size);
        point_count = pc.num_points;
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
        };
        renderer = renderers[params.renderer];
        ply_file_loaded = true;
      }else{
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        cameras=await load_camera_presets(file.value);
        camera.set_preset(cameras[0]);
        cam_file_loaded = true;
      }else{
        cam_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(
      params,
      'gaussian_multiplier',
      {min: 0, max: 1.5}
    ).on('change', (e) => {
      gaussian_renderer?.set_scale(e.value);
    });
  }

  document.addEventListener('keydown', (event) => {
    if (benchmarking) return;
    switch(event.key) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        const i = parseInt(event.key);
        console.log(`set to camera preset ${i}`);
        if (cameras?.[i]) camera.set_preset(cameras[i]);
        break;
    }
  });

  pane.addInput(params, 'tile_sorting', { label: 'Tile sorting' }).on('change', e => {
    gaussian_renderer?.set_tile_sorting(e.value);
  });
  pane.addInput(params, 'half_precision', {
    label: 'Half precision',
    disabled: !device.features.has('shader-f16'),
  }).on('change', e => {
    gaussian_renderer?.set_half_precision(e.value);
  });
  pane.addInput(params, 'culling').on('change', e => {
    gaussian_renderer?.set_culling(e.value);
  });
  pane.addInput(params, 'workgroup_size', {
    options: { '64': 64, '128': 128, '256': 256 },
  }).on('change', e => {
    gaussian_renderer?.set_workgroup_size(e.value);
  });
  pane.addButton({ title: 'Save benchmark' }).on('click', async () => {
    if (!renderer || !ply_file_loaded || !cam_file_loaded || benchmarking) return;
    benchmarking = true;
    pane.disabled = true;
    const previous_pointer_events = canvas.style.pointerEvents;
    canvas.style.pointerEvents = 'none';
    try {
      gaussian_renderer?.set_viewport(canvas.width, canvas.height);
      const result = await benchmark(renderer, device, presentation_format, canvas.width, canvas.height);
      const report = {
        ...result,
        browser: navigator.userAgent,
        width: canvas.width,
        height: canvas.height,
        points: point_count,
        renderer: params.renderer,
        scale: params.gaussian_multiplier,
        culling: params.culling,
        half_precision: params.renderer === 'gaussian' && params.half_precision,
        tile_sorting: params.renderer === 'gaussian' && params.tile_sorting,
        workgroup_size: params.workgroup_size,
        camera_position: Array.from(camera.position),
        camera_rotation: Array.from(camera.rotation),
      };
      const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `benchmark-${params.renderer}-${point_count}-${Date.now()}.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      console.error('Benchmark failed:', error);
    } finally {
      canvas.style.pointerEvents = previous_pointer_events;
      pane.disabled = false;
      benchmarking = false;
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      camera.on_update_canvas();
    }
  });

  function frame() {
    if (ply_file_loaded && cam_file_loaded && !benchmarking) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      gaussian_renderer?.set_viewport(canvas.width, canvas.height);
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
