import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer, GaussianRenderMode, PreprocessPrecision } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
  // optional profiling hooks
  after_submit?: () => void,
  stats_text?: () => string,
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
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
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
    renderer: 'pointcloud',
    gaussian_mode: 'quad' as GaussianRenderMode,
    gpu_stats: '',
    frustum_culling: true,
    workgroup_size: 256,
    precision: 'f32' as PreprocessPrecision,
    ply_file: '',
    cam_file: '',
  };

  // re-applies the GUI state when a new scene creates a new gaussian renderer
  function apply_gaussian_params(r: GaussianRenderer) {
    r.set_gaussian_multiplier(params.gaussian_multiplier);
    r.set_frustum_culling(params.frustum_culling);
    r.set_workgroup_size(params.workgroup_size);
    r.set_mode(params.gaussian_mode);
    r.set_precision(params.precision);
  }

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
    // quad: sorted instanced quads + hardware blending, tile: tile based sort + compute rasterizer
    pane.addInput(params, 'gaussian_mode', {
      options: { quad: 'quad', tile: 'tile' },
    }).on('change', (e) => {
      gaussian_renderer?.set_mode(e.value);
    });
  }
  {
    pane.addMonitor(params, 'gpu_stats', {
      multiline: true,
      lineCount: 7,
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
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer, canvas);
        apply_gaussian_params(gaussian_renderer);
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
      gaussian_renderer?.set_gaussian_multiplier(e.value);
    });
  }
  {
    pane.addInput(params, 'frustum_culling').on('change', (e) => {
      gaussian_renderer?.set_frustum_culling(e.value);
    });
  }
  {
    pane.addInput(params, 'workgroup_size', {
      options: { '32': 32, '64': 64, '128': 128, '256': 256 },
    }).on('change', (e) => {
      gaussian_renderer?.set_workgroup_size(e.value);
    });
  }
  {
    // preprocess compute precision (f16 needs the `shader-f16` feature)
    const precision_input = pane.addInput(params, 'precision', {
      options: { f32: 'f32', f16: 'f16' },
    });
    precision_input.disabled = !device.features.has('shader-f16');
    precision_input.on('change', (e) => {
      gaussian_renderer?.set_precision(e.value);
    });
  }

  document.addEventListener('keydown', (event) => {
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
        camera.set_preset(cameras[i]);
        break;
    }
  });

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);
      renderer.after_submit?.();
      params.gpu_stats = renderer.stats_text?.() ?? '';
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
