import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud, type PointCloudRenderer, } from './point-cloud-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';

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
  let sceneName = '';
  let loadedGaussianFraction = 1.0;
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: PointCloudRenderer | undefined;
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
  

  // Tweakpane: adding tweak control for parameters
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    gaussian_fraction: 1.0,
    renderer: 'pointcloud',
    preprocess_precision: 'f32',
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
    pane.addInput(params, 'gaussian_fraction', {
      options: {
        '25%': 0.25,
        '50%': 0.50,
        '75%': 0.75,
        '100%': 1.00,
      },
    });
  }
  {
    pane.addInput(params, 'preprocess_precision', {
      label: 'precision',
      // Only offer f16 when the device supports it, so the GUI never shows a mode the GPU is not running
      options: device.features.has('shader-f16')
        ? { 'f32 (baseline)': 'f32', 'f16 (experimental)': 'f16' }
        : { 'f32 (baseline)': 'f32' },
    }).on('change', (e) => {
      gaussian_renderer?.setF16Preprocessing(e.value === 'f16');
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
      console.log('Uploaded PLY value:', uploadedFile);
      console.log('Uploaded PLY type:', typeof uploadedFile);
      if (uploadedFile) {
        sceneName = (uploadedFile as unknown as File).name.replace(/\.ply$/i, '').toLowerCase().replace(/[^a-z0-9_-]/g, '_');
        console.log('Benchmark scene name:', sceneName);

        const requestedFraction = params.gaussian_fraction;
        const pc = await load(uploadedFile, device, requestedFraction);
        loadedGaussianFraction = requestedFraction;
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer.setF16Preprocessing(params.preprocess_precision === 'f16');
        // A new renderer starts with gaussian_scaling = 1.0; re-apply the GUI value
        gaussian_renderer.setGaussianMultiplier(params.gaussian_multiplier);
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
      //TODO: Bind constants to the gaussian renderer.
      gaussian_renderer?.setGaussianMultiplier(e.value);
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


  const benchmark = {
      running: false,

      mode: 'workgroup' as 'workgroup' | 'culling' | 'comparison' | 'precision',

      warmupRemaining: 30,
      samplesRemaining: 100,
      gpuTimesMs: [] as number[],
      readbackPending: false,

      workgroupSizes: [64, 128, 256] as number[],
      workgroupIndex: 0,

      cullingModes: [true, false],
      cullingIndex: 0,

      comparisonRenderers: ['pointcloud', 'gaussian'] as const,
      comparisonIndex: 0,
      precisionModes: [false, true],
      precisionIndex: 0,
    };

    (window as Window & {
      startGaussianBenchmark?: () => void;
    }).startGaussianBenchmark = () => {
      if (!gaussian_renderer || !ply_file_loaded || !cam_file_loaded) {
        console.error('Load the PLY and camera files first.');
        return;
      }

      if (!device.features.has('timestamp-query')) {
        console.error('GPU timestamp queries are unavailable.');
        return;
      }

      if (benchmark.running) {
        console.warn('A benchmark is already running.');
        return;
      }

      params.renderer = 'gaussian';
      renderer = gaussian_renderer;

      benchmark.mode = 'workgroup';
      gaussian_renderer.setFrustumCulling(true);

      benchmark.workgroupIndex = 0;
      gaussian_renderer.setPreprocessWorkgroupSize(
        benchmark.workgroupSizes[benchmark.workgroupIndex]
      );

      benchmark.running = true;
      benchmark.warmupRemaining = 30;
      benchmark.samplesRemaining = 100;
      benchmark.gpuTimesMs = [];
      benchmark.readbackPending = false;

      console.log('Automatic Gaussian benchmark started: 64, 128, 256 threads; ' + '30 warm-up frames and 100 samples per configuration.');
    };


    (window as Window & {
      startGaussianCullingBenchmark?: () => void;
    }).startGaussianCullingBenchmark = () => {
      if (!gaussian_renderer || !ply_file_loaded || !cam_file_loaded) {
        console.error('Load the PLY and camera files first.');
        return;
      }

      if (!device.features.has('timestamp-query')) {
        console.error('GPU timestamp queries are unavailable.');
        return;
      }

      if (benchmark.running) {
        console.warn('A benchmark is already running.');
        return;
      }

      params.renderer = 'gaussian';
      renderer = gaussian_renderer;

      benchmark.mode = 'culling';
      benchmark.cullingIndex = 0;

      gaussian_renderer.setPreprocessWorkgroupSize(256);
      gaussian_renderer.setFrustumCulling(true);

      benchmark.running = true;
      benchmark.warmupRemaining = 30;
      benchmark.samplesRemaining = 100;
      benchmark.gpuTimesMs = [];
      benchmark.readbackPending = false;

      console.log(
        'Automatic frustum-culling benchmark started: ON, OFF; ' +
        '256 threads; 30 warm-up frames and 100 samples per configuration.'
      );
    };


    (window as Window & {
      startRendererComparisonBenchmark?: () => void;
    }).startRendererComparisonBenchmark = () => {
      if (
        !pointcloud_renderer ||
        !gaussian_renderer ||
        !ply_file_loaded ||
        !cam_file_loaded
      ) {
        console.error('Load the PLY and camera files first.');
        return;
      }

      if (!device.features.has('timestamp-query')) {
        console.error('GPU timestamp queries are unavailable.');
        return;
      }

      if (benchmark.running) {
        console.warn('A benchmark is already running.');
        return;
      }

      benchmark.mode = 'comparison';
      benchmark.comparisonIndex = 0;

      // Use a fixed Gaussian configuration for the comparison
      gaussian_renderer.setPreprocessWorkgroupSize(256);
      gaussian_renderer.setFrustumCulling(true);

      // Begin with the point-cloud renderer
      params.renderer = 'pointcloud';
      renderer = pointcloud_renderer;

      benchmark.running = true;
      benchmark.warmupRemaining = 30;
      benchmark.samplesRemaining = 100;
      benchmark.gpuTimesMs = [];
      benchmark.readbackPending = false;

      console.log(
        'Renderer comparison started: pointcloud, then gaussian; ' +
        '30 warm-up frames and 100 GPU samples per renderer.'
      );
    };

    (window as Window & {
    startGaussianPrecisionBenchmark?: () => void;
  }).startGaussianPrecisionBenchmark = () => {

    if (!gaussian_renderer || !ply_file_loaded || !cam_file_loaded) {
      console.error('Load the PLY and camera files first.');
      return;
    }

    if (!device.features.has('timestamp-query')) {
      console.error('GPU timestamp queries are unavailable.');
      return;
    }

    if (!gaussian_renderer.f16PreprocessingSupported) {
      console.error('GPU shader-f16 is unavailable.');
      return;
    }

    if (benchmark.running) {
      console.warn('A benchmark is already running.');
      return;
    }

    benchmark.mode = 'precision';
    benchmark.precisionIndex = 0;

    params.renderer = 'gaussian';
    renderer = gaussian_renderer;

    gaussian_renderer.setPreprocessWorkgroupSize(256);
    gaussian_renderer.setFrustumCulling(true);
    gaussian_renderer.setF16Preprocessing(false);

    params.preprocess_precision = 'f32';
    pane.refresh();

    benchmark.running = true;
    benchmark.warmupRemaining = 30;
    benchmark.samplesRemaining = 100;
    benchmark.gpuTimesMs = [];
    benchmark.readbackPending = false;

    console.log(
      'Precision benchmark started: f32, then f16; ' +
      '256 threads, culling ON, 30 warm-up frames, 100 GPU samples per mode.'
    );
  };

  
  function frame() {
    if (ply_file_loaded && cam_file_loaded && renderer) {
      params.fps = 1.0 / timeReturn() * 1000.0;
      
      time();



      const encoder = device.createCommandEncoder();

      const texture_view = context.getCurrentTexture().createView();




      const capture = benchmark.running && benchmark.warmupRemaining === 0 && benchmark.samplesRemaining > 0 && !benchmark.readbackPending;



      const measuredRenderer = benchmark.mode === 'comparison' && benchmark.comparisonRenderers[benchmark.comparisonIndex] === 'pointcloud' ? pointcloud_renderer : gaussian_renderer;



      if (benchmark.running && benchmark.warmupRemaining > 0) {
        benchmark.warmupRemaining--;
      }




      if (capture && measuredRenderer) {
        measuredRenderer.setBenchmarkCapture(true);

        // benchmark.readbackPending = true;
      }



      renderer.frame(encoder, texture_view);



      if (capture && measuredRenderer) {
        measuredRenderer.setBenchmarkCapture(false);
        measuredRenderer.resolveBenchmark(encoder);
      }




      device.queue.submit([encoder.finish()]);



      if (capture && measuredRenderer) {
        benchmark.readbackPending = true;



        benchmark.samplesRemaining--;





        measuredRenderer.readBenchmarkMs().then((ms) => {
          benchmark.readbackPending = false;
          benchmark.gpuTimesMs.push(ms);
          
          
          
          console.log(`GPU sample ${benchmark.gpuTimesMs.length}/100: ${ms.toFixed(3)} ms`);



          if (benchmark.samplesRemaining === 0) {
            const values = benchmark.gpuTimesMs;
            const mean = values.reduce((a, b) => a + b, 0) / values.length;



            
            let configuration: string;

            if (benchmark.mode === 'precision') {
              configuration = benchmark.precisionModes[benchmark.precisionIndex]
                ? 'precision_f16_wg256_culling_on'
                : 'precision_f32_wg256_culling_on';

            } else if (benchmark.mode === 'comparison') {
              const rendererName = benchmark.comparisonRenderers[benchmark.comparisonIndex];



              configuration = rendererName === 'pointcloud'
                ? 'pointcloud'
                : 'gaussian_wg256_culling_on';




            } else if (benchmark.mode === 'workgroup') {
              configuration = `wg${gaussian_renderer!.preprocessWorkgroupSize}`;

            } else {
              configuration = `culling_${gaussian_renderer!.frustumCullingEnabled ? 'on' : 'off'}_wg256`;
            }

            console.log(`${configuration}: Mean GPU time = ${mean.toFixed(3)} ms`);




            const csv = [
              'sample,gpu_time_ms',
              ...values.map((ms, i) => `${i + 1},${ms.toFixed(6)}`),
            ].join('\n');





            const blob = new Blob([csv], {
              type: 'text/csv;charset=utf-8',
            });



            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');



            link.href = url;
            link.download = `${sceneName}_${Math.round(loadedGaussianFraction * 100)}pct_${configuration}.csv`;
            document.body.appendChild(link);
            link.click();
            link.remove();



            setTimeout(() => URL.revokeObjectURL(url), 1000);

            console.log(`Benchmark CSV exported: ${configuration}`);


            if (benchmark.mode === 'precision') {
              benchmark.precisionIndex++;

              if (benchmark.precisionIndex < benchmark.precisionModes.length) {
                const nextF16 = benchmark.precisionModes[benchmark.precisionIndex];



                gaussian_renderer!.setF16Preprocessing(nextF16);
                params.preprocess_precision = nextF16 ? 'f16' : 'f32';
                pane.refresh();




                benchmark.warmupRemaining = 30;
                benchmark.samplesRemaining = 100;
                benchmark.gpuTimesMs = [];
                benchmark.readbackPending = false;



                console.log(`Starting ${nextF16 ? 'f16' : 'f32'} GPU benchmark...`);
              } else {
                benchmark.running = false;
                console.log('Both precision benchmarks completed.');
              }

            } else if (benchmark.mode === 'workgroup') {
              benchmark.workgroupIndex++;



              if (benchmark.workgroupIndex < benchmark.workgroupSizes.length) {
                const nextSize = benchmark.workgroupSizes[benchmark.workgroupIndex];
                gaussian_renderer!.setPreprocessWorkgroupSize(nextSize);

                benchmark.warmupRemaining = 30;
                benchmark.samplesRemaining = 100;
                benchmark.gpuTimesMs = [];
                benchmark.readbackPending = false;
                benchmark.running = true;

                console.log(`Starting ${nextSize}-thread benchmark...`);


              } else {

                benchmark.running = false;
                console.log('All three workgroup-size benchmarks completed.');
              }

              
            } else if (benchmark.mode === 'culling') {
              
              benchmark.cullingIndex++;




              if (benchmark.cullingIndex < benchmark.cullingModes.length) {
                const nextCulling = benchmark.cullingModes[benchmark.cullingIndex];
                gaussian_renderer!.setFrustumCulling(nextCulling);



                benchmark.warmupRemaining = 30;
                benchmark.samplesRemaining = 100;
                benchmark.gpuTimesMs = [];
                benchmark.readbackPending = false;
                benchmark.running = true;


                console.log(`Starting frustum-culling ${nextCulling ? 'ON' : 'OFF'} benchmark...`);

              } else {
                benchmark.running = false;
                console.log('Both frustum-culling benchmarks completed.');
              }
            } else if (benchmark.mode === 'comparison') {
              benchmark.comparisonIndex++;

              if (benchmark.comparisonIndex < benchmark.comparisonRenderers.length) {
                const nextRenderer =
                  benchmark.comparisonRenderers[benchmark.comparisonIndex];

                params.renderer = nextRenderer;
                renderer = nextRenderer === 'pointcloud'
                  ? pointcloud_renderer
                  : gaussian_renderer;

                pane.refresh();

                benchmark.warmupRemaining = 30;
                benchmark.samplesRemaining = 100;
                benchmark.gpuTimesMs = [];
                benchmark.readbackPending = false;

                console.log(`Starting ${nextRenderer} GPU benchmark...`);
              } else {
                benchmark.running = false;
                console.log('Both renderer comparison benchmarks completed.');
              }
            }
          }

        }).catch((error) => {
          benchmark.readbackPending = false;
          benchmark.running = false;
          console.error('GPU benchmark failed:', error);
        });
      }
    }

    requestAnimationFrame(frame);
  }


  requestAnimationFrame(frame);
}
