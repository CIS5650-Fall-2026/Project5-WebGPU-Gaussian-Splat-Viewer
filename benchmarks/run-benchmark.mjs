#!/usr/bin/env node

import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const project_root = fileURLToPath(new URL('../', import.meta.url));

function usage() {
  console.log(`Usage:
  node benchmarks/run-benchmark.mjs --ply <scene.ply> --cameras <cameras.json> [options]

Options:
  --camera <index>       Camera preset, default 0
  --warmup <seconds>     Warm-up duration, default 5
  --seconds <seconds>    Duration per run, default 10
  --runs <count>         Number of runs, default 3
  --width <pixels>       Canvas width, default 1280
  --height <pixels>      Canvas height, default 720
  --wg <size>            Preprocess workgroup size, default 256
  --precision <mode>     f32 or f16 preprocessing, default f32
  --sort <mode>          global or tile sorting, default global
  --output <file.json>   Output path
  --headed               Show the browser window
  --help                 Print this message`);
}

function parse_arguments(argv) {
  const options = {
    camera: 0,
    warmup: 5,
    seconds: 10,
    runs: 3,
    width: 1280,
    height: 720,
    wg: 256,
    precision: 'f32',
    sort: 'global',
    headed: false,
  };

  const value_options = new Set([
    'ply',
    'cameras',
    'camera',
    'warmup',
    'seconds',
    'runs',
    'width',
    'height',
    'wg',
    'precision',
    'sort',
    'output',
  ]);

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') {
      usage();
      process.exit(0);
    }
    if (argument === '--headed') {
      options.headed = true;
      continue;
    }
    if (!argument.startsWith('--')) {
      throw new Error(`Unexpected argument: ${argument}`);
    }

    const name = argument.slice(2);
    if (!value_options.has(name)) {
      throw new Error(`Unknown option: ${argument}`);
    }
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for ${argument}`);
    }
    options[name] = value;
    index += 1;
  }

  for (const name of ['camera', 'warmup', 'seconds', 'runs', 'width', 'height', 'wg']) {
    options[name] = Number(options[name]);
    if (!Number.isFinite(options[name])) {
      throw new Error(`--${name} must be a finite number`);
    }
  }

  options.camera = Math.min(Math.max(Math.round(options.camera), 0), 9);
  options.warmup = Math.max(options.warmup, 0);
  options.seconds = Math.max(options.seconds, 1);
  options.runs = Math.min(Math.max(Math.round(options.runs), 1), 20);
  options.width = Math.min(Math.max(Math.round(options.width), 64), 7680);
  options.height = Math.min(Math.max(Math.round(options.height), 64), 4320);
  options.wg = Math.min(Math.max(Math.round(options.wg), 32), 512);
  if (options.precision !== 'f32' && options.precision !== 'f16') {
    throw new Error('--precision must be f32 or f16');
  }
  if (options.sort !== 'global' && options.sort !== 'tile') {
    throw new Error('--sort must be global or tile');
  }

  if (!options.ply || !options.cameras) {
    throw new Error('Both --ply and --cameras are required');
  }
  options.ply = resolve(options.ply);
  options.cameras = resolve(options.cameras);
  if (!existsSync(options.ply) || !statSync(options.ply).isFile()) {
    throw new Error(`PLY file not found: ${options.ply}`);
  }
  if (!existsSync(options.cameras) || !statSync(options.cameras).isFile()) {
    throw new Error(`Camera file not found: ${options.cameras}`);
  }

  return options;
}

function find_chrome() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);

  const executable = candidates.find((candidate) => existsSync(candidate));
  if (!executable) {
    throw new Error('Chrome was not found; set CHROME_PATH to its executable');
  }
  return executable;
}

function median(values) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function percentile(values, fraction) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(sorted.length * fraction) - 1),
  )];
}

function summarize_run(run, intervals, duration_seconds) {
  const mean_frame_ms = intervals.length > 0
    ? intervals.reduce((sum, value) => sum + value, 0) / intervals.length
    : 0;
  return {
    run,
    duration_seconds,
    frames: intervals.length,
    mean_fps: mean_frame_ms > 0 ? 1000 / mean_frame_ms : 0,
    median_fps: median(intervals) > 0 ? 1000 / median(intervals) : 0,
    min_fps: intervals.length > 0 ? 1000 / Math.max(...intervals) : 0,
    median_frame_ms: median(intervals),
    p95_frame_ms: percentile(intervals, 0.95),
    p99_frame_ms: percentile(intervals, 0.99),
    frame_intervals_ms: intervals.map((value) => Number(value.toFixed(3))),
  };
}

function wait_for_console(page, pattern, timeout_ms) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      page.off('console', on_console);
      reject(new Error(`Timed out waiting for console message: ${pattern}`));
    }, timeout_ms);

    function on_console(message) {
      if (pattern.test(message.text())) {
        clearTimeout(timeout);
        page.off('console', on_console);
        resolve(message.text());
      }
    }

    page.on('console', on_console);
  });
}

async function sample_frame_times(page, warmup_seconds, sample_seconds, runs) {
  return page.evaluate(async ({ warmupMs, sampleMs, runCount }) => {
    const wait_for_duration = (duration_ms) => new Promise((resolve) => {
      const end = performance.now() + duration_ms;
      function tick(now) {
        if (now >= end) {
          resolve();
        } else {
          requestAnimationFrame(tick);
        }
      }
      requestAnimationFrame(tick);
    });

    await wait_for_duration(warmupMs * 1000);
    const results = [];
    for (let run = 1; run <= runCount; run += 1) {
      const intervals = [];
      let hidden_frames = 0;
      let started_at = 0;
      let last_frame = 0;
      let ended_at = 0;

      await new Promise((resolve) => {
        function tick(now) {
          if (started_at === 0) {
            started_at = now;
            last_frame = now;
            requestAnimationFrame(tick);
            return;
          }

          const interval = now - last_frame;
          if (interval > 0) {
            intervals.push(interval);
            if (document.visibilityState !== 'visible') {
              hidden_frames += 1;
            }
          }
          last_frame = now;

          if (now - started_at >= sampleMs * 1000) {
            ended_at = now;
            resolve();
          } else {
            requestAnimationFrame(tick);
          }
        }
        requestAnimationFrame(tick);
      });

      results.push({
        run,
        duration_seconds: (ended_at - started_at) / 1000,
        hidden_frames,
        intervals,
      });
    }
    return results;
  }, {
    warmupMs: warmup_seconds,
    sampleMs: sample_seconds,
    runCount: runs,
  });
}

async function main() {
  const options = parse_arguments(process.argv.slice(2));
  const started_at = new Date().toISOString();
  const timestamp = started_at.replace(/[:.]/g, '-');
  const scene_name = options.ply.split(/[\\/]/).pop().replace(/\.ply$/i, '');
  const default_output = join(
    project_root,
    'benchmarks',
    'results',
    `${scene_name}-${timestamp}.json`,
  );
  const output_file = isAbsolute(options.output ?? '')
    ? options.output
    : resolve(options.output ?? default_output);

  const server = await createServer({
    root: project_root,
    logLevel: 'error',
    server: {
      host: '127.0.0.1',
      port: 0,
      strictPort: false,
    },
  });
  await server.listen();
  const address = server.httpServer?.address();
  if (!address || typeof address === 'string') {
    throw new Error('Unable to determine the Vite server port');
  }

  const console_errors = [];
  const http_errors = [];
  const page_errors = [];
  let point_count = null;
  let camera_count = null;
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: find_chrome(),
      headless: !options.headed,
      args: [
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
      ],
    });
    const context = await browser.newContext({
      viewport: { width: options.width, height: options.height },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    page.on('console', (message) => {
      const text = message.text();
      const point_match = text.match(/^num points: (\d+)$/);
      if (point_match) {
        point_count = Number(point_match[1]);
      }
      const camera_match = text.match(/^loaded cameras count: (\d+)$/);
      if (camera_match) {
        camera_count = Number(camera_match[1]);
      }
      if (message.type() === 'error') {
        console_errors.push({
          text,
          location: message.location(),
        });
      }
    });
    page.on('pageerror', (error) => {
      page_errors.push(String(error));
    });
    page.on('response', (response) => {
      if (response.status() >= 400) {
        http_errors.push({
          status: response.status(),
          url: response.url(),
        });
      }
    });

    const url = `http://127.0.0.1:${address.port}/?wg=${options.wg}&precision=${options.precision}&sort=${options.sort}`;
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelectorAll('input[type="file"]').length === 2, {
      timeout: 15000,
    });
    if (!await page.evaluate(() => Boolean(navigator.gpu))) {
      throw new Error('WebGPU is not available in the benchmark browser');
    }

    const ply_loaded = wait_for_console(page, /return result!/, 240000);
    const ply_started = performance.now();
    await page.locator('input[type="file"]').nth(0).setInputFiles(options.ply);
    await ply_loaded;
    const ply_load_seconds = (performance.now() - ply_started) / 1000;

    const cameras_loaded = wait_for_console(page, /loaded cameras count:/, 30000);
    await page.locator('input[type="file"]').nth(1).setInputFiles(options.cameras);
    await cameras_loaded;

    const renderer_select = page.locator('select');
    if (await renderer_select.count() !== 1) {
      throw new Error('Could not locate the renderer selector');
    }
    await renderer_select.selectOption('gaussian');
    await page.waitForFunction(
      ({ width, height }) => {
        const canvas = document.querySelector('canvas');
        return canvas?.width === width && canvas?.height === height;
      },
      { width: options.width, height: options.height },
      { timeout: 15000 },
    );

    if (options.camera !== 0) {
      const camera_loaded = wait_for_console(page, /camera intrinsics:/, 10000);
      await page.keyboard.press(String(options.camera));
      await camera_loaded;
    }

    await page.waitForTimeout(500);
    const raw_runs = await sample_frame_times(
      page,
      options.warmup,
      options.seconds,
      options.runs,
    );
    const runs = raw_runs.map((run) => ({
      ...summarize_run(run.run, run.intervals, run.duration_seconds),
      hidden_frames: run.hidden_frames,
    }));
    const fatal_console_errors = console_errors.filter((entry) =>
      !(entry.location.url.endsWith('/favicon.ico') && entry.text.includes('404')),
    );
    const fatal_http_errors = http_errors.filter((entry) =>
      !entry.url.endsWith('/favicon.ico'),
    );
    const valid = fatal_console_errors.length === 0 &&
      fatal_http_errors.length === 0 &&
      page_errors.length === 0;

    const run_mean_fps = runs.map((run) => run.mean_fps);
    const result = {
      schema_version: 1,
      started_at,
      ended_at: new Date().toISOString(),
      scene: {
        ply_file: options.ply,
        camera_file: options.cameras,
        ply_bytes: statSync(options.ply).size,
        ply_load_seconds,
        point_count,
        camera_count,
      },
      config: {
        camera: options.camera,
        warmup_seconds: options.warmup,
        sample_seconds: options.seconds,
        runs: options.runs,
        canvas_width: options.width,
        canvas_height: options.height,
        preprocess_workgroup_size: options.wg,
        precision: options.precision,
        sort: options.sort,
        renderer: 'gaussian',
      },
      environment: {
        browser: await browser.version(),
        user_agent: await page.evaluate(() => navigator.userAgent),
        device_scale_factor: await page.evaluate(() => window.devicePixelRatio),
        webgpu_adapter: await page.evaluate(async () => {
          const adapter = await navigator.gpu?.requestAdapter({
            powerPreference: 'high-performance',
          });
          return adapter ? {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            device: adapter.info.device,
            description: adapter.info.description,
            shader_f16_supported: adapter.features.has('shader-f16'),
          } : null;
        }),
      },
      summary: {
        valid,
        mean_fps: run_mean_fps.reduce((sum, value) => sum + value, 0) / runs.length,
        median_run_fps: median(run_mean_fps),
      },
      runs,
      diagnostics: {
        console_errors,
        http_errors,
        page_errors,
        fatal_console_errors,
        fatal_http_errors,
      },
    };

    mkdirSync(dirname(output_file), { recursive: true });
    writeFileSync(output_file, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
    console.log(JSON.stringify({
      output: output_file,
      scene: scene_name,
      valid,
      mean_fps: Number(result.summary.mean_fps.toFixed(2)),
      runs: runs.map((run) => Number(run.mean_fps.toFixed(2))),
    }, null, 2));
  } finally {
    await browser?.close();
    await server.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
