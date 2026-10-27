// Requires puppeteer-core (npm i --no-save puppeteer-core) and a local Chrome (override with CHROME=path).
// Headless test harness: loads a scene into the viewer and runs a list of actions.
// usage: node run.mjs <url> <ply> <cameras.json> <actions.json | inline json>
// env: VIEW=WxH (default 1280x800), RECORD_FILE=path (for "record")
// actions:
//   {"select": [index, value]}     tweakpane dropdown by index (matched on option text)
//   {"checkbox": [index, bool]}    tweakpane checkbox by index
//   {"key": "1"}                   key press (camera presets)
//   {"wait": ms}
//   {"ui": false|true}             hide / show the GUI and log overlay
//   {"shot": "file.png"}           screenshot of the canvas
//   {"record": "label", "samples": n, "interval": ms}   sample the gpu_stats monitor n times, mean + sd
//   {"orbit": [dx, dy, ms], "frames": "dir"?}  drag the camera; optionally capture frames while doing so
//   {"capture": ms, "frames": "dir"}            capture frames without input
//   {"stats": 1}                   print the gpu_stats monitor
import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer-core';

const [url, ply, cams, actionsArg] = process.argv.slice(2);
const actions = JSON.parse(fs.existsSync(actionsArg) ? fs.readFileSync(actionsArg, 'utf8') : actionsArg);
const [W, H] = (process.env.VIEW ?? '1280x800').split('x').map(Number);

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--disable-gpu-sandbox', '--force_high_performance_gpu'],
  defaultViewport: { width: W, height: H },
  protocolTimeout: 600000,
});
const page = await browser.newPage();
page.on('console', m => {
  const t = m.text();
  if (!t.startsWith('[vite]') && !t.includes('powerPreference')) console.log(`[console.${m.type()}] ${t}`);
});
page.on('pageerror', e => console.log(`[pageerror] ${e.message}`));

await page.evaluateOnNewDocument(() => {
  const orig = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...args) {
    const d = await orig.apply(this, args);
    d.addEventListener('uncapturederror', e => console.error('GPU uncaptured error: ' + e.error.message));
    const origSM = d.createShaderModule.bind(d);
    d.createShaderModule = (desc) => {
      const m = origSM(desc);
      m.getCompilationInfo().then(info => {
        for (const msg of info.messages) {
          if (msg.type !== 'info') console.error(`WGSL ${msg.type} (${desc.label}) ${msg.lineNum}:${msg.linePos} ${msg.message}`);
        }
      });
      return m;
    };
    return d;
  };
});

await page.goto(url, { waitUntil: 'networkidle0' });
console.log('adapter: ' + await page.evaluate(async () => {
  const a = await navigator.gpu?.requestAdapter();
  return a ? [a.info.vendor, a.info.architecture].join(' / ') : 'none';
}));

const wait = ms => new Promise(r => setTimeout(r, ms));
let loaded = false;
page.on('console', m => { if (m.text().includes('return result')) loaded = true; });
await page.waitForSelector('input[type=file]', { timeout: 20000 });
const inputs = await page.$$('input[type=file]');
await inputs[0].uploadFile(ply);
for (let i = 0; i < 1200 && !loaded; i++) await wait(100);
console.log('ply loaded: ' + loaded);
await inputs[1].uploadFile(cams);
await wait(1500);

const shot = async (file) => {
  const canvas = await page.$('#webgpu-canvas');
  await canvas.screenshot({ path: file });
  console.log('saved ' + file);
};

const read_stats = () => page.evaluate(() => [...document.querySelectorAll('textarea')].map(t => t.value).join('\n'));
const parse_stats = (text) => {
  const row = {};
  for (const line of text.split('\n')) {
    const m = line.trim().match(/^([a-z ]+?)\s+([\d.]+)/);
    if (m) row[m[1].trim().replace(/ /g, '_')] = parseFloat(m[2]);
  }
  return row;
};

// frames are captured with the CDP screencast (much faster than screenshots)
// continuous recording ({"rec_start": dir} ... {"rec_stop": true}); writes frames + an ffmpeg concat list
// with the real frame durations from the screencast timestamps
let recorder;
async function rec_start(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const client = await page.createCDPSession();
  const frames = [];
  client.on('Page.screencastFrame', async (f) => {
    const file = `f${String(frames.length).padStart(5, '0')}.jpg`;
    fs.writeFileSync(path.join(dir, file), Buffer.from(f.data, 'base64'));
    frames.push({ file, t: f.metadata.timestamp });
    await client.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {});
  });
  await client.send('Page.startScreencast', { format: 'jpeg', quality: 95, everyNthFrame: 1 });
  recorder = { client, frames, dir };
}
async function rec_stop() {
  const { client, frames, dir } = recorder;
  await client.send('Page.stopScreencast');
  await client.detach();
  let list = '';
  frames.forEach((f, i) => {
    const next = frames[i + 1]?.t ?? f.t + 1 / 30;
    list += `file '${f.file}'\nduration ${Math.max(0.001, next - f.t).toFixed(4)}\n`;
  });
  list += `file '${frames[frames.length - 1].file}'\n`;
  fs.writeFileSync(path.join(dir, 'frames.txt'), list);
  console.log(`recorded ${frames.length} frames, ${(frames[frames.length - 1].t - frames[0].t).toFixed(1)} s`);
  recorder = undefined;
}

async function with_frames(dir, fn) {
  if (!dir) return fn();
  fs.mkdirSync(dir, { recursive: true });
  let n = fs.readdirSync(dir).length;
  const client = await page.createCDPSession();
  client.on('Page.screencastFrame', async (f) => {
    fs.writeFileSync(path.join(dir, `f${String(n++).padStart(5, '0')}.jpg`), Buffer.from(f.data, 'base64'));
    await client.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {});
  });
  await client.send('Page.startScreencast', { format: 'jpeg', quality: 92, everyNthFrame: 1 });
  await fn();
  await client.send('Page.stopScreencast');
  await client.detach();
  console.log(`frames in ${dir}: ${n}`);
}

for (const a of actions) {
  if (a.select) {
    const sel = (await page.$$('select'))[a.select[0]];
    // match on the option text, tweakpane uses indices as option values
    const value = await sel.evaluate((s, text) => [...s.options].find(o => o.text === text)?.value, String(a.select[1]));
    if (value === undefined) throw new Error(`no option ${a.select[1]}`);
    await sel.select(value);
  } else if (a.checkbox) {
    const cb = (await page.$$('input[type=checkbox]'))[a.checkbox[0]];
    const checked = await cb.evaluate(e => e.checked);
    if (checked !== a.checkbox[1]) await cb.evaluate(e => e.click());
  } else if (a.number) {
    // tweakpane slider text field, by index among the number text inputs
    const field = (await page.$$('.tp-sldtxtv .tp-txtv_i'))[a.number[0]];
    // set programmatically: typing would also trigger the 0-9 camera preset shortcuts
    await field.evaluate((e, v) => {
      e.value = String(v);
      e.dispatchEvent(new Event('change', { bubbles: true }));
    }, a.number[1]);
  } else if ('wheel' in a) {
    // the camera dollies 0.001 units per wheel delta along the view direction
    await page.mouse.move(W / 2, H / 2);
    await page.mouse.wheel({ deltaY: a.wheel });
  } else if (a.key) {
    await page.keyboard.press(a.key);
  } else if (a.wait) {
    await wait(a.wait);
  } else if ('ui' in a) {
    await page.evaluate((show) => {
      for (const el of document.querySelectorAll('.console, .tp-dfwv')) el.style.visibility = show ? '' : 'hidden';
    }, a.ui);
  } else if (a.shot) {
    await shot(a.shot);
  } else if (a.record) {
    const n = a.samples ?? 5;
    const rows = [];
    for (let i = 0; i < n; i++) {
      await wait(a.interval ?? 400);
      rows.push(parse_stats(await read_stats()));
    }
    const out = { label: a.record };
    for (const k of Object.keys(rows[0])) {
      const v = rows.map(r => r[k]);
      const mean = v.reduce((x, y) => x + y, 0) / n;
      out[k] = +mean.toFixed(4);
      out[k + '_sd'] = +Math.sqrt(v.reduce((x, y) => x + (y - mean) ** 2, 0) / n).toFixed(4);
    }
    fs.appendFileSync(process.env.RECORD_FILE, JSON.stringify(out) + '\n');
    console.log(JSON.stringify(out));
  } else if (a.orbit) {
    const [dx, dy, ms] = a.orbit;
    await with_frames(a.frames, async () => {
      const cx = W / 2, cy = H / 2, steps = Math.max(1, Math.round(ms / 16));
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      for (let i = 1; i <= steps; i++) {
        await page.mouse.move(cx + dx * i / steps, cy + dy * i / steps);
        await wait(16);
      }
      await page.mouse.up();
    });
  } else if (a.hide_log) {
    await page.evaluate(() => { document.querySelector('.console').style.visibility = 'hidden'; });
  } else if (a.rec_start) {
    await rec_start(a.rec_start);
  } else if (a.rec_stop) {
    await rec_stop();
  } else if (a.slide) {
    // animate a slider: {"slide": [index, from, to, ms]}
    const [i, from, to, ms] = a.slide;
    const field = (await page.$$('.tp-sldtxtv .tp-txtv_i'))[i];
    const steps = Math.max(1, Math.round(ms / 50));
    for (let s = 0; s <= steps; s++) {
      await field.evaluate((e, v) => {
        e.value = v.toFixed(2);
        e.dispatchEvent(new Event('change', { bubbles: true }));
      }, from + (to - from) * s / steps);
      await wait(50);
    }
  } else if (a.arc) {
    // orbit around a point `dist` units ahead: per step rotate (left drag) and pan (right drag)
    // the camera rotates 0.01 rad and pans 0.01 units per pixel
    const [deg, dist, ms, sign] = a.arc;
    const steps = Math.max(1, Math.round(ms / 33));
    const rot_px = (deg * Math.PI / 180) / 0.01 / steps;
    const pan_px = rot_px * dist;
    const cx = W / 2, cy = H / 2;
    for (let s = 0; s < steps; s++) {
      await page.mouse.move(cx, cy);
      await page.mouse.down({ button: 'left' });
      await page.mouse.move(cx + rot_px, cy);
      await page.mouse.up({ button: 'left' });
      await page.mouse.move(cx, cy);
      await page.mouse.down({ button: 'right' });
      await page.mouse.move(cx + (sign ?? 1) * pan_px, cy);
      await page.mouse.up({ button: 'right' });
      await wait(16);
    }
  } else if (a.capture) {
    await with_frames(a.frames, () => wait(a.capture));
  } else if (a.stats) {
    console.log('stats:\n' + await read_stats());
  }
}
await browser.close();
