const TOKEN = document.querySelector('meta[name="twin-session"]').content;
const PLAY = ["demo", "real", "leader", "udp", "replay"];
const JOB_ZH = { demo: "演示", real: "真机遥操作", leader: "只用主臂", udp: "远程接收", replay: "回放", setup: "安装环境",
  calibrate: "标定", doctor: "体检", tests: "测试", "import-isaac": "导入 Isaac 仓库", stream: "推流" };
const PHASE_ZH = { lobby: "等待开工（双捏夹爪或点“开工”）", sync: "对齐姿态", countdown: "倒计时", shift: "上班中",
  paused: "暂停（姿态没对上）", ready: "准备", results: "结算" };
const DIFF_ZH = { chill: "轻松", normal: "普通", hard: "困难" };
const VEH_ZH = { to_racks: "去装货", loading: "装货中", ready: "装好了", delivering: "送货中", unloading: "卸货中",
  parked: "待命", busy: "忙", returning: "返回中" };
const $ = (id) => document.getElementById(id);
let state = null, mode = "demo", logNext = 0, logText = "", lastJobStarted = null, cfgFilled = false;

async function api(path, body) {
  const opt = { headers: { "X-Twin-Token": TOKEN } };
  if (body !== undefined) { opt.method = "POST"; opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, opt);
  return r.json();
}
function flash(el, text, bad) { el.textContent = text; el.style.color = bad ? "var(--bad)" : ""; }

const observation = mountObservation($("vision"), {
  request: (path, options) => fetch("/api/observation" + path, {
    ...options, headers: {...options.headers, "X-Twin-Token": TOKEN}
  }),
  visible: () => $("vision-card").open,
  start: () => updateVision(true)
});
let visionCheck;
function updateVision(start = false) {
  if (visionCheck) return visionCheck.then(() => start && !visionOnline ? updateVision(true) : undefined);
  visionCheck = discoverVision(start).finally(() => {visionCheck = null;});
  return visionCheck;
}
let visionOnline = false;
async function discoverVision(start) {
  const root = $("vision").shadowRoot;
  const connect = root.getElementById("embeddedConnect");
  if (start) { connect.disabled = true; connect.textContent = "正在启用…"; }
  try {
    let result = await api("/api/vision", start ? {} : undefined);
    while (result.starting) {
      await new Promise(resolve => setTimeout(resolve, 850));
      result = await api("/api/vision");
    }
    visionOnline = !!result.url;
    $("arm-debug").disabled = !visionOnline;
    $("vision-summary").textContent = visionOnline ? "视觉工位已连接" : "视觉工位未启用";
    observation.connect(result.url);
    if (result.error) root.getElementById("status").textContent = result.error;
    if (start && visionOnline) root.getElementById("pairDialog").showModal();
  } catch {
    visionOnline = false; observation.connect(null);
    root.getElementById("status").textContent = "无法连接视觉工位";
  } finally { connect.disabled = false; }
}
$("arm-debug").onclick = () => observation.openArm();
function openObservations() {
  $("vision-card").open = true;
  $("vision-card").scrollIntoView({block:"start"});
}
$("nav-vision").onclick = (e) => { e.preventDefault(); openObservations(); };
$("nav-shift").onclick = () => { $("vision-card").open = false; };
if (location.hash === "#vision") openObservations();
updateVision();
setInterval(() => { if (!document.hidden && !visionCheck) updateVision(); }, 5000);

let warehouseJob = null, warehouseEpoch = 0, warehouseUrl = null;
function showWarehouse(job) {
  if (job === warehouseJob) return;
  warehouseJob = job;
  const epoch = ++warehouseEpoch;
  const img = $("warehouse-image");
  img.removeAttribute("src"); img.hidden = true;
  $("warehouse-wait").hidden = false;
  if (warehouseUrl) URL.revokeObjectURL(warehouseUrl);
  warehouseUrl = null;
  observation.warehouse(Boolean(job));
  if (!job) return;
  let tag = "";
  async function pollWarehouse() {
    if (epoch !== warehouseEpoch) return;
    try {
      if (document.hidden || !observation.wantsWarehouseFrame()) return;
      const response = await fetch("/api/warehouse.jpg", {headers:{"X-Twin-Token":TOKEN, "If-None-Match":tag}, cache:"no-store"});
      if (response.status !== 200) return;
      const blob = await response.blob();
      if (epoch !== warehouseEpoch || !observation.wantsWarehouseFrame()) return;
      const old = warehouseUrl;
      warehouseUrl = URL.createObjectURL(blob);
      img.src = warehouseUrl;
      img.hidden = false; $("warehouse-wait").hidden = true;
      tag = response.headers.get("ETag") || "";
      if (old) URL.revokeObjectURL(old);
    } catch { /* retry on the next poll */ }
    finally { if (epoch === warehouseEpoch) setTimeout(pollWarehouse, document.hidden ? 700 : 60); }
  }
  pollWarehouse();
}

// ------------------------------------------------------------------ mode tabs + options
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  mode = t.dataset.mode;
  document.querySelectorAll(".tab").forEach((x) => x.setAttribute("aria-pressed", String(x === t)));
  renderMode();
}));
function renderMode() {
  document.querySelectorAll(".only-real").forEach((e) => e.classList.toggle("hidden", mode !== "real"));
  document.querySelectorAll(".only-udp").forEach((e) => e.classList.toggle("hidden", mode !== "udp"));
  const guess = state && !state.ports.configured && (mode === "real" || mode === "leader");
  $("port-guess-warn").classList.toggle("hidden", !guess);
}
function playArgs() {
  const a = ["--difficulty", $("o-difficulty").value, "--shift", String(Number($("o-shift").value) || 150), "--hall", $("o-hall").value];
  if ($("o-viewer").value === "native") a.push("--native-viewer");
  if ($("o-player").value.trim()) a.push("--player", $("o-player").value.trim());
  if (!$("o-assist").checked) a.push("--no-assist");
  if (!$("o-hints").checked) a.push("--no-hints");
  a.push($("o-shadows").checked ? "--shadows" : "--no-shadows");
  if (mode === "real" && $("o-maxrel").value) a.push("--max-relative-target", String(Number($("o-maxrel").value)));
  if (mode === "udp") a.push("--udp-port", String(Number($("o-udpport").value) || 5566));
  return a;
}
async function runJob(cmd, args, noteEl) {
  const r = await api("/api/run", { cmd, args: args || [] });
  if (!r.ok) { flash(noteEl || $("start-note"), r.error, true); return false; }
  if (noteEl) flash(noteEl, "");
  if (PLAY.includes(cmd) && !args.includes("--native-viewer")) openObservations();
  logNext = 0; logText = ""; $("log").textContent = "";
  refresh();
  return true;
}
$("start-btn").addEventListener("click", () => runJob(mode, playArgs(), $("start-note")));
document.querySelectorAll("[data-job]").forEach((b) => b.addEventListener("click", () => runJob(b.dataset.job, [], $("env-note"))));
$("calib-follower").addEventListener("click", () => runJob("calibrate", ["follower"], $("env-note")));
$("calib-leader").addEventListener("click", () => runJob("calibrate", ["leader"], $("env-note")));
$("stream-btn").addEventListener("click", () => {
  const host = $("stream-host").value.trim();
  if (!host) return flash($("env-note"), "先填仿真机的 IP", true);
  runJob("stream", ["--twin-host", host], $("env-note"));
});

// ------------------------------------------------------------------ live game controls
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => api("/api/input", { line: b.dataset.cmd })));
$("c-stop").addEventListener("click", () => api("/api/stop", {}));
$("stop-btn").addEventListener("click", () => api("/api/stop", {}));
function sendStdin() { const v = $("stdin").value; $("stdin").value = ""; api("/api/input", { line: v }); }
$("enter-btn").addEventListener("click", sendStdin);
$("stdin").addEventListener("keydown", (e) => { if (e.key === "Enter") sendStdin(); });
$("quit-btn").addEventListener("click", async () => {
  if (!confirm("关闭控制台？正在运行的程序也会停止。")) return;
  await api("/api/shutdown", {});
  document.body.innerHTML = '<p style="padding:40px;font-size:18px">控制台已关闭。重新打开：双击 twin.bat / twin.command，或运行 ./twin.sh。</p>';
});

// ------------------------------------------------------------------ settings
const CFG_KEYS = ["leader_id", "follower_id", "arm", "player", "leader_port", "follower_port"];
$("cfg-save").addEventListener("click", async () => {
  const u = {};
  CFG_KEYS.forEach((k) => (u[k] = $("c-" + k).value.trim()));
  if (u.leader_port !== state.config.leader_port) u.leader_serial = "";
  if (u.follower_port !== state.config.follower_port) u.follower_serial = "";
  const r = await api("/api/config", { updates: u });
  flash($("cfg-saved"), r.ok ? "已保存" : r.error, !r.ok);
  cfgFilled = false; refresh();
});

// ------------------------------------------------------------------ port wizard (unplug / replug, detected automatically)
let wiz = null, wizTimer = null;
const usbSet = (ports) => new Set(ports.filter((p) => p.usb).map((p) => p.device));
const ROLE_ZH = { leader: "主臂", follower: "从臂" };
async function wizPoll() {
  const r = await api("/api/ports");
  if (!wiz) return;
  if (!r.ok) { $("wiz-step").textContent = r.error; return; }
  const now = usbSet(r.ports);
  wiz.ports = r.ports;
  $("wiz-ports").innerHTML = [...now].sort().map((d) => {
    const p = r.ports.find((x) => x.device === d);
    return `<li>${d}${p.serial ? "  SN " + p.serial : ""}${p.hint ? "  ⚠ " + p.hint : ""}</li>`;
  }).join("") || "<li>（没有 USB 串口）</li>";
  const role = wiz.role;
  if (wiz.stage === "start") {
    if (now.size === 0) return setWiz("先把两只臂的 USB 线插上并通电", "插好后这里会自动继续。");
    wiz.baseline = now; wiz.stage = "unplug";
  }
  if (wiz.stage === "unplug") {
    const gone = [...wiz.baseline].filter((d) => !now.has(d));
    const added = [...now].filter((d) => !wiz.baseline.has(d));
    if (added.length) wiz.baseline = now;
    if (gone.length === 1) { wiz.remaining = now; wiz.stage = "replug"; wiz.gone = gone[0]; }
    else if (gone.length > 1) { wiz.baseline = now; return setWiz("一次只拔一只臂", "检测到多个串口同时消失，请都插回去再试。"); }
    else return setWiz(`拔掉${ROLE_ZH[role]}的 USB 线`, "拔掉后会自动检测，不用点任何按钮。");
  }
  if (wiz.stage === "replug") {
    const back = [...now].filter((d) => !wiz.remaining.has(d));
    if (back.length !== 1) return setWiz(`${ROLE_ZH[role]}是 ${wiz.gone}，现在把它插回去`, "插回后自动继续（有的系统插回后串口名会变，以插回后的为准）。");
    wiz.found[role] = back[0];
    const others = [...now].filter((d) => d !== back[0] && !Object.values(wiz.found).includes(d));
    if (role === "leader" && others.length === 1) wiz.found.follower = others[0];
    else if (role === "leader") { wiz.role = "follower"; wiz.stage = "unplug"; wiz.baseline = now; return wizPoll(); }
    return wizSave();
  }
}
function setWiz(step, detail) { $("wiz-step").textContent = step; $("wiz-detail").textContent = detail || ""; }
async function wizSave() {
  const ports = wiz.ports.filter((p) => p.usb);
  const sn = (dev) => {
    const s = (ports.find((p) => p.device === dev) || {}).serial || "";
    return s && ports.filter((p) => p.serial === s).length === 1 ? s : "";
  };
  const f = wiz.found;
  const u = { leader_port: f.leader, follower_port: f.follower, leader_serial: sn(f.leader), follower_serial: sn(f.follower) };
  stopWiz();
  const r = await api("/api/config", { updates: u });
  const both = u.leader_serial && u.follower_serial;
  $("port-notes").innerHTML = r.ok
    ? `<div class="${both ? "note" : "warnbox"}">已识别：主臂 ${u.leader_port}，从臂 ${u.follower_port}。` +
      (both ? "已记住两块驱动板的序列号，换 USB 口也能自动找回。" : "有驱动板没有唯一序列号，只能按端口名记；换了 USB 口要重新识别。") + "</div>"
    : `<div class="errbox">${r.error}</div>`;
  cfgFilled = false; refresh();
}
function stopWiz() { wiz = null; clearInterval(wizTimer); $("wizard").classList.add("hidden"); }
$("wiz-btn").addEventListener("click", () => {
  wiz = { stage: "start", role: "leader", found: {}, ports: [] };
  $("wizard").classList.remove("hidden");
  setWiz("正在读取串口…");
  clearInterval(wizTimer); wizTimer = setInterval(wizPoll, 600); wizPoll();
});
$("wiz-cancel").addEventListener("click", stopWiz);

// ------------------------------------------------------------------ rendering
function badge(text, ok) { return `<span class="badge ${ok ? "ok" : "warn"}">${text}</span>`; }
function render() {
  const s = state;
  $("badges").innerHTML = [
    badge(s.os, true),
    badge(s.env_python ? "环境就绪" : "未安装环境", !!s.env_python),
    badge(s.isaac ? "Isaac 仓库已导入" : "内置大厅", s.isaac),
    badge(s.ports.configured ? "端口已识别" : "端口未识别", s.ports.configured),
  ].join("");
  $("cfg-path").textContent = s.config_path;
  $("a-leader").textContent = s.ports.leader;
  $("a-follower").textContent = s.ports.follower;
  const tag = (role) => s.config[role + "_serial"] ? "SN " + s.config[role + "_serial"] : (s.config[role + "_port"] ? "按端口名记住" : "猜测值");
  $("a-leader-sn").textContent = tag("leader");
  $("a-follower-sn").textContent = tag("follower");
  if (!wiz && s.port_notes.length) $("port-notes").innerHTML = s.port_notes.map((n) => `<div class="warnbox">${n}</div>`).join("");
  if (!cfgFilled) { CFG_KEYS.forEach((k) => ($("c-" + k).value = s.config[k] || "")); if (!$("o-player").value) $("o-player").value = s.config.player || ""; cfgFilled = true; }
  $("env-note").textContent = s.env_python ? "环境：" + s.env_python : "第一次使用：先点“安装 / 更新环境”（会下载 LeRobot、MuJoCo 和 Isaac 仓库，需要几分钟）。";
  const job = s.job, running = job && job.running;
  $("jobline").textContent = job ? `${JOB_ZH[job.name] || job.name} · ${running ? "运行中" : "已结束（退出码 " + job.rc + "）"}` : "空闲";
  $("start-btn").disabled = !!running || !s.env_python;
  document.querySelectorAll("[data-job]").forEach((b) => (b.disabled = !!running || (!s.env_python && b.dataset.job !== "setup")));
  ["calib-follower", "calib-leader", "stream-btn"].forEach((id) => ($(id).disabled = !!running || !s.env_python));
  $("stop-btn").disabled = !running;
  if (job && job.name === "calibrate" && running) $("stdin").placeholder = "标定：按输出里的提示摆好姿态，然后点“回车”";
  renderMode();
  showWarehouse(running && PLAY.includes(job.name) && (job.web_preview ?? ($("o-viewer").value === "web")) ? job.started : null);
  renderLive(running && PLAY.includes(job.name) ? s.status : null, running && PLAY.includes(job.name));
}
function renderLive(st, playing) {
  $("live-card").classList.toggle("hidden", !playing);
  if (!playing) return;
  if (!st) { $("s-phase").textContent = "正在加载场景…"; return; }
  $("live-sub").textContent = `${st.hall === "isaac" ? "Isaac 仓库" : "内置大厅"} · ${DIFF_ZH[st.difficulty] || st.difficulty} · ${st.source === "autopilot" ? "自动驾驶" : st.source}`;
  $("s-score").textContent = st.score;
  $("s-mult").textContent = st.multiplier > 1 ? "×" + st.multiplier : "";
  $("s-phase").textContent = PHASE_ZH[st.phase] || st.phase;
  $("s-phase").className = "phase " + st.phase;
  const left = Math.max(0, st.time_left);
  $("s-time").textContent = st.phase === "shift" || st.phase === "paused" ? `剩余 ${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, "0")}` : "";
  $("s-bar").style.width = (100 * (1 - left / Math.max(1, st.shift_s))).toFixed(1) + "%";
  $("s-sorted").textContent = st.sorted; $("s-errors").textContent = st.errors; $("s-missed").textContent = st.missed;
  $("s-acc").textContent = st.accuracy == null ? "—" : st.accuracy + "%";
  $("s-queue").textContent = st.queue; $("s-cage").textContent = `${st.cage}/${st.cage_capacity}`;
  $("s-agv").textContent = "· " + (VEH_ZH[st.agv] || st.agv);
  $("s-tug").textContent = "· " + (VEH_ZH[st.tug] || st.tug);
  $("c-assist").textContent = "磁吸辅助：" + (st.assist ? "开" : "关");
  $("c-hints").textContent = "提示：" + (st.hints ? "开" : "关");
  $("c-start").classList.toggle("hidden", st.mode !== "live");
  const r = st.results;
  $("s-results").classList.toggle("hidden", !r);
  if (r) $("s-results").innerHTML = `<b>结算：${r.score} 分 · 评级 ${r.rank}（${r.title_zh}）</b><br>准确率 ${r.accuracy}% · 最大连击 ${r.max_combo} · 漏拣 ${r.missed} · 摔地 ${r.spilled}${r.place ? " · 排行榜第 " + r.place : ""}`;
  $("s-events").innerHTML = st.events.slice().reverse().map((e) =>
    `<li class="${e.kind}"><span>${escapeHtml(e.text)}</span>${e.points ? `<span class="pts">${e.points > 0 ? "+" : ""}${e.points}</span>` : ""}</li>`).join("");
}
function escapeHtml(t) { return String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

// terminal output: strip colour codes, honour carriage returns (progress bars)
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;
function cleanLog(t) {
  return t.replace(ANSI, "").split("\n").map((l) => { const i = l.lastIndexOf("\r"); return i >= 0 ? l.slice(i + 1) : l; }).join("\n");
}
async function pollLog() {
  try {
    const r = await api("/api/log?since=" + logNext);
    if (r.reset) { logText = ""; }
    if (r.text) {
      logText = (logText + r.text).slice(-200000);
      const el = $("log"), atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      el.textContent = cleanLog(logText);
      if (atBottom) el.scrollTop = el.scrollHeight;
    }
    logNext = r.next;
  } catch (e) { /* server closed */ }
}
async function refresh() {
  try {
    state = await api("/api/state");
    if (state.job && state.job.started !== lastJobStarted) { lastJobStarted = state.job.started; }
    render();
  } catch (e) {
    $("jobline").textContent = "控制台后台已断开";
  }
}
renderMode();
refresh(); pollLog();
setInterval(refresh, 700);
setInterval(() => { if (!document.hidden && state?.job) pollLog(); }, 800);
