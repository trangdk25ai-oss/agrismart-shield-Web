/* AgriSmart Shield – giao diện web (không cần thư viện ngoài, chạy được khi không có Internet). */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = v => (Number(v) || 0).toFixed(2).replace('.', ',') + '%';
  const vnDate = iso => { const [y, m, d] = String(iso).split('-'); return `${d}/${m}/${y}`; };
  const todayIso = () => { const d = new Date(); d.setMinutes(d.getMinutes() - d.getTimezoneOffset()); return d.toISOString().slice(0, 10); };
  const store = {
    get(k, def) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : def; } catch (e) { return def; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* trình duyệt chặn bộ nhớ */ } }
  };
  const S = { meta: null, file: null, previewUrl: null, sampleId: null, res: null, leaf: null, rec: null, samples: null, sfilter: 'all' };

  async function api(path, opts) {
    const r = await fetch(path, opts);
    const isJson = (r.headers.get('content-type') || '').includes('application/json');
    const body = isJson ? await r.json() : await r.text();
    if (!r.ok) {
      let msg = body && body.detail ? body.detail : ('Lỗi máy chủ (' + r.status + ')');
      if (Array.isArray(msg)) msg = 'Dữ liệu gửi lên không hợp lệ';
      throw new Error(msg);
    }
    return body;
  }
  function busy(on) { $('busy').hidden = !on; }

  // ------------------------------------------------------------------ khởi động
  async function init() {
    try {
      S.meta = await api('/api/meta');
    } catch (e) {
      $('mode-badge').textContent = 'Không kết nối được máy chủ';
      return;
    }
    const m = S.meta;
    const badge = $('mode-badge');
    if (m.mode === 'model') {
      badge.textContent = 'Mô hình: ' + m.model.file;
      badge.className = 'mode ok';
    } else {
      badge.textContent = 'Chế độ thử';
      badge.className = 'mode test';
      $('banner-mode').hidden = false;
      $('banner-mode').innerHTML = '<b>Chế độ thử – chưa có mô hình AI.</b> Vùng “vết bệnh” chỉ là vùng đổi màu trên lá, ' +
        'tên bệnh do bạn chọn. Dùng để chạy thử giao diện, <b>không dùng để chẩn đoán</b>. Đặt mô hình vào <code>web/models/</code> rồi khởi động lại.' +
        (m.model_error ? '<br><b>Lỗi nạp mô hình:</b> ' + esc(m.model_error) : '');
    }
    $('crop').innerHTML = m.crops.map(c => `<option value="${esc(c.code)}">${esc(c.name)}</option>`).join('');
    if (m.mode === 'color_rule') {
      $('lesion-wrap').hidden = false;
      $('lesion').innerHTML = m.crops.map(c => `<optgroup label="${esc(c.name)}">` +
        c.classes.map(k => `<option value="${esc(k.class)}">${esc(k.disease_name)}</option>`).join('') + '</optgroup>').join('');
      syncCropFromLesion();
    }
    $('tanks').innerHTML = m.tanks.map((t, i) =>
      `<label><input type="radio" name="tank" value="${esc(t.code)}"${i === 0 ? ' checked' : ''}> ${esc(t.label)}</label>`).join('');
    $('start-date').value = todayIso();
    const lan = (m.lan_urls || []).map(u => `<code>${esc(u)}</code>`).join(' · ');
    $('foot').innerHTML =
      `<p>Mô hình: ${m.mode === 'model' ? esc(m.model.file) + ' (' + esc(m.model.backend) + (m.model.size_mb ? ', ' + String(m.model.size_mb).replace('.', ',') + ' MB' : '') + ')' : 'chưa có (chế độ thử)'}` +
      ` · Tri thức IPM ${esc(m.kb.version)}${m.kb.review_status !== 'APPROVED' ? ' (DỰ THẢO)' : ''}</p>` +
      (lan ? `<p>Mở trên điện thoại cùng mạng Wi-Fi: ${lan}</p>` : '') +
      '<p>Kết quả mang tính hỗ trợ; đối chiếu nhãn thuốc và hỏi cán bộ bảo vệ thực vật trước khi phun.</p>';
    renderHistory();
  }
  function syncCropFromLesion() {
    const cls = $('lesion').value || '';
    const crop = cls.split('__')[0];
    if (crop && [...$('crop').options].some(o => o.value === crop)) $('crop').value = crop;
  }

  // ------------------------------------------------------------------ thẻ chức năng
  const tabs = ['diagnose', 'samples', 'history'];
  function showTab(name) {
    tabs.forEach(t => {
      $('tab-' + t).setAttribute('aria-selected', String(t === name));
      $('view-' + t).hidden = t !== name;
    });
    if (name === 'samples' && !S.samples) loadSamples();
    if (name === 'history') renderHistory();
  }
  tabs.forEach(t => $('tab-' + t).addEventListener('click', () => showTab(t)));

  // ------------------------------------------------------------------ chọn ảnh
  function onFile(ev) {
    const f = ev.target.files && ev.target.files[0];
    ev.target.value = '';
    if (!f) return;
    if (S.meta && f.size > S.meta.max_upload_mb * 1024 * 1024) {
      alert('Ảnh lớn hơn ' + S.meta.max_upload_mb + ' MB');
      return;
    }
    S.file = f;
    S.sampleId = null;
    if (S.previewUrl) URL.revokeObjectURL(S.previewUrl);
    S.previewUrl = URL.createObjectURL(f);
    $('preview-img').src = S.previewUrl;
    $('preview').hidden = false;
    $('step-result').hidden = true;
    $('step-plan').hidden = true;
    if (!$('blurbox').hidden) diagnose(false); // đang ở bước "Chụp lại" -> phân tích luôn ảnh mới
  }
  $('file-camera').addEventListener('change', onFile);
  $('file-pick').addEventListener('change', onFile);
  $('btn-clear').addEventListener('click', () => { S.file = null; $('preview').hidden = true; });
  $('btn-analyze').addEventListener('click', () => diagnose(false));
  $('btn-force').addEventListener('click', () => diagnose(true));
  $('lesion').addEventListener('change', syncCropFromLesion);

  // ------------------------------------------------------------------ chẩn đoán
  async function diagnose(force) {
    if (!S.file && !S.sampleId) return;
    const fd = new FormData();
    if (S.sampleId) fd.append('sample_id', S.sampleId);
    else fd.append('image', S.file, S.file.name || 'anh.jpg');
    fd.append('crop_code', $('crop').value);
    fd.append('tank_code', currentTank());
    fd.append('start_date', $('start-date').value || todayIso());
    fd.append('force', force ? 'true' : 'false');
    if (S.meta.mode === 'color_rule' && !S.sampleId) fd.append('lesion_class', $('lesion').value);
    busy(true);
    const t0 = performance.now();
    try {
      const res = await api('/api/diagnose', { method: 'POST', body: fd });
      res.timings_ms.round_trip = Math.round(performance.now() - t0);
      renderResult(res);
    } catch (e) {
      $('step-result').hidden = false;
      $('result-body').hidden = true;
      $('blurbox').hidden = true;
      $('step-plan').hidden = true;
      $('status').className = 'status err';
      $('status').textContent = e.message;
    } finally {
      busy(false);
    }
  }
  const currentTank = () => (document.querySelector('input[name=tank]:checked') || {}).value || 'TANK_16L';

  function renderResult(res) {
    S.res = res;
    $('step-result').hidden = false;
    $('status').className = 'status';
    $('status').textContent = '';
    if (res.status === 'BLURRY') {
      $('blurbox').hidden = false;
      $('result-body').hidden = true;
      $('step-plan').hidden = true;
      $('blur-text').innerHTML = esc(res.message) + ` <span class="muted">(điểm nét ${String(res.blur.score).replace('.', ',')} &lt; ngưỡng ${res.blur.threshold})</span>`;
      $('step-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    $('blurbox').hidden = true;
    $('result-body').hidden = false;
    $('res-img').src = 'data:image/jpeg;base64,' + res.image_jpeg_b64;
    $('res-overlay').src = res.overlay_png_b64 ? 'data:image/png;base64,' + res.overlay_png_b64 : '';
    $('res-overlay').hidden = !res.overlay_png_b64 || !$('ov-toggle').checked;
    $('legend').innerHTML = (res.legend || []).map(l => `<span><i style="background:${esc(l.color)}"></i>${esc(l.label)}</span>`).join('');
    if (res.message) $('status').textContent = res.message;
    if (res.blur && res.blur.is_blurry) $('status').textContent = (res.message ? res.message + ' ' : '') + 'Ảnh có thể bị mờ – kết quả kém tin cậy hơn.';
    const primary = (res.leaves || []).find(l => l.primary);
    S.leaf = primary ? primary.index : null;
    renderLeafBoxes();
    $('leaf-hint').hidden = (res.leaves || []).length < 2;
    S.rec = res.recommendation || null;
    renderSeverity();
    renderCompare();
    renderTech();
    renderPlan();
    if (S.rec) saveHistory();
    $('step-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  $('ov-toggle').addEventListener('change', () => { $('res-overlay').hidden = !$('ov-toggle').checked || !(S.res && S.res.overlay_png_b64); });

  function renderLeafBoxes() {
    const leaves = (S.res && S.res.leaves) || [];
    $('leaf-boxes').innerHTML = leaves.length < 2 ? '' : leaves.map((l, k) => {
      const [x1, y1, x2, y2] = l.box;
      return `<button type="button" class="leafbox${l.index === S.leaf ? ' sel' : ''}" data-i="${l.index}" ` +
        `style="left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%" ` +
        `aria-label="Lá ${k + 1}: ${pct(l.severity_pct)}"><span>Lá ${k + 1} · ${pct(l.severity_pct)}</span></button>`;
    }).join('');
    $('leaf-boxes').querySelectorAll('.leafbox').forEach(b => b.addEventListener('click', () => selectLeaf(Number(b.dataset.i))));
  }

  async function selectLeaf(index) {
    S.leaf = index;
    renderLeafBoxes();
    await refreshRecommendation();
  }

  async function refreshRecommendation() {
    const leaf = ((S.res && S.res.leaves) || []).find(l => l.index === S.leaf);
    if (!leaf) return;
    try {
      S.rec = await api('/api/recommend', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop_code: S.res.crop_code, model_class: leaf.dominant_class, severity_pct: leaf.severity_pct,
          has_lesion: leaf.lesion_px > 0, tank_code: currentTank(), start_date: $('start-date').value || todayIso(),
          detection_conf: leaf.dominant_conf
        })
      });
      renderSeverity();
      renderPlan();
    } catch (e) {
      $('status').className = 'status err';
      $('status').textContent = e.message;
    }
  }
  $('tanks').addEventListener('change', refreshRecommendation);
  $('start-date').addEventListener('change', refreshRecommendation);

  function renderSeverity() {
    const r = S.rec;
    const leaf = ((S.res && S.res.leaves) || []).find(l => l.index === S.leaf);
    if (!r) {
      $('sev-pct').textContent = '–';
      $('sev-level').textContent = '';
      $('sev-level').style.background = 'transparent';
      $('sev-name').textContent = 'Không đủ cơ sở để kết luận';
      $('sev-sub').textContent = '';
      return;
    }
    $('sev-pct').textContent = pct(r.severity_pct);
    $('sev-pct').style.color = r.level_color || '';
    $('sev-level').textContent = r.level_label || '';
    $('sev-level').hidden = !r.level_label;
    $('sev-level').style.background = r.level_color || '#757575';
    const found = (leaf && leaf.dominant_label) || r.model_class || '';
    $('sev-name').textContent = r.status === 'HEALTHY' ? 'Không phát hiện vết bệnh'
      : r.status === 'CROP_MISMATCH' ? 'Phát hiện ' + found + ' – khác cây trồng đã chọn'
      : r.status === 'NO_RULE' ? 'Phát hiện ' + found + ' – chưa có phác đồ'
      : (r.disease_name || found);
    const sub = [];
    if (r.pathogen) sub.push(r.pathogen);
    if (leaf && leaf.dominant_conf != null && S.meta.mode === 'model') sub.push('độ tin cậy ' + Math.round(leaf.dominant_conf * 100) + '%');
    if (leaf) sub.push(leaf.n_lesions + ' vết bệnh trên lá');
    $('sev-sub').textContent = sub.join(' · ');
  }

  function sysLabel() {
    const leaf = ((S.res && S.res.leaves) || []).find(l => l.index === S.leaf);
    return (leaf && leaf.dominant_label) || '–';
  }
  function renderCompare() {
    const g = S.res.ground_truth;
    $('compare').hidden = !g;
    if (!g) return;
    const r = S.rec || {};
    const mark = ok => ok == null ? '' : (ok ? '<span class="ok">✓</span>' : '<span class="no">✗</span>');
    $('compare').innerHTML = '<table><thead><tr><th></th><th>Đáp án (từ nhãn)</th><th>Hệ thống</th><th></th></tr></thead><tbody>' +
      `<tr><td>Bệnh</td><td>${esc(g.gt_disease_name || 'Lá khỏe')}</td><td>${esc(r.disease_name || (r.status === 'HEALTHY' ? 'Lá khỏe' : sysLabel()))}</td><td>${mark(g.gt_class ? g.same_class : r.status === 'HEALTHY')}</td></tr>` +
      `<tr><td>Tỷ lệ tổn thương</td><td>${pct(g.gt_severity_pct)}</td><td>${r.severity_pct != null ? pct(r.severity_pct) : '–'}</td><td>${g.abs_error_pct != null ? 'lệch ' + String(g.abs_error_pct).replace('.', ',') : ''}</td></tr>` +
      `<tr><td>Mức độ</td><td>${esc(g.gt_level_label || '')}</td><td>${esc(r.level_label || '–')}</td><td>${mark(g.same_level)}</td></tr>` +
      `</tbody></table><p class="muted" style="margin:6px 9px">Nguồn ảnh: ${g.source_id ? esc(g.source_id) : 'chưa ghi nguồn'}${g.license ? ' · ' + esc(g.license) : ''}</p>`;
  }

  function renderTech() {
    const r = S.res;
    const t = r.timings_ms || {};
    const ms = v => v == null ? '–' : Math.round(v) + ' ms';
    $('tech').innerHTML = '<dl class="kv">' +
      `<dt>Trạng thái</dt><dd><code>${esc(r.status)}</code></dd>` +
      `<dt>Kích thước xử lý</dt><dd>${r.image_size[0]} × ${r.image_size[1]} px</dd>` +
      `<dt>Điểm nét (Blur)</dt><dd>${String(r.blur.score).replace('.', ',')} (ngưỡng ${r.blur.threshold})</dd>` +
      `<dt>Đối tượng</dt><dd>${r.detections.length} (lá: ${r.leaves.length})</dd>` +
      `<dt>Suy luận</dt><dd>${ms(t.inference)}</dd>` +
      `<dt>Tiền/hậu xử lý</dt><dd>${ms(t.preprocess)} / ${ms(t.postprocess)}</dd>` +
      `<dt>Máy chủ (tổng)</dt><dd>${ms(t.server_total)}</dd>` +
      `<dt>Cả vòng (gồm mạng)</dt><dd>${ms(t.round_trip)}</dd>` +
      '</dl>';
  }

  function renderPlan() {
    const r = S.rec;
    $('step-plan').hidden = !r;
    if (!r) return;
    let h = '';
    if (r.warnings && r.warnings.length) h += `<div class="box warn"><h3>Cảnh báo</h3><ul>${r.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>`;
    if (r.message) h += `<div class="box danger">${esc(r.message)}</div>`;
    if (r.escalation) h += `<div class="box danger">${esc(r.escalation)}</div>`;
    if (r.actions && r.actions.length) h += `<div class="box"><h3>Việc cần làm ngay</h3><ol>${r.actions.map(a => `<li>${esc(a)}</li>`).join('')}</ol></div>`;
    if (r.dosage && r.dosage.length) h += `<div class="box"><h3>Công thức pha</h3><ul>${r.dosage.map(d => `<li>${esc(d.text)}${d.verified ? '' : ' <b>(liều mẫu)</b>'}</li>`).join('')}</ul></div>`;
    if (r.spray_schedule && r.spray_schedule.length) {
      const q = new URLSearchParams({ crop_code: r.crop_code, model_class: r.model_class, severity_pct: r.severity_pct, tank_code: currentTank(), start_date: $('start-date').value || todayIso() });
      h += `<div class="box"><h3>Lịch phun</h3><ul>${r.spray_schedule.map(s => `<li>Lần ${s.application}: ${vnDate(s.date)}</li>`).join('')}</ul>` +
        `<div class="actions"><a class="btn" id="ics-link" href="/api/calendar.ics?${q}" download>Tải lịch phun (.ics)</a></div>` +
        '<p class="muted">Mở tệp .ics bằng ứng dụng Lịch để được nhắc đúng ngày (tùy ứng dụng lịch trên máy; có thể nhập qua Google Calendar trên máy tính).</p></div>';
    }
    if (r.status === 'HEALTHY') h += '<div class="box">Lá không có vết bệnh: chưa cần phun. Tiếp tục theo dõi định kỳ.</div>';
    if (r.disclaimer) h += `<p class="disclaimer">${esc(r.disclaimer)}</p>`;
    $('plan').innerHTML = h;
  }

  // ------------------------------------------------------------------ ảnh mẫu
  async function loadSamples() {
    try {
      S.samples = (await api('/api/samples')).items;
    } catch (e) {
      S.samples = [];
    }
    $('samples-empty').hidden = S.samples.length > 0;
    const crops = [...new Set(S.samples.map(s => s.crop_code))];
    const cropName = c => ((S.meta.crops.find(x => x.code === c) || {}).name || c);
    $('sfilter').innerHTML = crops.length > 1 ? ['all', ...crops].map(c =>
      `<button type="button" data-c="${esc(c)}" aria-pressed="${c === S.sfilter}">${c === 'all' ? 'Tất cả' : esc(cropName(c))}</button>`).join('') : '';
    $('sfilter').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { S.sfilter = b.dataset.c; loadSamplesRender(); }));
    loadSamplesRender();
  }
  function loadSamplesRender() {
    $('sfilter').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.c === S.sfilter)));
    const items = S.samples.filter(s => S.sfilter === 'all' || s.crop_code === S.sfilter);
    $('samples').innerHTML = items.map(s =>
      `<button type="button" class="sample" data-id="${esc(s.id)}"><img loading="lazy" alt="" src="/api/samples/${encodeURIComponent(s.id)}/thumb">` +
      `<div><b>${esc(s.gt_disease_name || 'Lá khỏe')}</b>Đáp án: ${pct(s.gt_severity_pct)} · ${esc(s.gt_level_label || '')}</div></button>`).join('');
    $('samples').querySelectorAll('.sample').forEach(b => b.addEventListener('click', () => {
      const s = S.samples.find(x => x.id === b.dataset.id);
      S.sampleId = s.id;
      S.file = null;
      if ([...$('crop').options].some(o => o.value === s.crop_code)) $('crop').value = s.crop_code;
      $('preview-img').src = '/api/samples/' + encodeURIComponent(s.id) + '/thumb';
      $('preview').hidden = false;
      showTab('diagnose');
      diagnose(true);
    }));
  }

  // ------------------------------------------------------------------ lịch sử
  function saveHistory() {
    const r = S.rec;
    const h = store.get('ags-history', []);
    h.unshift({ t: new Date().toISOString(), crop: r.crop_name || r.crop_code, disease: r.disease_name || (r.status === 'HEALTHY' ? 'Lá khỏe' : r.status),
      pct: r.severity_pct, level: r.level_label || '', color: r.level_color || '#757575', sample: S.sampleId || null });
    store.set('ags-history', h.slice(0, 30));
  }
  function renderHistory() {
    const h = store.get('ags-history', []);
    $('history-empty').hidden = h.length > 0;
    $('history').innerHTML = h.map(x => `<li><b>${esc(x.disease)}</b> – ${esc(x.crop)} – <span style="color:${esc(x.color)}">${pct(x.pct)} ${esc(x.level)}</span>` +
      ` <span class="muted">${new Date(x.t).toLocaleString('vi-VN')}${x.sample ? ' · ảnh mẫu ' + esc(x.sample) : ''}</span></li>`).join('');
  }
  $('btn-history-clear').addEventListener('click', () => { store.set('ags-history', []); renderHistory(); });

  init();
})();
