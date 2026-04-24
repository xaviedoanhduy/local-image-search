const q = document.getElementById('q');
const btn = document.getElementById('btn');
const grid = document.getElementById('grid');
const meta = document.getElementById('meta');
const total = document.getElementById('total');
const lb = document.getElementById('lb');
const lbImg = document.getElementById('lb-img');
const lbLink = document.getElementById('lb-link');
const lbLoading = document.getElementById('lb-loading');

q.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
q.addEventListener('input', () => {
  const empty = q.value.trim() === '';
  btn.disabled = empty;
  document.getElementById('sort').disabled = empty;
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });
lb.addEventListener('click', e => { if (e.target === lb) closeLb(); });
window.addEventListener('popstate', () => { if (lb.classList.contains('open')) closeLb(false); });

fetch('/health').then(r => r.json()).then(d => {
  const count = d.embeddings.toLocaleString();
  total.textContent = count + ' photos indexed';
  document.getElementById('placeholder').textContent = `Enter a description to search ${count} company photos.`;
});

async function doSearch() {
  const query = q.value.trim();
  if (!query) return;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Searching…';
  grid.innerHTML = '';
  meta.textContent = '';

  const sort = document.getElementById('sort').value;
  const minRel = sort === 'quality' ? 0.08 : 0.0;

  try {
    const t0 = Date.now();
    const res = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 20, sort_by: sort, quality_weight: 0.6, min_relevance: minRel }),
    });
    const data = await res.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    meta.textContent = `${data.results.length} results from ${data.total_images.toLocaleString()} photos — ${elapsed}s`;

    if (!data.results.length) {
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1">No results found. Try a different description.</div>';
      return;
    }

    // Build result cards; store data on element to avoid inline onclick escaping issues
    const cards = data.results.map(r => {
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `
        <div class="card-thumb">
          <div class="card-thumb-spinner"><span class="spinner"></span></div>
          <img src="${r.file_id ? `/image/${r.file_id}?size=400` : ''}"
               loading="lazy" alt="${escHtml(r.filename)}">
        </div>
        <div class="card-info">
          <div class="card-name" title="${escHtml(r.filename)}">${escHtml(r.filename)}</div>
          <div class="card-scores">
            <span class="badge badge-rel">relevance ${r.relevance}</span>
            ${r.aesthetic_score != null ? `<span class="badge badge-aes">quality ${r.aesthetic_score}</span>` : ''}
          </div>
          ${r.drive_url ? `<a class="card-link" href="${escHtml(r.drive_url)}" target="_blank">Open in Drive ↗</a>` : ''}
        </div>`;

      const img = div.querySelector('img');
      const thumbSpinner = div.querySelector('.card-thumb-spinner');
      img.onload = () => { thumbSpinner.style.display = 'none'; };
      img.onerror = () => { thumbSpinner.style.display = 'none'; };

      // Attach click handler via JS (no inline onclick, no escaping issues)
      if (r.file_id) {
        div.addEventListener('click', e => {
          if (e.target.closest('.card-link')) return; // let Drive link handle itself
          openLb(r.file_id, r.drive_url, r.filename);
        });
      }
      return div;
    });

    grid.innerHTML = '';
    cards.forEach(c => grid.appendChild(c));
  } finally {
    btn.disabled = false;
    btn.textContent = 'Search';
  }
}

function openLb(fileId, driveUrl, name) {
  lbLink.href = driveUrl || '#';
  lbLink.style.display = driveUrl ? 'inline-block' : 'none';
  lb.classList.add('open');
  history.pushState({ lightbox: true }, '');

  lbImg.style.display = 'none';
  lbLoading.style.display = 'flex';
  lbImg.alt = name || '';

  lbImg.onload = () => {
    lbLoading.style.display = 'none';
    lbImg.style.display = 'block';
    // Load full-size in background, swap when ready
    const full = new Image();
    full.onload = () => { lbImg.src = full.src; };
    full.src = `/image/${fileId}?size=1600`;
  };
  lbImg.src = `/image/${fileId}?size=400`;
}

function closeLb(popHistory = true) {
  if (!lb.classList.contains('open')) return;
  lb.classList.remove('open');
  lbImg.src = '';
  lbImg.style.display = 'none';
  lbLoading.style.display = 'none';
  if (popHistory) history.back();
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
