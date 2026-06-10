const DATA_URL = "../data/demo/demo_data.json?t=" + Date.now();

let demoData = null;
let selectedProductId = null;
let selectedCaseId = null;

const els = {
  statusStrip: document.getElementById("statusStrip"),
  productSearch: document.getElementById("productSearch"),
  productSelect: document.getElementById("productSelect"),
  selectedProduct: document.getElementById("selectedProduct"),
  similarTable: document.getElementById("similarTable"),
  caseSelect: document.getElementById("caseSelect"),
  caseMetrics: document.getElementById("caseMetrics"),
  stateList: document.getElementById("stateList"),
  dqnList: document.getElementById("dqnList"),
  targetList: document.getElementById("targetList"),
  metricTable: document.getElementById("metricTable"),
  metricBars: document.getElementById("metricBars"),
  trainingChart: document.getElementById("trainingChart"),
};

const formatNumber = new Intl.NumberFormat("vi-VN", {
  maximumFractionDigits: 4,
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pct(value) {
  return `${formatNumber.format((Number(value) || 0) * 100)}%`;
}

function itemLabel(item) {
  const raw = item.raw_product_id ? `Product ${item.raw_product_id}` : `Item ${item.item_id}`;
  return `${item.name} (${raw}, idx ${item.item_id})`;
}

function renderStatus() {
  const meta = demoData.meta;
  const dataset = demoData.dataset;
  const model = meta.model;

  const statusItems = [];
  if (dataset.full?.users) {
    statusItems.push(`${formatNumber.format(dataset.full.users)} full users`);
  }

  const demoCaseUsers = new Set((demoData.cases || []).map((row) => row.user_id)).size;

  statusItems.push(
    `${formatNumber.format(dataset.eval.users)} test users`,
    `${formatNumber.format(demoCaseUsers)} demo case users`,
    `${formatNumber.format(demoData.cases?.length || 0)} demo cases`,
    `${formatNumber.format(dataset.eval.interactions)} test interactions`,
    `${dataset.valid_actions} valid actions`,
    `Top-${meta.top_k}`,
    model.available ? `Model action_dim ${model.action_dim}` : "No DQN checkpoint",
  );

  els.statusStrip.innerHTML = statusItems
    .map((text) => `<span>${escapeHtml(text)}</span>`)
    .join("");
}

function fillProductOptions(filter = "") {
  const query = filter.trim().toLowerCase();
  const options = demoData.product_options.filter((item) => {
    if (!query) return true;
    return itemLabel(item).toLowerCase().includes(query);
  });

  const visible = options.slice(0, 250);
  els.productSelect.innerHTML = visible
    .map((item) => {
      const selected = Number(item.item_id) === Number(selectedProductId) ? "selected" : "";
      return `<option value="${item.item_id}" ${selected}>${escapeHtml(itemLabel(item))}</option>`;
    })
    .join("");

  if (!visible.some((item) => Number(item.item_id) === Number(selectedProductId))) {
    selectedProductId = visible[0]?.item_id ?? demoData.product_options[0]?.item_id;
  }
}

function renderSelectedProduct() {
  const item = demoData.items[String(selectedProductId)] || demoData.product_options[0];
  if (!item) {
    els.selectedProduct.innerHTML = `<div class="empty">Không có sản phẩm để hiển thị.</div>`;
    return;
  }

  els.selectedProduct.innerHTML = `
    <div>
      <strong>${escapeHtml(item.name)}</strong>
      <span class="product-id">Indexed item ${item.item_id} · Raw product ${item.raw_product_id ?? "N/A"}</span>
    </div>
    <div class="item-meta">
      <span>Popularity ${formatNumber.format(item.popularity_count || 0)}</span>
      <span>Aisle ${item.aisle_id ?? "N/A"}</span>
      <span>Department ${item.department_id ?? "N/A"}</span>
    </div>
  `;
}

function renderSimilarTable() {
  const rows = demoData.similar_items[String(selectedProductId)] || [];

  if (!rows.length) {
    els.similarTable.innerHTML = `<div class="empty">Chưa có recommendation cho sản phẩm này.</div>`;
    return;
  }

  els.similarTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th style="width:34%">Recommended product</th>
          <th>Final score</th>
          <th>Embedding sim</th>
          <th>Co-occurrence</th>
          <th>Popularity</th>
        </tr>
      </thead>
      <tbody>
        ${rows
      .map(
        (item) => `
              <tr>
                <td>
                  <span class="product-name">${escapeHtml(item.name)}</span>
                  <span class="product-id">idx ${item.item_id} · raw ${item.raw_product_id ?? "N/A"}</span>
                </td>
                <td><span class="score good">${formatNumber.format(item.score)}</span></td>
                <td>${formatNumber.format(item.embedding_similarity)}</td>
                <td>${formatNumber.format(item.cooccurrence_count)}</td>
                <td>${formatNumber.format(item.popularity_count || 0)}</td>
              </tr>
            `,
      )
      .join("")}
      </tbody>
    </table>
  `;
}

function fillCaseOptions() {
  els.caseSelect.innerHTML = demoData.cases
    .map((caseRow) => {
      const selected = caseRow.case_id === selectedCaseId ? "selected" : "";
      return `<option value="${caseRow.case_id}" ${selected}>${escapeHtml(caseRow.label)}</option>`;
    })
    .join("");
}

function metricCard(label, value, className = "") {
  return `
    <div class="metric-card">
      <span>${escapeHtml(label)}</span>
      <strong class="${className}">${escapeHtml(value)}</strong>
    </div>
  `;
}

function renderCaseMetrics(caseRow) {
  const rewardClass = caseRow.reward > 0 ? "score good" : "score bad";
  els.caseMetrics.innerHTML = [
    metricCard("Reward", formatNumber.format(caseRow.reward), rewardClass),
    metricCard("Hits trong Top-5", `${caseRow.hits}/5`, caseRow.hits > 0 ? "score good" : "score bad"),
    metricCard("HitRate@5", pct(caseRow.hit_rate_at_5), caseRow.hit_rate_at_5 > 0 ? "score good" : "score bad"),
    metricCard("User / Window", `${caseRow.user_id} / ${caseRow.pointer}`),
  ].join("");
}

function renderItemList(container, items, mode) {
  container.innerHTML = items
    .map((item, idx) => {
      const classes = ["item-row"];
      if (mode === "recommendation") {
        classes.push(item.is_hit ? "hit" : "miss");
      }

      let meta = "";
      if (mode === "recommendation") {
        meta = `
          <span>Rank ${item.rank}</span>
          <span>Q ${formatNumber.format(item.q_value)}</span>
          <span>Bonus ${formatNumber.format(item.recent_bonus)}</span>
          <span>Score ${formatNumber.format(item.score)}</span>
          <span class="${item.is_hit ? "hit-pill" : "miss-pill"}">${item.is_hit ? "HIT" : "MISS"}</span>
        `;
      } else {
        meta = `
          <span>#${idx + 1}</span>
          <span>idx ${item.item_id}</span>
          <span>Pop ${formatNumber.format(item.popularity_count || 0)}</span>
        `;
      }

      return `
        <div class="${classes.join(" ")}">
          <span class="product-name">${escapeHtml(item.name)}</span>
          <span class="product-id">idx ${item.item_id} · raw ${item.raw_product_id ?? "N/A"}</span>
          <div class="item-meta">${meta}</div>
        </div>
      `;
    })
    .join("");
}

function renderCase() {
  const caseRow =
    demoData.cases.find((row) => row.case_id === selectedCaseId) || demoData.cases[0];

  if (!caseRow) {
    els.stateList.innerHTML = `<div class="empty">Không có case demo.</div>`;
    return;
  }

  selectedCaseId = caseRow.case_id;
  renderCaseMetrics(caseRow);
  renderItemList(els.stateList, caseRow.state, "state");
  renderItemList(els.dqnList, caseRow.recommendations, "recommendation");
  renderItemList(els.targetList, caseRow.target, "target");
}

function renderMetricTable() {
  const rows = demoData.metrics.rows || [];
  if (!rows.length) {
    els.metricTable.innerHTML = `<div class="empty">Không tìm thấy evaluation CSV.</div>`;
    return;
  }

  els.metricTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Type</th>
          <th>Avg Reward</th>
          <th>HitRate@5</th>
        </tr>
      </thead>
      <tbody>
        ${rows
      .map(
        (row) => `
              <tr>
                <td><strong>${escapeHtml(row.method)}</strong></td>
                <td>${escapeHtml(row.type || "")}</td>
                <td><span class="score ${Number(row.average_reward) >= 0 ? "good" : "warn"}">${formatNumber.format(row.average_reward)}</span></td>
                <td><span class="score good">${pct(row.hit_rate_at_5)}</span></td>
              </tr>
            `,
      )
      .join("")}
      </tbody>
    </table>
  `;
}

function renderMetricBars() {
  const rows = demoData.metrics.rows || [];
  const maxHit = Math.max(...rows.map((row) => Number(row.hit_rate_at_5) || 0), 0.001);

  els.metricBars.innerHTML = rows
    .map((row) => {
      const width = `${Math.max(2, ((Number(row.hit_rate_at_5) || 0) / maxHit) * 100)}%`;
      const type = row.type === "dqn" ? "dqn" : "baseline";
      return `
        <div class="bar-row">
          <strong>${escapeHtml(row.method)}</strong>
          <div class="bar-track">
            <div class="bar-fill ${type}" style="width:${width}"></div>
          </div>
          <span class="score">${pct(row.hit_rate_at_5)}</span>
        </div>
      `;
    })
    .join("");
}

function renderTrainingChart() {
  const points = demoData.training.points || [];
  if (!points.length) {
    els.trainingChart.innerHTML = `<div class="empty">Không tìm thấy training log.</div>`;
    return;
  }

  const width = 960;
  const height = 180;
  const pad = 20;
  const rewards = points.map((point) => Number(point.reward));
  const minReward = Math.min(...rewards);
  const maxReward = Math.max(...rewards);
  const rewardSpan = Math.max(1, maxReward - minReward);
  const xStep = (width - pad * 2) / Math.max(1, points.length - 1);

  const path = points
    .map((point, idx) => {
      const x = pad + idx * xStep;
      const y = height - pad - ((Number(point.reward) - minReward) / rewardSpan) * (height - pad * 2);
      return `${idx === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  els.trainingChart.innerHTML = `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" role="img" aria-label="Training reward chart">
      <line x1="${pad}" x2="${width - pad}" y1="${height - pad}" y2="${height - pad}" stroke="#d8dfdc" />
      <line x1="${pad}" x2="${pad}" y1="${pad}" y2="${height - pad}" stroke="#d8dfdc" />
      <path d="${path}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linejoin="round" />
      <text x="${pad}" y="${pad - 4}" font-size="12" fill="#68716f">max ${formatNumber.format(maxReward)}</text>
      <text x="${pad}" y="${height - 5}" font-size="12" fill="#68716f">min ${formatNumber.format(minReward)}</text>
    </svg>
  `;
}

function switchTab(tabName) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  els.productSearch.addEventListener("input", () => {
    fillProductOptions(els.productSearch.value);
    renderSelectedProduct();
    renderSimilarTable();
  });

  els.productSelect.addEventListener("change", () => {
    selectedProductId = Number(els.productSelect.value);
    renderSelectedProduct();
    renderSimilarTable();
  });

  els.caseSelect.addEventListener("change", () => {
    selectedCaseId = els.caseSelect.value;
    renderCase();
  });
}

function initialSelections() {
  selectedProductId = demoData.product_options[0]?.item_id ?? null;
  selectedCaseId = demoData.cases[0]?.case_id ?? null;
}

function renderAll() {
  renderStatus();
  fillProductOptions();
  renderSelectedProduct();
  renderSimilarTable();
  fillCaseOptions();
  renderCase();
  renderMetricTable();
  renderMetricBars();
  renderTrainingChart();
}

async function boot() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    demoData = await response.json();
    initialSelections();
    bindEvents();
    renderAll();
  } catch (error) {
    document.body.innerHTML = `
      <div class="app-shell">
        <section class="panel active">
          <h1>Không tải được demo data</h1>
          <p>Hãy chạy <strong>python demo/build_demo_data.py</strong>, sau đó mở bằng local server từ root project.</p>
          <p class="product-id">${escapeHtml(error.message)}</p>
        </section>
      </div>
    `;
  }
}

boot();
