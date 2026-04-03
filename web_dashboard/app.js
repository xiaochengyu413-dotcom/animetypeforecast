(function () {
  const data = window.DASHBOARD_DATA;

  if (!data || !Array.isArray(data.themes)) {
    document.body.innerHTML = "<main style='padding:32px;font-family:sans-serif;'>未找到 dashboard 数据，请先运行 build_web_dashboard.py。</main>";
    return;
  }

  const colorMap = {
    heat_stronger_than_rating: "#c75a37",
    rating_stronger_than_heat: "#0e7c73",
    roughly_aligned: "#d9a63c",
  };

  const state = {
    selectedTheme: data.themeOrder[0],
  };

  const themeMap = new Map(data.themes.map((item) => [item.theme, item]));

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "—";
    }
    return Number(value).toFixed(digits);
  }

  function formatSigned(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "—";
    }
    const number = Number(value);
    const prefix = number > 0 ? "+" : "";
    return `${prefix}${number.toFixed(digits)}`;
  }

  function formatPercent(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "—";
    }
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function deltaClass(value) {
    if (value > 0) return "positive";
    if (value < 0) return "negative";
    return "";
  }

  function encodePath(path) {
    return path
      .split("/")
      .map((part) => (part === "." || part === ".." ? part : encodeURIComponent(part)))
      .join("/");
  }

  function versionedAssetPath(path) {
    const version = data?.meta?.generatedAt || window.DASHBOARD_ASSET_VERSION || "";
    const encoded = encodePath(path);
    if (!version) {
      return encoded;
    }
    return `${encoded}?v=${encodeURIComponent(version)}`;
  }

  function createCard(label, value, detail) {
    return `
      <article class="metric-card">
        <small>${label}</small>
        <strong>${value}</strong>
        <span>${detail}</span>
      </article>
    `;
  }

  function createKpiCard(label, value, detail) {
    return `
      <article class="kpi-card">
        <small>${label}</small>
        <strong>${value}</strong>
        <p>${detail}</p>
      </article>
    `;
  }

  function createAnalysisCard(title, body) {
    return `
      <article class="analysis-card">
        <strong>${title}</strong>
        <p>${body}</p>
      </article>
    `;
  }

  function trendVariant(value) {
    if (typeof value !== "string") {
      return "neutral";
    }
    if (value.includes("上升")) {
      return "positive";
    }
    if (value.includes("下降")) {
      return "negative";
    }
    return "neutral";
  }

  function createTrendPill(label, value) {
    return `
      <span class="trend-pill ${trendVariant(value)}">
        <span class="trend-pill-label">${label}</span>
        <span class="trend-pill-value">${value}</span>
      </span>
    `;
  }

  function setImage(id, relativePath, alt) {
    const image = document.getElementById(id);
    image.alt = alt;
    image.src = versionedAssetPath(relativePath);
    image.onerror = function onError() {
      image.alt = `${alt}（图片不存在）`;
      image.removeAttribute("src");
    };
  }

  function renderHero() {
    const meta = data.meta;
    document.getElementById("hero-subtitle").textContent =
      `${meta.subtitle} 当前页面使用的最新观测季度为 ${meta.latestObservedQuarter}（${meta.latestObservedDate}），预测延伸到 ${meta.forecastEndQuarter}。`;

    document.getElementById("hero-chips").innerHTML = [
      `<span class="chip">${meta.themeCount} 个高层主题</span>`,
      `<span class="chip">建模覆盖 ${meta.dataFirstQuarter} - ${meta.dataLastQuarter}</span>`,
      `<span class="chip">未来 ${meta.forecastHorizonQuarters} 个季度</span>`,
      `<span class="chip">${meta.scenarioCount} 个权重场景</span>`,
    ].join("");

    const heroMetrics = [
      createCard(
        "热度榜首",
        `${data.overview.heatLeader.theme} #${data.overview.heatLeader.rank}`,
        `末期热度 ${formatNumber(data.overview.heatLeader.forecastFinalValue)}，相对最后实测 ${formatSigned(data.overview.heatLeader.delta)}`
      ),
      createCard(
        "评分榜首",
        `${data.overview.ratingLeader.theme} #${data.overview.ratingLeader.rank}`,
        `末期评分 ${formatNumber(data.overview.ratingLeader.forecastFinalValue)}，相对最后实测 ${formatSigned(data.overview.ratingLeader.delta)}`
      ),
      createCard(
        "热度上行主题",
        `${data.overview.popularityUpCount}/${data.meta.themeCount}`,
        `评分同时上行的主题只有 ${data.overview.ratingUpCount}/${data.meta.themeCount}`
      ),
      createCard(
        "非基线稳健性",
        `${data.overview.stableNonBaselineTop5Count}/${data.overview.nonBaselineScenarioCount}`,
        "非基线场景里，Top5 排名集合完全重合的次数"
      ),
    ];

    document.getElementById("hero-metrics").innerHTML = heroMetrics.join("");
  }

  function renderHighlights() {
    const heatBias = data.overview.strongestHeatBias;
    const ratingBias = data.overview.strongestRatingBias;
    const validation = data.validation.summary;

    const cards = [
      {
        label: "热度最强而评分靠后",
        value: heatBias.theme,
        detail: `热度第 ${heatBias.popularityRank}，评分第 ${heatBias.ratingRank}，错位 ${Math.abs(heatBias.rankGap)} 名。`,
      },
      {
        label: "评分最强而热度靠后",
        value: ratingBias.theme,
        detail: `评分第 ${ratingBias.ratingRank}，热度第 ${ratingBias.popularityRank}，错位 ${Math.abs(ratingBias.rankGap)} 名。`,
      },
      {
        label: "回测最稳",
        value: validation.bestTheme.theme,
        detail: `Prophet MAE ${formatNumber(validation.bestTheme.prophetMae)}，是当前验证里误差最低的主题。`,
      },
      {
        label: "最需要警惕",
        value: validation.hardestTheme.theme,
        detail: `Prophet MAE ${formatNumber(validation.hardestTheme.prophetMae)}，说明该主题波动更难预测。`,
      },
    ];

    document.getElementById("insight-grid").innerHTML = cards
      .map(
        (card) => `
          <article class="insight-card">
            <small>${card.label}</small>
            <strong>${card.value}</strong>
            <p class="sub">${card.detail}</p>
          </article>
        `
      )
      .join("");
  }

  function renderThemeHome() {
    const target = document.getElementById("theme-card-grid");
    target.innerHTML = data.themes
      .map((theme) => {
        const activeClass = theme.theme === state.selectedTheme ? "is-active" : "";
        return `
          <article class="theme-card ${activeClass}" data-theme-card="${theme.theme}">
            <div class="theme-card-top">
              <div>
                <small>Theme</small>
                <strong>${theme.theme}</strong>
              </div>
              <span class="theme-badge">${theme.comparison.comparisonLabelZh}</span>
            </div>
            <p class="sub">最后实测 ${theme.popularity.lastObservedQuarter}，预测延伸到 ${theme.popularity.forecastEndQuarter}。</p>
            <div class="theme-card-badges">
              <span class="theme-badge">热度 #${theme.comparison.popularityRank}</span>
              <span class="theme-badge">评分 #${theme.comparison.ratingRank}</span>
              <span class="theme-badge">可用季度 ${theme.readiness.usableQuarters ?? "—"}</span>
            </div>
            <div class="theme-card-stats">
              <div class="theme-stat">
                <small>热度变化</small>
                <strong class="delta ${deltaClass(theme.popularity.forecastDeltaFromLastActual)}">${formatSigned(theme.popularity.forecastDeltaFromLastActual)}</strong>
              </div>
              <div class="theme-stat">
                <small>评分变化</small>
                <strong class="delta ${deltaClass(theme.rating.forecastDeltaFromLastActual)}">${formatSigned(theme.rating.forecastDeltaFromLastActual)}</strong>
              </div>
            </div>
            <span class="theme-card-action">查看这个主题</span>
          </article>
        `;
      })
      .join("");

    document.querySelectorAll("[data-theme-card]").forEach((card) => {
      card.addEventListener("click", () => {
        const theme = card.getAttribute("data-theme-card");
        setSelectedTheme(theme, { scrollToDetail: true });
      });
    });
  }

  function renderLeaderboards() {
    const popularityBody = document.getElementById("popularity-table-body");
    const ratingBody = document.getElementById("rating-table-body");

    popularityBody.innerHTML = data.leaderboards.popularity
      .map((row) => {
        const activeClass = row.theme === state.selectedTheme ? "is-active" : "";
        return `
          <tr class="clickable-row ${activeClass}" data-theme="${row.theme}">
            <td><span class="rank-pill">${row.rank}</span></td>
            <td>${row.theme}</td>
            <td>${formatNumber(row.forecastFinalValue)}</td>
            <td><span class="delta ${deltaClass(row.forecastDeltaFromLastActual)}">${formatSigned(row.forecastDeltaFromLastActual)}</span></td>
          </tr>
        `;
      })
      .join("");

    ratingBody.innerHTML = data.leaderboards.rating
      .map((row) => {
        const activeClass = row.theme === state.selectedTheme ? "is-active" : "";
        return `
          <tr class="clickable-row ${activeClass}" data-theme="${row.theme}">
            <td><span class="rank-pill">${row.rank}</span></td>
            <td>${row.theme}</td>
            <td>${formatNumber(row.forecastFinalValue)}</td>
            <td><span class="delta ${deltaClass(row.forecastDeltaFromLastActual)}">${formatSigned(row.forecastDeltaFromLastActual)}</span></td>
          </tr>
        `;
      })
      .join("");

    document.querySelectorAll(".clickable-row").forEach((row) => {
      row.addEventListener("click", () => {
        const theme = row.getAttribute("data-theme");
        setSelectedTheme(theme);
      });
    });
  }

  function renderLegend() {
    const counts = data.comparison.counts;
    const legend = document.getElementById("comparison-legend");
    const items = [
      ["heat", "热度明显强于评分", counts.heat_stronger_than_rating],
      ["rating", "评分明显强于热度", counts.rating_stronger_than_heat],
      ["aligned", "两种排序大体一致", counts.roughly_aligned],
    ];
    legend.innerHTML = items
      .map(
        ([variant, label, count]) =>
          `<span class="legend-chip" data-variant="${variant}">${label} · ${count}</span>`
      )
      .join("");
  }

  function renderComparisonNarrative() {
    const rows = data.comparison.rows;
    const selected = themeMap.get(state.selectedTheme);
    const aligned = rows.find((row) => row.theme === selected.theme);
    const notes = [
      {
        title: "当前选中主题",
        body: `${selected.theme} 在热度榜排第 ${selected.comparison.popularityRank}，评分榜排第 ${selected.comparison.ratingRank}，属于“${selected.comparison.comparisonLabelZh}”。`,
      },
      {
        title: "热度最压过评分",
        body: `${data.overview.strongestHeatBias.theme} 的热度比评分高出 ${Math.abs(data.overview.strongestHeatBias.rankGap)} 个名次。`,
      },
      {
        title: "评分最压过热度",
        body: `${data.overview.strongestRatingBias.theme} 的评分比热度高出 ${Math.abs(data.overview.strongestRatingBias.rankGap)} 个名次。`,
      },
      {
        title: "当前主题的解释",
        body:
          aligned.rankGap < 0
            ? `它在热度维度更占优势，说明受众规模或讨论度可能高于它的评分排序。`
            : aligned.rankGap > 0
            ? `它在评分维度更占优势，说明口碑排序好于市场热度排序。`
            : "它在热度和评分两个目标下几乎给出相同的位置。",
      },
    ];

    document.getElementById("comparison-narrative").innerHTML = notes
      .map(
        (note) => `
          <article class="narrative-card">
            <strong>${note.title}</strong>
            <p>${note.body}</p>
          </article>
        `
      )
      .join("");
  }

  function svgEl(tag) {
    return document.createElementNS("http://www.w3.org/2000/svg", tag);
  }

  function renderScatter() {
    const svg = document.getElementById("comparison-scatter");
    svg.innerHTML = "";

    const rows = data.comparison.rows;
    const width = 520;
    const height = 420;
    const padding = { top: 28, right: 36, bottom: 56, left: 56 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxRank = data.meta.themeCount;
    const ticks = [...new Set([1, Math.ceil(maxRank / 3), Math.ceil((maxRank * 2) / 3), maxRank])];

    function x(rank) {
      return padding.left + ((rank - 1) / Math.max(maxRank - 1, 1)) * plotWidth;
    }

    function y(rank) {
      return padding.top + ((rank - 1) / Math.max(maxRank - 1, 1)) * plotHeight;
    }

    const bg = svgEl("rect");
    bg.setAttribute("x", padding.left);
    bg.setAttribute("y", padding.top);
    bg.setAttribute("width", plotWidth);
    bg.setAttribute("height", plotHeight);
    bg.setAttribute("fill", "rgba(255,255,255,0.58)");
    svg.appendChild(bg);

    ticks.forEach((tick) => {
      const vx = x(tick);
      const hy = y(tick);

      const vLine = svgEl("line");
      vLine.setAttribute("x1", vx);
      vLine.setAttribute("x2", vx);
      vLine.setAttribute("y1", padding.top);
      vLine.setAttribute("y2", height - padding.bottom);
      vLine.setAttribute("stroke", "rgba(45,43,51,0.12)");
      svg.appendChild(vLine);

      const hLine = svgEl("line");
      hLine.setAttribute("x1", padding.left);
      hLine.setAttribute("x2", width - padding.right);
      hLine.setAttribute("y1", hy);
      hLine.setAttribute("y2", hy);
      hLine.setAttribute("stroke", "rgba(45,43,51,0.12)");
      svg.appendChild(hLine);

      const xLabel = svgEl("text");
      xLabel.setAttribute("x", vx);
      xLabel.setAttribute("y", height - 24);
      xLabel.setAttribute("text-anchor", "middle");
      xLabel.setAttribute("fill", "#5d6776");
      xLabel.setAttribute("font-size", "12");
      xLabel.textContent = String(tick);
      svg.appendChild(xLabel);

      const yLabel = svgEl("text");
      yLabel.setAttribute("x", 36);
      yLabel.setAttribute("y", hy + 4);
      yLabel.setAttribute("text-anchor", "middle");
      yLabel.setAttribute("fill", "#5d6776");
      yLabel.setAttribute("font-size", "12");
      yLabel.textContent = String(tick);
      svg.appendChild(yLabel);
    });

    const diagonal = svgEl("line");
    diagonal.setAttribute("x1", x(1));
    diagonal.setAttribute("x2", x(maxRank));
    diagonal.setAttribute("y1", y(1));
    diagonal.setAttribute("y2", y(maxRank));
    diagonal.setAttribute("stroke", "rgba(29,36,48,0.45)");
    diagonal.setAttribute("stroke-dasharray", "6 6");
    svg.appendChild(diagonal);

    rows.forEach((row) => {
      const circle = svgEl("circle");
      const isSelected = row.theme === state.selectedTheme;
      circle.setAttribute("cx", x(row.ratingRank));
      circle.setAttribute("cy", y(row.popularityRank));
      circle.setAttribute("r", isSelected ? "8.5" : "6");
      circle.setAttribute("fill", colorMap[row.comparisonLabel] || "#5d6776");
      circle.setAttribute("stroke", isSelected ? "#1d2430" : "#ffffff");
      circle.setAttribute("stroke-width", isSelected ? "2.5" : "1.5");
      circle.style.cursor = "pointer";

      const title = svgEl("title");
      title.textContent = `${row.theme}: 热度第 ${row.popularityRank}，评分第 ${row.ratingRank}`;
      circle.appendChild(title);
      circle.addEventListener("click", () => setSelectedTheme(row.theme));
      svg.appendChild(circle);

      if (isSelected) {
        const label = svgEl("text");
        label.setAttribute("x", x(row.ratingRank) + 10);
        label.setAttribute("y", y(row.popularityRank) - 10);
        label.setAttribute("fill", "#1d2430");
        label.setAttribute("font-size", "13");
        label.setAttribute("font-weight", "700");
        label.textContent = row.theme;
        svg.appendChild(label);
      }
    });

    const xAxisLabel = svgEl("text");
    xAxisLabel.setAttribute("x", padding.left + plotWidth / 2);
    xAxisLabel.setAttribute("y", height - 6);
    xAxisLabel.setAttribute("text-anchor", "middle");
    xAxisLabel.setAttribute("fill", "#1d2430");
    xAxisLabel.setAttribute("font-size", "13");
    xAxisLabel.textContent = "评分排名（越靠左越强）";
    svg.appendChild(xAxisLabel);

    const yAxisLabel = svgEl("text");
    yAxisLabel.setAttribute("x", 18);
    yAxisLabel.setAttribute("y", padding.top + plotHeight / 2);
    yAxisLabel.setAttribute("text-anchor", "middle");
    yAxisLabel.setAttribute("fill", "#1d2430");
    yAxisLabel.setAttribute("font-size", "13");
    yAxisLabel.setAttribute("transform", `rotate(-90 18 ${padding.top + plotHeight / 2})`);
    yAxisLabel.textContent = "热度排名（越靠上越强）";
    svg.appendChild(yAxisLabel);
  }

  function renderThemeSelect() {
    const select = document.getElementById("theme-select");
    select.innerHTML = data.themeOrder
      .map((theme) => `<option value="${theme}">${theme}</option>`)
      .join("");
    select.value = state.selectedTheme;
    select.addEventListener("change", (event) => setSelectedTheme(event.target.value));
  }

  function renderThemeDetail() {
    const theme = themeMap.get(state.selectedTheme);
    const comparison = theme.comparison;
    const readiness = theme.readiness || {};
    const evaluation = theme.evaluation || {};
    const popularity = theme.popularity;
    const rating = theme.rating;
    const analysis = theme.analysis || {};

    document.getElementById("detail-theme-name").textContent = theme.theme;
    document.getElementById("detail-theme-label").textContent = comparison.comparisonLabelZh;
    document.getElementById("detail-theme-window").textContent =
      `预测窗口 ${popularity.forecastStartQuarter} - ${popularity.forecastEndQuarter}`;
    document.getElementById("theme-select").value = theme.theme;

    document.getElementById("detail-summary").textContent =
      `${theme.theme} 的最后实测季度是 ${popularity.lastObservedQuarter}。未来末期热度预测为 ${formatNumber(popularity.forecastFinalValue)}，相对最后实测 ${formatSigned(popularity.forecastDeltaFromLastActual)}；评分预测为 ${formatNumber(rating.forecastFinalValue)}，相对最后实测 ${formatSigned(rating.forecastDeltaFromLastActual)}。${analysis.summary ? ` ${analysis.summary}` : ""}`;

    document.getElementById("analysis-headline").textContent =
      analysis.headline || `${theme.theme} 分析`;
    document.getElementById("analysis-confidence").textContent =
      analysis.confidenceLabel || "谨慎参考";
    document.getElementById("analysis-trend-row").innerHTML = (analysis.trendDirections || [])
      .map((item) => createTrendPill(item.label, item.value))
      .join("");
    document.getElementById("analysis-plain-summary").textContent =
      analysis.plainSummary || analysis.summary || "当前缺少可自动生成的摘要。";
    document.getElementById("analysis-list").innerHTML = (analysis.bullets || [])
      .map((item) => createAnalysisCard(item.title, item.body))
      .join("");
    document.getElementById("analysis-conclusion").textContent =
      analysis.conclusion || "当前缺少可自动生成的结论。";

    document.getElementById("detail-kpis").innerHTML = [
      createKpiCard("热度排名", `#${comparison.popularityRank}`, `相对于评分排名差值 ${comparison.rankGap}`),
      createKpiCard("评分排名", `#${comparison.ratingRank}`, `双目标标签：${comparison.comparisonLabelZh}`),
      createKpiCard("热度变化", formatSigned(popularity.forecastDeltaFromLastActual), `最终热度 ${formatNumber(popularity.forecastFinalValue)}`),
      createKpiCard("评分变化", formatSigned(rating.forecastDeltaFromLastActual), `最终评分 ${formatNumber(rating.forecastFinalValue)}`),
      createKpiCard("预计供给量", formatNumber(popularity.forecastFinalTitleCount, 1), `未来末期标题数预测`),
      createKpiCard("可用季度", readiness.usableQuarters ?? "—", `建模覆盖率 ${formatPercent(readiness.usableCoverageRatio)}`),
      createKpiCard("Prophet MAE", formatNumber(evaluation.prophetMae), `naive MAE ${formatNumber(evaluation.naiveMae)}`),
      createKpiCard("建模跨度", readiness.firstQuarter ? `${readiness.firstQuarter} - ${readiness.lastQuarter}` : "—", `总作品数 ${readiness.totalTitles ?? "—"}`),
    ].join("");

    setImage("plot-popularity", theme.assets.futurePopularityPlot, `${theme.theme} 未来热度预测`);
    setImage("plot-rating", theme.assets.futureRatingPlot, `${theme.theme} 未来评分预测`);
    setImage("plot-comparison", theme.assets.comparisonTrendPlot, `${theme.theme} 热度与评分走势对照`);
    setImage("plot-evaluation", theme.assets.evaluationPlot, `${theme.theme} 历史回溯验证`);
    setImage(
      "plot-evaluation-window",
      theme.assets.evaluationWindowPlot,
      `${theme.theme} 2024Q1 到 2025Q4 评分三模型回溯验证对比`
    );
    setImage(
      "plot-popularity-evaluation-window",
      theme.assets.popularityEvaluationWindowPlot,
      `${theme.theme} 2024Q1 到 2025Q4 热度三模型回溯验证对比`
    );

    document.getElementById("scenario-rank-body").innerHTML = theme.scenarioRanks
      .map(
        (row) => `
          <tr>
            <td>${row.scenarioLabel}</td>
            <td>#${row.forecastRank}</td>
            <td>${formatNumber(row.forecastFinalValue)}</td>
            <td><span class="delta ${deltaClass(row.forecastDeltaFromLastActual)}">${formatSigned(row.forecastDeltaFromLastActual)}</span></td>
          </tr>
        `
      )
      .join("");

    const prophetVsNaive =
      evaluation.prophetBeatsNaive === true
        ? "这个主题在回测中，Prophet 的 MAE 低于 seasonal naive。"
        : "这个主题在回测中，seasonal naive 的 MAE 仍然不差于 Prophet。";

    document.getElementById("detail-reading-note").innerHTML = `
      <p>如果你想判断“这个主题究竟是更热，还是只是评分更高”，先看热度排名与评分排名的差值，再看第三张对照图。</p>
      <p>${prophetVsNaive}</p>
      <p>需要注意的是，2025Q4 虽然是当前归档快照下最新的完整播出季度，但在 2026-03-10 抓取时，投票等互动信号仍在继续累积，因此该季度的 actual 尤其是 popularity 指标可能会因成熟度滞后而偏低。</p>
      <p>在当前数据里，${theme.theme} 的建模覆盖率为 ${formatPercent(readiness.usableCoverageRatio)}，可用季度数为 ${readiness.usableQuarters ?? "—"}。</p>
    `;
  }

  function renderValidation() {
    const summary = data.validation.summary;
    document.getElementById("validation-cards").innerHTML = [
      {
        label: "Prophet 优于 naive",
        value: `${summary.prophetBetterCount}/${data.meta.themeCount}`,
        detail: "以 MAE 为准，统计多少主题回测时 Prophet 更好。",
      },
      {
        label: "平均 Prophet MAE",
        value: formatNumber(summary.avgProphetMae),
        detail: `平均 naive MAE 为 ${formatNumber(summary.avgNaiveMae)}。`,
      },
      {
        label: "平均 Prophet MAPE",
        value: `${formatNumber(summary.avgProphetMape)}%`,
        detail: `平均 naive MAPE 为 ${formatNumber(summary.avgNaiveMape)}%。`,
      },
      {
        label: "最佳回测主题",
        value: summary.bestTheme.theme,
        detail: `Prophet MAE ${formatNumber(summary.bestTheme.prophetMae)}。`,
      },
    ]
      .map(
        (card) => `
          <article class="mini-card">
            <small>${card.label}</small>
            <strong>${card.value}</strong>
            <p>${card.detail}</p>
          </article>
        `
      )
      .join("");

    document.getElementById("validation-note").textContent =
      `这里最值得注意的是：Prophet 并没有在所有主题上稳定优于 seasonal naive。当前验证中，Prophet 只在 ${summary.prophetBetterCount} 个主题上赢过 naive，而在 ${summary.naiveBetterCount} 个主题上没有优势，所以这套网页更适合做“探索性解释”和“结果汇报”，而不是当成已经完全定型的生产级预测系统。`;
  }

  function renderSensitivity() {
    setImage("sensitivity-summary", data.sensitivity.summaryPlot, "敏感性分析总览");
    document.getElementById("sensitivity-body").innerHTML = data.sensitivity.scenarios
      .map(
        (row) => `
          <tr>
            <td>${row.label}</td>
            <td>${formatNumber(row.futureRankSpearman, 3)}</td>
            <td>${formatPercent(row.futureTopkOverlapRatio)}</td>
            <td>${formatNumber(row.futureMeanAbsDiff, 2)}</td>
          </tr>
        `
      )
      .join("");
  }

  function renderRoadmap() {
    document.getElementById("roadmap-grid").innerHTML = data.roadmap
      .map(
        (item) => `
          <article class="roadmap-card">
            <h3>${item.title}</h3>
            <p class="sub">${item.detail}</p>
          </article>
        `
      )
      .join("");
  }

  function setSelectedTheme(theme, options = {}) {
    state.selectedTheme = theme;
    renderThemeHome();
    renderLeaderboards();
    renderComparisonNarrative();
    renderScatter();
    renderThemeDetail();
    if (options.scrollToDetail) {
      document.getElementById("detail").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderFooter() {
    const archiveInfo = data.meta.archiveName
      ? ` 数据源归档：${data.meta.archiveName}。`
      : "";
    document.getElementById("footer-meta").textContent =
      `页面生成时间：${data.meta.generatedAt}。当前网页展示的是截至 ${data.meta.latestObservedDate}（${data.meta.latestObservedQuarter}）纳入模型的本地结果。${archiveInfo}`;
  }

  function init() {
    renderHero();
    renderThemeHome();
    renderHighlights();
    renderLegend();
    renderThemeSelect();
    renderLeaderboards();
    renderValidation();
    renderSensitivity();
    renderRoadmap();
    renderFooter();
    renderComparisonNarrative();
    renderScatter();
    renderThemeDetail();
  }

  init();
})();
