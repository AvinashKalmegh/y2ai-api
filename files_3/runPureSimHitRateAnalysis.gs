/**
 * runPureSimHitRateAnalysis()
 * ─────────────────────────────────────────────────────────────────────────────
 * Applies Strategy 5 entry/exit conditions to the GS DM history sheets
 * and calculates per-ticker signal counts, hit rates, and avg returns.
 *
 * Strategy 5 Rules:
 *   Entry: DM (EMA5) >= 65 AND 5d MA of DM is rising
 *   Exit:  20d MA of DM drops below 50
 *   Hit:   Position achieves 10%+ return before exit
 *
 * NOTE: This uses price-only hit calculation (no price MA condition)
 *       because we need to isolate signal quality from the price filter.
 *       A second pass can apply the full Strategy 5 price filter.
 *
 * Run: runPureSimHitRateAnalysis()
 * ─────────────────────────────────────────────────────────────────────────────
 */

function runPureSimHitRateAnalysis() {

  // ── Sheet IDs ────────────────────────────────────────────────────────────
  const SHEETS = [
    { id: '1DgUqXnk9tXwoBdiR4yoFOClW3eIS7CVjqcW3RBBdFKk', label: '2016-2019', tab: 'DM_2016_2019' },
    { id: '1IuiXxcF6z4mL3Mex7Mrh7DFsf_RpNfxzweI_XfXyjKM', label: '2020-2023', tab: 'DM_2020_2023' },
    { id: '1GiLsxrgW-nssIhuUGYzn7SHYzcNjP4yg_7p1G2nPNiE', label: '2024-2026', tab: 'DM_2024_2026' },
  ];

  // ── Tickers to evaluate ──────────────────────────────────────────────────
  const TICKERS = [
    // PureSim 8 — primary question
    'AMD', 'ARM', 'DLR', 'EQIX', 'INTU', 'KLAC', 'MRVL', 'VRT',
    // Top Strategy 5 performers — for comparison / sanity check
    'APP', 'META', 'PLTR', 'VST', 'TEAM', 'CRWD', 'NVDA', 'TSLA',
    // Tradeable 26 sample — for comparison
    'DNN', 'CEG', 'MU', 'UUUU', 'LEU'
  ];

  // ── Column indices (0-based, 18-column schema) ───────────────────────────
  const COL = { DATE: 0, TICKER: 1, CLOSE: 2, DM: 13 };

  // ── Strategy 5 parameters ────────────────────────────────────────────────
  const ENTRY_DM_MIN   = 65;    // DM EMA5 must be >= this
  const EXIT_MA20      = 50;    // 20d MA of DM drops below this = exit
  const HIT_RETURN     = 0.10;  // 10% return = hit
  const FORWARD_DAYS   = 90;    // trading days to measure return

  Logger.log('════════════════════════════════════════════════════════════');
  Logger.log('STRATEGY 5 HIT RATE ANALYSIS — GS DM HISTORY');
  Logger.log('Entry: DM >= ' + ENTRY_DM_MIN + ' + 5d MA rising');
  Logger.log('Exit:  20d MA < ' + EXIT_MA20);
  Logger.log('Hit:   ' + (HIT_RETURN*100) + '% return within ' + FORWARD_DAYS + ' days');
  Logger.log('════════════════════════════════════════════════════════════');

  // ── Load all data ────────────────────────────────────────────────────────
  const tickerSet = new Set(TICKERS);
  const allRows   = {};  // ticker → sorted array of {date, close, dm}
  TICKERS.forEach(tk => allRows[tk] = []);

  for (const src of SHEETS) {
    Logger.log('Loading ' + src.label + '...');
    try {
      const ss    = SpreadsheetApp.openById(src.id);
      const sheet = ss.getSheetByName(src.tab);
      if (!sheet) { Logger.log('  Tab not found: ' + src.tab); continue; }

      const lastRow  = sheet.getLastRow();
      const readRows = Math.min(lastRow - 1, 40000);
      const startRow = lastRow - readRows + 1;
      const data     = sheet.getRange(startRow, 1, readRows, 18).getValues();

      for (const row of data) {
        const ticker = String(row[COL.TICKER]).trim().toUpperCase();
        if (!tickerSet.has(ticker)) continue;
        const dateStr = String(row[COL.DATE]).substring(0, 10);
        const close   = parseFloat(row[COL.CLOSE]) || 0;
        const dm      = parseFloat(row[COL.DM])    || 0;
        if (!dateStr || close <= 0 || dm <= 0) continue;
        allRows[ticker].push({ date: dateStr, close, dm });
      }
      Logger.log('  Loaded from ' + src.label);
    } catch(e) {
      Logger.log('  ERROR: ' + e);
    }
  }

  // Sort each ticker's history by date ascending
  for (const tk of TICKERS) {
    allRows[tk].sort((a, b) => a.date.localeCompare(b.date));
  }

  // ── Compute signals and hit rates ────────────────────────────────────────
  const results = [];

  for (const ticker of TICKERS) {
    const rows = allRows[ticker];
    if (rows.length < 25) {
      results.push({
        ticker, dataPoints: rows.length, signals: 0, hits: 0,
        hitRate: 0, avgReturn: 0, bestReturn: 0, worstReturn: 0,
        firstDate: 'N/A', lastDate: 'N/A', verdict: 'INSUFFICIENT DATA'
      });
      continue;
    }

    // Compute rolling indicators
    const n = rows.length;
    const dm5   = new Array(n).fill(0);  // 5d MA of DM
    const dm20  = new Array(n).fill(0);  // 20d MA of DM

    for (let i = 0; i < n; i++) {
      // 5d MA
      const start5 = Math.max(0, i - 4);
      let sum5 = 0;
      for (let j = start5; j <= i; j++) sum5 += rows[j].dm;
      dm5[i] = sum5 / (i - start5 + 1);

      // 20d MA
      const start20 = Math.max(0, i - 19);
      let sum20 = 0;
      for (let j = start20; j <= i; j++) sum20 += rows[j].dm;
      dm20[i] = sum20 / (i - start20 + 1);
    }

    // Find entry signals: DM >= 65 AND 5d MA rising (today > yesterday)
    let signals = 0, hits = 0, totalReturn = 0;
    let bestReturn = -999, worstReturn = 999;
    const signalDates = [];

    for (let i = 1; i < n; i++) {
      const dm     = rows[i].dm;
      const ma5    = dm5[i];
      const ma5prev = dm5[i-1];

      // Entry condition
      if (dm < ENTRY_DM_MIN) continue;
      if (ma5 <= ma5prev)    continue;
      // Don't enter if already in exit territory
      if (dm20[i] < EXIT_MA20) continue;

      signals++;
      const entryClose = rows[i].close;
      const entryDate  = rows[i].date;
      signalDates.push(entryDate);

      // Find exit — either MA20 < 50 or FORWARD_DAYS reached
      let exitReturn = 0;
      for (let j = i + 1; j < Math.min(i + FORWARD_DAYS + 1, n); j++) {
        exitReturn = (rows[j].close - entryClose) / entryClose;

        // Exit condition
        if (dm20[j] < EXIT_MA20) break;

        // If we've reached forward days limit
        if (j === Math.min(i + FORWARD_DAYS, n - 1)) break;
      }

      totalReturn += exitReturn;
      if (exitReturn >= HIT_RETURN) hits++;
      if (exitReturn > bestReturn)  bestReturn  = exitReturn;
      if (exitReturn < worstReturn) worstReturn = exitReturn;

      // Skip ahead to avoid overlapping signals (minimum 20 days between entries)
      i += 19;
    }

    const hitRate   = signals > 0 ? Math.round(hits / signals * 100) : 0;
    const avgReturn = signals > 0 ? Math.round(totalReturn / signals * 1000) / 10 : 0;

    const verdict = signals === 0 ? 'NO SIGNALS'
      : signals < 3             ? 'THIN (<3 signals)'
      : hitRate >= 65           ? 'STRONG — KEEP'
      : hitRate >= 50           ? 'QUALIFIED — KEEP'
      : hitRate >= 40           ? 'BORDERLINE — REVIEW'
      : 'BELOW THRESHOLD — REMOVE';

    results.push({
      ticker,
      dataPoints: rows.length,
      signals,
      hits,
      hitRate,
      avgReturn,
      bestReturn:  Math.round(bestReturn  * 1000) / 10,
      worstReturn: Math.round(worstReturn * 1000) / 10,
      firstDate:   rows[0].date,
      lastDate:    rows[n-1].date,
      verdict
    });
  }

  // ── Print results ────────────────────────────────────────────────────────
  Logger.log('');
  Logger.log('════════════════════════════════════════════════════════════');
  Logger.log('RESULTS — PURESIM 8');
  Logger.log('════════════════════════════════════════════════════════════');
  Logger.log('Ticker | DtPts | Signals | Hits | Hit% | AvgRet | Best  | Worst | Verdict');
  Logger.log('-------|-------|---------|------|------|--------|-------|-------|--------');

  const pureSim8 = ['AMD', 'ARM', 'DLR', 'EQIX', 'INTU', 'KLAC', 'MRVL', 'VRT'];
  for (const r of results.filter(r => pureSim8.includes(r.ticker))) {
    Logger.log(
      r.ticker.padEnd(6) + ' | ' +
      String(r.dataPoints).padStart(5) + ' | ' +
      String(r.signals).padStart(7) + ' | ' +
      String(r.hits).padStart(4) + ' | ' +
      String(r.hitRate).padStart(4) + '% | ' +
      String(r.avgReturn).padStart(6) + '% | ' +
      String(r.bestReturn).padStart(5) + '% | ' +
      String(r.worstReturn).padStart(5) + '% | ' +
      r.verdict
    );
  }

  Logger.log('');
  Logger.log('════════════════════════════════════════════════════════════');
  Logger.log('RESULTS — COMPARISON TICKERS');
  Logger.log('════════════════════════════════════════════════════════════');
  Logger.log('Ticker | DtPts | Signals | Hits | Hit% | AvgRet | Verdict');
  Logger.log('-------|-------|---------|------|------|--------|--------');

  const comparison = ['APP','META','PLTR','VST','TEAM','CRWD','NVDA','TSLA','DNN','CEG','MU','UUUU','LEU'];
  for (const r of results.filter(r => comparison.includes(r.ticker))) {
    Logger.log(
      r.ticker.padEnd(6) + ' | ' +
      String(r.dataPoints).padStart(5) + ' | ' +
      String(r.signals).padStart(7) + ' | ' +
      String(r.hits).padStart(4) + ' | ' +
      String(r.hitRate).padStart(4) + '% | ' +
      String(r.avgReturn).padStart(6) + '% | ' +
      r.verdict
    );
  }

  Logger.log('');
  Logger.log('════════════════════════════════════════════════════════════');
  Logger.log('SUMMARY — PURESIM 8');
  Logger.log('════════════════════════════════════════════════════════════');
  const strong    = results.filter(r => pureSim8.includes(r.ticker) && r.hitRate >= 65 && r.signals >= 3);
  const qualified = results.filter(r => pureSim8.includes(r.ticker) && r.hitRate >= 50 && r.hitRate < 65 && r.signals >= 3);
  const borderline= results.filter(r => pureSim8.includes(r.ticker) && r.hitRate >= 40 && r.hitRate < 50 && r.signals >= 3);
  const below     = results.filter(r => pureSim8.includes(r.ticker) && r.hitRate < 40  && r.signals >= 3);
  const thin      = results.filter(r => pureSim8.includes(r.ticker) && r.signals < 3);

  Logger.log('STRONG (65%+, 3+ signals):     ' + strong.map(r=>r.ticker).join(', '));
  Logger.log('QUALIFIED (50-65%, 3+ signals):' + qualified.map(r=>r.ticker).join(', '));
  Logger.log('BORDERLINE (40-50%):           ' + borderline.map(r=>r.ticker).join(', '));
  Logger.log('BELOW THRESHOLD (<40%):        ' + below.map(r=>r.ticker).join(', '));
  Logger.log('THIN (<3 signals):             ' + thin.map(r=>r.ticker).join(', '));
  Logger.log('════════════════════════════════════════════════════════════');
}
