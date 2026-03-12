function validateOnly() {
  Logger.log('============================================================');
  Logger.log('VALIDATION ONLY — NO TRANSFER WILL OCCUR');
  Logger.log('============================================================');

  let staging, production;
  try { staging    = SpreadsheetApp.openById(STAGING_ID);    } catch(e) { Logger.log('FATAL: Cannot open staging: ' + e);    return; }
  try { production = SpreadsheetApp.openById(PRODUCTION_ID); } catch(e) { Logger.log('FATAL: Cannot open production: ' + e); return; }

  const errors = [], warnings = [];

  // ── DM_2024_2026 ──────────────────────────────────────────
  Logger.log('');
  Logger.log('--- DM_2024_2026 ---');
  const stagingHistory = staging.getSheetByName(HISTORY_TAB);
  const prodHistory    = production.getSheetByName(HISTORY_TAB);

  if (stagingHistory && prodHistory) {
    const stagingData  = stagingHistory.getDataRange().getValues();
    const prodLastRow  = prodHistory.getLastRow();
    const prodRowCount = prodLastRow - 1;
    const stagingDataRows = stagingData.slice(1);

    // Sort oldest-first so slice logic correctly identifies new rows at the tail
    stagingDataRows.sort((a, b) => {
      const da = parseDateSafe_(a[TCOL.DATE]);
      const db = parseDateSafe_(b[TCOL.DATE]);
      if (!da || !db) return 0;
      return da - db;
    });

    const newRowCount = stagingDataRows.length - prodRowCount;

    // Read newest rows from TOP of production (newest-first order)
    const TAIL = 10000;
    const tailCount = Math.min(TAIL, prodRowCount);
    const prodTail  = prodHistory.getRange(2, 1, tailCount, 18).getValues();
    const prodKeys  = buildTickerDateKeys_(prodTail);

    Logger.log('Staging: ' + stagingDataRows.length + ' rows | Production: ' + prodRowCount + ' | New by count: ' + newRowCount + ' | Tail read: ' + tailCount + ' rows');

    if (newRowCount > 0) {
      const candidateRows = stagingDataRows.slice(stagingDataRows.length - newRowCount);
      const newRows = candidateRows.filter(row => {
        const key = makeKey_(row[TCOL.TICKER], row[TCOL.DATE]);
        return key && !prodKeys.has(key);
      });
      const dupsBlocked = candidateRows.length - newRows.length;
      if (dupsBlocked > 0) warnings.push(dupsBlocked + ' duplicate row(s) would be blocked.');
      if (newRows.length > 0) {
        validateHistoryRows_(newRows, stagingDataRows, prodKeys, errors, warnings);
      }
    } else {
      Logger.log('No new rows to validate.');
    }
  } else {
    if (!stagingHistory) errors.push('DM_2024_2026 not found in staging.');
    if (!prodHistory)    errors.push('DM_2024_2026 not found in production.');
  }

  // ── DM_Latest ─────────────────────────────────────────────
  Logger.log('');
  Logger.log('--- DM_Latest ---');
  const stagingLatest = staging.getSheetByName(LATEST_TAB);
  const prodLatest    = production.getSheetByName(LATEST_TAB);

  if (stagingLatest && prodLatest) {
    const rawData = stagingLatest.getDataRange().getValues();
    if (rawData.length <= 1) {
      errors.push('DM_Latest is empty in staging.');
    } else {
      Logger.log('DM_Latest staging rows: ' + (rawData.length - 1));
      const prodData = prodLatest.getDataRange().getValues();
      if (prodData.length > 1) {
        Logger.log('DM_Latest production date: ' + formatDateSafe_(prodData[1][TLCOL.DATE]) + ' ← will be replaced');
      }
      validateLatestRows_(rawData.slice(1), errors, warnings);
    }
  } else {
    if (!stagingLatest) errors.push('DM_Latest not found in staging.');
    if (!prodLatest)    errors.push('DM_Latest not found in production.');
  }

  // ── Duplicate check on production top (newest rows) ──────
  Logger.log('');
  Logger.log('--- PRODUCTION DUPLICATE CHECK (top 10,000 rows) ---');
  try {
    const prodHistoryForDup = production.getSheetByName(HISTORY_TAB);
    if (prodHistoryForDup) {
      const dupLastRow   = prodHistoryForDup.getLastRow();
      const dupTailCount = Math.min(10000, dupLastRow - 1);
      const dupData      = prodHistoryForDup.getRange(2, 1, dupTailCount, 18).getValues();

      const dupSeen = new Set();
      const dupDates = {};
      let dupCount = 0;

      for (const row of dupData) {
        const key = makeKey_(row[TCOL.TICKER], row[TCOL.DATE]);
        if (!key) continue;
        if (dupSeen.has(key)) {
          dupCount++;
          const d = formatDateSafe_(row[TCOL.DATE]);
          if (!dupDates[d]) dupDates[d] = 0;
          dupDates[d]++;
        } else {
          dupSeen.add(key);
        }
      }

      if (dupCount === 0) {
        Logger.log('No duplicates found in production top rows. ✓');
      } else {
        const dupDateSummary = Object.keys(dupDates).sort()
          .map(d => d + '(' + dupDates[d] + ')').join(', ');
        errors.push('Production contains ' + dupCount + ' duplicate row(s) in top rows. Affected dates: ' + dupDateSummary + '. Run dedup recovery before transferring.');
        Logger.log('DUPLICATES FOUND: ' + dupCount + ' rows on dates: ' + dupDateSummary);
      }
    }
  } catch(e) {
    warnings.push('Duplicate check failed: ' + e);
  }

  // ── Report ────────────────────────────────────────────────
  Logger.log('');
  Logger.log('--- RESULTS ---');
  Logger.log('Errors: ' + errors.length + '  |  Warnings: ' + warnings.length);
  errors.forEach(e   => Logger.log('  ERROR: ' + e));
  warnings.forEach(w => Logger.log('  WARNING: ' + w));
  Logger.log('');
  Logger.log(errors.length === 0
    ? 'VALIDATION PASSED — safe to run validateAndTransfer()'
    : 'VALIDATION FAILED — fix ' + errors.length + ' error(s) before transferring.');
}
