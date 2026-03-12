function validateAndTransfer() {
  Logger.log('============================================================');
  Logger.log('FLOWOS DATA VALIDATION & TRANSFER');
  Logger.log('Started: ' + new Date().toLocaleString());
  Logger.log('============================================================');

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const auditSheet = getOrCreateAuditSheet_(ss);
  const runTimestamp = new Date();
  const errors = [];
  const warnings = [];

  let staging, production;
  try { staging    = SpreadsheetApp.openById(STAGING_ID);    } catch(e) { logAudit_(auditSheet, runTimestamp, 'FAIL', 'Cannot open staging: ' + e); return; }
  try { production = SpreadsheetApp.openById(PRODUCTION_ID); } catch(e) { logAudit_(auditSheet, runTimestamp, 'FAIL', 'Cannot open production: ' + e); return; }

  // ── DM_2024_2026 ──────────────────────────────────────────
  Logger.log('');
  Logger.log('--- VALIDATING DM_2024_2026 ---');

  const stagingHistory = staging.getSheetByName(HISTORY_TAB);
  const prodHistory    = production.getSheetByName(HISTORY_TAB);
  let historyNewRows   = null;

  if (!stagingHistory) { errors.push('DM_2024_2026 not found in staging.');    }
  else if (!prodHistory) { errors.push('DM_2024_2026 not found in production.'); }
  else {
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

    Logger.log('Staging rows: ' + stagingDataRows.length + ' | Production rows: ' + prodRowCount + ' | New by count: ' + newRowCount);

    // Read newest rows from TOP of production (newest-first order)
    const TAIL = 10000;
    const tailCount = Math.min(TAIL, prodRowCount);
    const prodTail  = prodHistory.getRange(2, 1, tailCount, 18).getValues();
    const prodKeys  = buildTickerDateKeys_(prodTail);
    Logger.log('Production top read: ' + tailCount + ' rows for duplicate check. Keys: ' + prodKeys.size);

    if (newRowCount <= 0) {
      warnings.push('No new rows in DM_2024_2026 — staging and production row counts are equal.');
      Logger.log('WARNING: Nothing new to transfer.');
    } else if (newRowCount > 5000) {
      errors.push('Too many new rows (' + newRowCount + '). Expected <5000. Review manually.');
    } else {
      // Take the last newRowCount rows from staging as candidates (newest, after oldest-first sort)
      const candidateRows = stagingDataRows.slice(stagingDataRows.length - newRowCount);

      // Filter out any that already exist in the production top (duplicate safety net)
      historyNewRows = candidateRows.filter(row => {
        const key = makeKey_(row[TCOL.TICKER], row[TCOL.DATE]);
        return key && !prodKeys.has(key);
      });

      const dupsBlocked = candidateRows.length - historyNewRows.length;
      if (dupsBlocked > 0) {
        warnings.push(dupsBlocked + ' duplicate row(s) blocked — already exist in production.');
      }

      Logger.log('Candidate rows: ' + candidateRows.length + ' | Duplicates blocked: ' + dupsBlocked + ' | To validate: ' + historyNewRows.length);
      if (historyNewRows.length > 0) {
        const jumps = [];
        validateHistoryRows_(historyNewRows, stagingDataRows, prodKeys, errors, warnings, jumps);
      }
    }
  }

  // ── DM_Latest ─────────────────────────────────────────────
  Logger.log('');
  Logger.log('--- VALIDATING DM_Latest ---');

  const stagingLatest = staging.getSheetByName(LATEST_TAB);
  const prodLatest    = production.getSheetByName(LATEST_TAB);
  let latestData      = null;

  if (!stagingLatest) { errors.push('DM_Latest not found in staging.');    }
  else if (!prodLatest) { errors.push('DM_Latest not found in production.'); }
  else {
    const rawData = stagingLatest.getDataRange().getValues();
    if (rawData.length <= 1) {
      errors.push('DM_Latest is empty (header only).');
    } else {
      latestData = rawData;
      Logger.log('DM_Latest rows: ' + (rawData.length - 1));
      validateLatestRows_(rawData.slice(1), errors, warnings);
    }
  }

  // ── Decision ──────────────────────────────────────────────
  Logger.log('');
  Logger.log('--- VALIDATION RESULTS ---');
  Logger.log('Errors: ' + errors.length + '  |  Warnings: ' + warnings.length);
  errors.forEach(e   => Logger.log('  ERROR: ' + e));
  warnings.forEach(w => Logger.log('  WARNING: ' + w));

  if (errors.length > 0) {
    Logger.log('');
    Logger.log('TRANSFER BLOCKED — production was NOT modified.');
    logAudit_(auditSheet, runTimestamp, 'BLOCKED',
      errors.length + ' errors, ' + warnings.length + ' warnings. ' + errors.join(' | '));
    return;
  }

  // ── Transfer ──────────────────────────────────────────────
  Logger.log('');
  Logger.log('--- TRANSFERRING TO PRODUCTION ---');
  let transferred = 0;

  if (historyNewRows && historyNewRows.length > 0) {
    // Reverse to newest-first order for insertion at top of production
    historyNewRows.reverse();

    const originalLastRow = prodHistory.getLastRow();

    // Insert blank rows at row 2 (after header) to make space
    prodHistory.insertRowsAfter(1, historyNewRows.length);

    // Write new rows at row 2
    prodHistory.getRange(2, 1, historyNewRows.length, historyNewRows[0].length).setValues(historyNewRows);
    SpreadsheetApp.flush();

    const newLastRow = prodHistory.getLastRow();
    const expectedLastRow = originalLastRow + historyNewRows.length;
    Logger.log('DM_2024_2026 inserted: ' + historyNewRows.length + ' rows at top | Expected last row: ' + expectedLastRow + ' | Actual last row: ' + newLastRow);
    if (newLastRow < expectedLastRow) {
      errors.push('Write verification FAILED — expected row ' + expectedLastRow + ' but got ' + newLastRow + '. Production may be partially written.');
    } else {
      Logger.log('Write verification PASSED ✓');
    }
    transferred++;
  }

  if (latestData && latestData.length > 1) {
    const prodLastRow = prodLatest.getLastRow();
    if (prodLastRow > 1) {
      prodLatest.getRange(2, 1, prodLastRow - 1, prodLatest.getLastColumn()).clearContent();
    }
    const dataRows = latestData.slice(1);
    prodLatest.getRange(2, 1, dataRows.length, dataRows[0].length).setValues(dataRows);
    Logger.log('DM_Latest replaced: ' + dataRows.length + ' rows.');
    transferred++;
  }

  Logger.log('');
  Logger.log('TRANSFER COMPLETE — ' + transferred + ' sheet(s) updated.');
  if (warnings.length > 0) Logger.log(warnings.length + ' warning(s) — review when convenient.');

  if (typeof jumps !== 'undefined' && jumps && jumps.length > 0) {
    logJumpsToMonitor_(production, jumps);
  }

  logAudit_(auditSheet, runTimestamp, 'SUCCESS',
    'Transferred ' + transferred + ' tab(s). ' + warnings.length + ' warnings.' +
    (warnings.length > 0 ? ' ' + warnings.join(' | ') : ''));
}
