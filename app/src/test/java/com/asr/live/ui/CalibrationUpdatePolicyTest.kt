package com.asr.live.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationUpdatePolicyTest {
    @Test fun benchmarkLockBlocksManualChangesButAllowsApplyingItsOwnWinner() {
        assertTrue(shouldRejectConfigUpdate(busy = false, benchmarkBusy = true,
            running = false, automaticCalibration = false))
        assertFalse(shouldRejectConfigUpdate(busy = false, benchmarkBusy = true,
            running = false, automaticCalibration = true))
    }

    @Test fun calibrationCannotChangeConfigurationWhileTheServiceIsRunningOrModelsAreBusy() {
        assertTrue(shouldRejectConfigUpdate(busy = true, benchmarkBusy = true,
            running = false, automaticCalibration = true))
        assertTrue(shouldRejectConfigUpdate(busy = false, benchmarkBusy = true,
            running = true, automaticCalibration = true))
    }
}
