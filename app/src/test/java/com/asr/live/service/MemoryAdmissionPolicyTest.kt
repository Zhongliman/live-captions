package com.asr.live.service

import org.junit.Assert.*
import org.junit.Test

class MemoryAdmissionPolicyTest {
    private val mib = 1024L * 1024
    private val gib = 1024L * mib

    @Test fun tenGiBAllocationIsAllowedWhenAndroidCanRetainItsReserve() {
        assertTrue(MemoryAdmissionPolicy.assess(11 * gib, 16 * gib, 256 * mib, false,
            10 * gib).allowed)
        assertFalse(MemoryAdmissionPolicy.assess(6 * gib, 16 * gib, 256 * mib, false,
            10 * gib).allowed)
    }

    @Test fun explicitAppAllowanceCanAdmitSevenBWhileAndroidReclaimsCachedProcesses() {
        val estimate = MemoryAdmissionPolicy.translatorEstimate(4_624_648_896L,
            largeModel = true, openCl = true)
        val assessment = MemoryAdmissionPolicy.assess(4 * gib, 15 * gib, 256 * mib, false,
            estimate.additionalBytes, currentAppBytes = 14 * gib / 10, appBudgetBytes = 10 * gib)
        assertTrue(estimate.additionalBytes > assessment.availableBytes)
        assertTrue(assessment.allowed)
        assertEquals(10 * gib, assessment.effectiveAppBudgetBytes)
        assertTrue(assessment.description.contains("app-budget mode"))
        assertTrue(assessment.description.contains("current app PSS"))
        assertTrue(assessment.description.contains("Android may reclaim cached processes"))
        assertTrue(assessment.description.contains("not a guaranteed peak"))
        // The default remains strict available-RAM admission.
        assertFalse(MemoryAdmissionPolicy.assess(4 * gib, 15 * gib, 256 * mib, false,
            estimate.additionalBytes).allowed)
    }

    @Test fun explicitAllowanceNeverBypassesCurrentAndroidMemoryPressure() {
        assertFalse(MemoryAdmissionPolicy.assess(4 * gib, 15 * gib, 256 * mib, true,
            5 * gib, currentAppBytes = gib, appBudgetBytes = 10 * gib).allowed)
        assertFalse(MemoryAdmissionPolicy.assess(512 * mib, 15 * gib, 256 * mib, false,
            5 * gib, currentAppBytes = gib, appBudgetBytes = 10 * gib).allowed)
    }

    @Test fun explicitAllowanceRespectsApplicationAndPhysicalHeadroomLimits() {
        assertFalse(MemoryAdmissionPolicy.assess(4 * gib, 15 * gib, 256 * mib, false,
            6 * gib, currentAppBytes = 5 * gib, appBudgetBytes = 10 * gib).allowed)
        val physicalLimit = MemoryAdmissionPolicy.assess(4 * gib, 6 * gib, 256 * mib, false,
            5 * gib, currentAppBytes = gib, appBudgetBytes = 10 * gib)
        assertFalse(physicalLimit.allowed)
        assertEquals(6 * gib - physicalLimit.reserveBytes, physicalLimit.effectiveAppBudgetBytes)
        assertFalse(MemoryAdmissionPolicy.assess(4 * gib, 15 * gib, 256 * mib, false,
            0, currentAppBytes = 11 * gib, appBudgetBytes = 10 * gib).allowed)
    }

    @Test fun explicitAllowanceAllowsExactBoundaryWithoutOverflow() {
        assertTrue(MemoryAdmissionPolicy.assess(gib, 15 * gib, 256 * mib, false,
            6 * gib, reserveBytes = gib, currentAppBytes = 4 * gib, appBudgetBytes = 10 * gib).allowed)
        assertFalse(MemoryAdmissionPolicy.assess(gib, 15 * gib, 256 * mib, false,
            6 * gib + 1, reserveBytes = gib, currentAppBytes = 4 * gib, appBudgetBytes = 10 * gib).allowed)
        assertFalse(MemoryAdmissionPolicy.assess(Long.MAX_VALUE, Long.MAX_VALUE, 256 * mib, false,
            Long.MAX_VALUE, currentAppBytes = Long.MAX_VALUE, appBudgetBytes = Long.MAX_VALUE).allowed)
    }

    @Test(expected = IllegalArgumentException::class) fun zeroAppAllowanceIsRejected() {
        MemoryAdmissionPolicy.assess(gib, 15 * gib, 256 * mib, false, gib, appBudgetBytes = 0)
    }

    @Test fun newTranslatorBudgetIncludesOnlyItsWeightsAndRuntime() {
        val estimate = MemoryAdmissionPolicy.translatorEstimate(4_624_648_896L, largeModel = true)
        assertEquals(4_624_648_896L, estimate.weightsBytes)
        assertTrue(estimate.contextBytes > 0)
        assertTrue(estimate.computeBytes > 0)
        assertEquals(0L, estimate.backendStagingBytes)
        assertEquals(estimate.weightsBytes + estimate.contextBytes + estimate.computeBytes, estimate.additionalBytes)
        // Already-resident ASR/QNN/OPUS are reflected by availableBytes and are never re-added.
        assertTrue(MemoryAdmissionPolicy.assess(7 * gib, 16 * gib, 256 * mib, false,
            estimate.additionalBytes).allowed)
    }

    @Test fun largerBatchesAndOpenClIncreaseTheAllocationBudget() {
        val baseline = MemoryAdmissionPolicy.translatorEstimate(gib, batch = 128, ubatch = 64)
        val larger = MemoryAdmissionPolicy.translatorEstimate(gib, batch = 512, ubatch = 256, openCl = true)
        assertEquals(baseline.weightsBytes, larger.weightsBytes)
        assertTrue(larger.computeBytes > baseline.computeBytes)
        assertTrue(larger.backendStagingBytes > 0)
        assertTrue(larger.additionalBytes > baseline.additionalBytes)
    }

    @Test fun headroomScalesAndRespectsAndroidThreshold() {
        assertEquals(512 * mib, MemoryAdmissionPolicy.systemReserve(4 * gib, 128 * mib))
        assertEquals(gib, MemoryAdmissionPolicy.systemReserve(16 * gib, 128 * mib))
        assertEquals(gib, MemoryAdmissionPolicy.systemReserve(64 * gib, 128 * mib))
        assertEquals(1536 * mib, MemoryAdmissionPolicy.systemReserve(16 * gib, 1280 * mib))
    }

    @Test fun lowMemoryAlwaysRejectsAndDescriptionExplainsTheBudget() {
        val assessment = MemoryAdmissionPolicy.assess(8 * gib, 16 * gib, 256 * mib, true, gib)
        assertFalse(assessment.allowed)
        assertTrue(assessment.description.contains("8192 MiB available"))
        assertTrue(assessment.description.contains("1024 MiB estimated additional allocation"))
        assertTrue(assessment.description.contains("Android low-memory=yes"))
    }

    @Test fun boundaryPreservesReserveAndOverflowCannotAdmitAHugeAllocation() {
        assertTrue(MemoryAdmissionPolicy.assess(2 * gib, 16 * gib, 256 * mib, false, gib).allowed)
        assertFalse(MemoryAdmissionPolicy.assess(2 * gib - 1, 16 * gib, 256 * mib, false, gib).allowed)
        val huge = MemoryAdmissionPolicy.translatorEstimate(Long.MAX_VALUE, largeModel = true, openCl = true)
        assertEquals(Long.MAX_VALUE, huge.additionalBytes)
        val assessment = MemoryAdmissionPolicy.assess(Long.MAX_VALUE, 16 * gib, 256 * mib, false, huge.additionalBytes)
        assertFalse(assessment.allowed)
        assertEquals(Long.MAX_VALUE, assessment.requiredBytes)
    }

    @Test(expected = IllegalArgumentException::class) fun missingWeightEstimateIsRejected() {
        MemoryAdmissionPolicy.translatorEstimate(0)
    }

    @Test(expected = IllegalArgumentException::class) fun negativeAdditionalAllocationIsRejected() {
        MemoryAdmissionPolicy.assess(gib, 16 * gib, 256 * mib, false, -1)
    }
}
