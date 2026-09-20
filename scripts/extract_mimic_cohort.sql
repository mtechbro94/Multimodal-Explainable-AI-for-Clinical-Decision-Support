-- ==============================================================================
-- Google BigQuery SQL Cohort Extraction for Multimodal MIMIC-IV + MIMIC-CXR
-- Paper: "Faithful by Design: A Cross-Modal Concept Bottleneck Framework for 
--         Trustworthy Multimodal Clinical Decision Support"
-- ==============================================================================
-- Instructions:
-- 1. Open Google BigQuery Console (https://console.cloud.google.com/bigquery)
-- 2. Ensure your Google account is linked to your credentialed PhysioNet profile.
-- 3. Run this query and click "Export" -> "Save to Google Cloud Storage" or "Download CSV".
-- ==============================================================================

WITH cohort_admissions AS (
    SELECT 
        adm.subject_id,
        adm.hadm_id,
        icu.stay_id,
        adm.admittime,
        adm.dischtime,
        adm.hospital_expire_flag,
        pat.anchor_age + (EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year) AS age,
        pat.gender,
        adm.race AS ethnicity,
        icu.intime,
        icu.outtime
    FROM `physionet-data.mimiciv_hosp.admissions` adm
    INNER JOIN `physionet-data.mimiciv_icu.icustays` icu
        ON adm.hadm_id = icu.hadm_id
    INNER JOIN `physionet-data.mimiciv_hosp.patients` pat
        ON adm.subject_id = pat.subject_id
    WHERE (pat.anchor_age + (EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year)) >= 18
),

-- First 24-hour Vitals aggregation
vitals_first_24h AS (
    SELECT 
        ce.stay_id,
        AVG(CASE WHEN ce.itemid IN (220045) THEN ce.valuenum END) AS heart_rate,
        AVG(CASE WHEN ce.itemid IN (220179, 220050) THEN ce.valuenum END) AS sbp,
        AVG(CASE WHEN ce.itemid IN (220180, 220051) THEN ce.valuenum END) AS dbp,
        AVG(CASE WHEN ce.itemid IN (220277) THEN ce.valuenum END) AS spo2,
        AVG(CASE WHEN ce.itemid IN (223762, 223761) THEN 
            CASE WHEN ce.itemid = 223761 THEN (ce.valuenum - 32) * 5/9 ELSE ce.valuenum END 
        END) AS temperature,
        AVG(CASE WHEN ce.itemid IN (220210, 224690) THEN ce.valuenum END) AS respiratory_rate
    FROM `physionet-data.mimiciv_icu.chartevents` ce
    INNER JOIN cohort_admissions ca ON ce.stay_id = ca.stay_id
    WHERE ce.charttime BETWEEN ca.intime AND DATETIME_ADD(ca.intime, INTERVAL 24 HOUR)
    GROUP BY ce.stay_id
),

-- First 24-hour Labs aggregation
labs_first_24h AS (
    SELECT 
        ca.stay_id,
        AVG(CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum END) AS wbc,
        AVG(CASE WHEN le.itemid IN (51222) THEN le.valuenum END) AS hemoglobin,
        AVG(CASE WHEN le.itemid IN (51265) THEN le.valuenum END) AS platelets,
        AVG(CASE WHEN le.itemid IN (50912) THEN le.valuenum END) AS creatinine,
        AVG(CASE WHEN le.itemid IN (50885) THEN le.valuenum END) AS bilirubin,
        AVG(CASE WHEN le.itemid IN (50813) THEN le.valuenum END) AS lactate,
        AVG(CASE WHEN le.itemid IN (50821) THEN le.valuenum END) AS pao2_fio2_ratio
    FROM `physionet-data.mimiciv_hosp.labevents` le
    INNER JOIN cohort_admissions ca ON le.hadm_id = ca.hadm_id
    WHERE le.charttime BETWEEN ca.intime AND DATETIME_ADD(ca.intime, INTERVAL 24 HOUR)
    GROUP BY ca.stay_id
),

-- Paired Frontal Chest Radiograph within 24h of ICU admission
matched_cxr AS (
    SELECT 
        ca.stay_id,
        meta.study_id,
        meta.dicom_id,
        meta.ViewPosition,
        meta.StudyDateTime,
        ROW_NUMBER() OVER (
            PARTITION BY ca.stay_id 
            ORDER BY ABS(DATETIME_DIFF(meta.StudyDateTime, ca.intime, MINUTE)) ASC
        ) as cxr_rank
    FROM `physionet-data.mimic_cxr.metadata` meta
    INNER JOIN cohort_admissions ca 
        ON meta.subject_id = ca.subject_id
    WHERE meta.ViewPosition IN ('AP', 'PA')
      AND meta.StudyDateTime BETWEEN DATETIME_SUB(ca.intime, INTERVAL 12 HOUR) 
                                 AND DATETIME_ADD(ca.intime, INTERVAL 24 HOUR)
)

-- Final Linked Multimodal Table
SELECT 
    ca.subject_id,
    ca.hadm_id,
    ca.stay_id,
    ca.hospital_expire_flag,
    ca.age,
    ca.gender,
    ca.ethnicity,
    -- Vitals
    COALESCE(v.heart_rate, 85.0) AS heart_rate,
    COALESCE(v.sbp, 120.0) AS sbp,
    COALESCE(v.dbp, 75.0) AS dbp,
    COALESCE(v.spo2, 96.0) AS spo2,
    COALESCE(v.temperature, 37.0) AS temperature,
    COALESCE(v.respiratory_rate, 18.0) AS respiratory_rate,
    -- Labs
    COALESCE(l.wbc, 10.0) AS wbc,
    COALESCE(l.hemoglobin, 12.0) AS hemoglobin,
    COALESCE(l.platelets, 220.0) AS platelets,
    COALESCE(l.creatinine, 1.1) AS creatinine,
    COALESCE(l.bilirubin, 1.2) AS bilirubin,
    COALESCE(l.lactate, 1.8) AS lactate,
    COALESCE(l.pao2_fio2_ratio, 280.0) AS pao2_fio2_ratio,
    -- Clinical Acuity Proxy Scores
    (CASE WHEN COALESCE(v.sbp, 120) < 90 THEN 1 ELSE 0 END +
     CASE WHEN COALESCE(v.respiratory_rate, 18) >= 22 THEN 1 ELSE 0 END +
     CASE WHEN COALESCE(l.creatinine, 1.1) >= 2.0 THEN 2 ELSE 0 END) AS sofa_score,
    (CASE WHEN ca.age >= 65 THEN 2 ELSE 0 END +
     CASE WHEN COALESCE(v.heart_rate, 85) >= 110 THEN 1 ELSE 0 END +
     CASE WHEN COALESCE(l.lactate, 1.8) >= 4.0 THEN 3 ELSE 0 END) AS apache_score,
    -- Imaging Linkage
    cxr.study_id,
    cxr.dicom_id,
    cxr.ViewPosition
FROM cohort_admissions ca
INNER JOIN matched_cxr cxr 
    ON ca.stay_id = cxr.stay_id AND cxr.cxr_rank = 1
LEFT JOIN vitals_first_24h v 
    ON ca.stay_id = v.stay_id
LEFT JOIN labs_first_24h l 
    ON ca.stay_id = l.stay_id;
