-- How many Exact / Partial / Irrelevant judgments do we have?
-- Tells us how skewed the labels are — important for training pair sampling.
SELECT
    label,
    COUNT(*) AS cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM `wands.labels`
GROUP BY label
ORDER BY cnt DESC;
