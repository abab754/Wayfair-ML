-- How many queries per query class, and avg labeled products per query?
-- We slice eval metrics by query class, so need to know the distribution.
SELECT
    q.query_class,
    COUNT(DISTINCT q.query_id) AS num_queries,
    COUNT(l.id) AS num_labels,
    ROUND(COUNT(l.id) / COUNT(DISTINCT q.query_id), 1) AS avg_labels_per_query
FROM `wands.queries` q
LEFT JOIN `wands.labels` l ON q.query_id = l.query_id
GROUP BY q.query_class
ORDER BY num_queries DESC;
