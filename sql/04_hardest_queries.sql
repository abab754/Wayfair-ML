-- Queries with the fewest Exact matches — these are our "hard" queries.
-- Core to the project: we want to improve results on exactly these.
SELECT
    q.query_id,
    q.query,
    q.query_class,
    COUNTIF(l.label = 'Exact') AS exact_count,
    COUNTIF(l.label = 'Partial') AS partial_count,
    COUNTIF(l.label = 'Irrelevant') AS irrelevant_count
FROM `wands.queries` q
JOIN `wands.labels` l ON q.query_id = l.query_id
GROUP BY q.query_id, q.query, q.query_class
ORDER BY exact_count ASC, partial_count ASC
LIMIT 30;
