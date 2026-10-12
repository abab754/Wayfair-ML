-- Rating and review count distribution by product class.
-- These become features in the LTR reranker — need to understand their range.
SELECT
    product_class,
    COUNT(*) AS num_products,
    ROUND(AVG(average_rating), 2) AS avg_rating,
    ROUND(AVG(review_count), 1) AS avg_reviews,
    MAX(review_count) AS max_reviews
FROM `wands.products`
GROUP BY product_class
ORDER BY num_products DESC
LIMIT 20;
