-- Top 20 categories by product count.
-- Sparse categories may hurt retrieval — good to know where coverage is thin.
SELECT
    product_class,
    COUNT(*) AS num_products
FROM `wands.products`
GROUP BY product_class
ORDER BY num_products DESC
LIMIT 20;
