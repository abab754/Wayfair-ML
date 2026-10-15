# Sprint 2 Ablation Table (val split, 72 queries)

## Overall

| Model | NDCG@10 | MRR | Recall@100 |
|-------|---------|-----|------------|
| BM25 (baseline) | 0.6098 | 0.7777 | 0.3066 |
| Dense zero-shot | 0.5943 | 0.7997 | 0.2735 |
| Dense fine-tuned | 0.2865 | 0.5323 | 0.1359 |
| **Hybrid RRF** | **0.6653** | **0.8374** | **0.3152** |

## NDCG@10 by Query Class

| Query Class | BM25 | Dense ZS | Dense FT | Hybrid RRF | n |
|-------------|------|----------|----------|------------|---|
| Accent Chairs | 0.4818 | 0.3611 | 0.2500 | **0.6348** | 2 |
| Accent Pillows | 1.0000 | 1.0000 | 0.1651 | 1.0000 | 1 |
| Accent Stools | 0.2935 | 0.0694 | **0.5483** | 0.4153 | 1 |
| Area Rugs | **0.5024** | 0.2532 | 0.2282 | 0.3904 | 4 |
| Bedding Sets | 0.7636 | 0.6669 | 0.3210 | **0.8469** | 2 |
| Beds | 0.6107 | 0.8100 | 0.0000 | **1.0000** | 1 |
| Bike And Sport Racks | 0.9241 | **0.9526** | 0.3208 | **0.9526** | 1 |
| Bookcases | 0.0000 | 0.3300 | **0.3779** | 0.3214 | 1 |
| Cabinet and Drawer Pulls | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| Candle Holders | 0.0000 | **0.0511** | 0.0000 | 0.0000 | 1 |
| Coffee & Cocktail Tables | 0.5965 | **1.0000** | **1.0000** | 0.8611 | 1 |
| Cookware Sets | 0.9653 | **1.0000** | 0.0000 | 0.9633 | 1 |
| Curtains & Drapes | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1 |
| Deck Boxes | 0.9216 | **1.0000** | 0.0347 | 0.7721 | 1 |
| Decorative Objects | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 |
| Desks | 0.0000 | 0.1797 | 0.1052 | **0.1799** | 1 |
| Dining Chairs | 1.0000 | 0.6848 | 0.0000 | 1.0000 | 1 |
| Dining Tables | 0.5122 | 0.4844 | 0.1086 | **0.5615** | 2 |
| Dressers & Chests | 0.7675 | **0.8537** | 0.7111 | 0.8510 | 3 |
| End Tables | 0.6643 | **0.7160** | 0.0000 | 0.7151 | 3 |
| Floor & Wall Tile | 0.7652 | 0.7678 | 0.3211 | **0.7972** | 2 |
| Furniture Cushions | **1.0000** | 0.7894 | 0.3480 | 0.9653 | 1 |
| Hampers & Baskets | **0.9315** | 0.5206 | 0.3341 | 0.8379 | 1 |
| Kids Desks | 0.3527 | 0.2844 | **0.5000** | 0.2816 | 1 |
| Lighting Shades | 0.6263 | **0.7554** | 0.0685 | 0.4842 | 1 |
| Log Storage | 0.2918 | 0.4261 | 0.1859 | **0.4335** | 1 |
| Office Chairs | 0.4330 | 0.4925 | 0.4833 | **0.7917** | 1 |
| Office Storage Cabinets | 0.5279 | 0.6034 | 0.5512 | **0.7739** | 2 |
| Outdoor Conversation Sets | 0.1582 | 0.2276 | 0.1171 | **0.2429** | 3 |
| Patio Bar Stools | 0.1086 | 0.0000 | **0.3637** | 0.1090 | 1 |
| Pendant Lights | **1.0000** | 0.4734 | 0.0000 | 0.9149 | 1 |
| Plant & Telephone Tables | 1.0000 | 1.0000 | 0.9052 | 1.0000 | 1 |
| Plates & Saucers | 0.9052 | **1.0000** | 0.3341 | **1.0000** | 1 |
| Range Hoods | 0.9322 | 0.9207 | 0.5000 | **0.9682** | 1 |
| Recliners | 0.5318 | 0.4572 | 0.2745 | **0.5426** | 2 |
| Sectionals | 0.8611 | **1.0000** | **1.0000** | **1.0000** | 1 |
| Sheets And Sheet Sets | **1.0000** | 0.9266 | 0.4690 | **1.0000** | 1 |
| Shoe Storage | 0.5000 | 0.5000 | 0.0000 | **0.5367** | 1 |
| Shower & Tub Doors | 0.6032 | **0.8900** | 0.2408 | 0.8611 | 1 |
| Shower Heads | 1.0000 | 1.0000 | 0.3747 | 1.0000 | 1 |
| Sofas | **0.1934** | 0.0000 | 0.0000 | 0.0000 | 1 |
| String Lights | 1.0000 | 0.9841 | 0.0469 | 1.0000 | 1 |
| TV Stands & Entertainment Centers | 0.4829 | **0.7359** | 0.6890 | **0.7359** | 1 |
| Unknown | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| Vanities | 0.0000 | **0.4537** | 0.2470 | 0.2614 | 1 |
| Vanity Lighting | 1.0000 | 1.0000 | 0.5584 | 1.0000 | 1 |
| Wall & Accent Mirrors | 0.7941 | 0.4267 | 0.0444 | **0.7954** | 2 |
| Wall Art | 0.5917 | 0.5345 | 0.1810 | **0.6168** | 6 |
| Wall Clocks | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 |
| Wall Decor | 0.2852 | 0.4110 | 0.1988 | **0.6576** | 1 |
| Wall Plates | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1 |

## Biggest Hybrid RRF improvements over BM25

| Query Class | BM25 | Hybrid | Delta |
|-------------|------|--------|-------|
| Beds | 0.6107 | 1.0000 | +0.3893 |
| Wall Decor | 0.2852 | 0.6576 | +0.3724 |
| Office Chairs | 0.4330 | 0.7917 | +0.3588 |
| Bookcases | 0.0000 | 0.3214 | +0.3214 |
| Coffee & Cocktail Tables | 0.5965 | 0.8611 | +0.2646 |
| Vanities | 0.0000 | 0.2614 | +0.2614 |
| Shower & Tub Doors | 0.6032 | 0.8611 | +0.2579 |
| TV Stands & Entertainment Centers | 0.4829 | 0.7359 | +0.2529 |
| Office Storage Cabinets | 0.5279 | 0.7739 | +0.2459 |
| Desks | 0.0000 | 0.1799 | +0.1799 |

## Regressions (Hybrid worse than BM25)

| Query Class | BM25 | Hybrid | Delta |
|-------------|------|--------|-------|
| Sofas | 0.1934 | 0.0000 | -0.1934 |
| Deck Boxes | 0.9216 | 0.7721 | -0.1495 |
| Lighting Shades | 0.6263 | 0.4842 | -0.1421 |
| Area Rugs | 0.5024 | 0.3904 | -0.1120 |
| Hampers & Baskets | 0.9315 | 0.8379 | -0.0936 |
