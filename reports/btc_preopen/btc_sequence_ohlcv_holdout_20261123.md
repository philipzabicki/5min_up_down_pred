# E0051 - frozen holdout evaluation of exact-sequence models A/B

Holdout: [2026-07-01T00:00:00+00:00, 2026-10-07T00:00:00+00:00) UTC; generator `gen_0630`; models, temperatures, thresholds frozen at 2026-11-23T00:14:31.654044+00:00. Status: EVALUATED_ONCE.
Exposure: the historical holdout period has been inspected by earlier studies; this is not an unseen test. Results from 2026-07-01 are used only to evaluate this frozen experiment; TWAP60 (from 2026-08-14) is not covered by training labels.
Feature readiness: nominal T-60s clock; candle receipt NOT_PROVEN (diagnostic).
Official rows read: 79680 (starts < 2026-10-07T00:00:00+00:00); development part identical to frozen export: True.

## Support

| n | calendar windows | complete | label available by end | with features |
|---|---|---|---|---|
| 1 | 28224 | 28221 | 28220 | 28220 |
| 2 | 28223 | 28217 | 28216 | 28216 |
| 3 | 28222 | 28213 | 28212 | 28212 |
| 4 | 28221 | 28209 | 28208 | 28208 |
| 5 | 28220 | 28205 | 28204 | 28204 |

## Metrics by settlement-rule regime

| n | regime | model | N | log loss | Brier | hit |
|---|---|---|---|---|---|---|
| 1 | all | global_benchmark | 28220 | 0.69317 | 0.50002 | 0.4997 |
| 1 | single_price | global_benchmark | 10654 | 0.69321 | 0.50006 | 0.4969 |
| 1 | twap30 | global_benchmark | 2016 | 0.69310 | 0.49995 | 0.5055 |
| 1 | twap60 | global_benchmark | 15550 | 0.69316 | 0.50001 | 0.5009 |
| 1 | all | conditional_benchmark | 28220 | 0.69339 | 0.50024 | 0.5004 |
| 1 | single_price | conditional_benchmark | 10654 | 0.69331 | 0.50016 | 0.5071 |
| 1 | twap30 | conditional_benchmark | 2016 | 0.69338 | 0.50023 | 0.5010 |
| 1 | twap60 | conditional_benchmark | 15550 | 0.69344 | 0.50029 | 0.4956 |
| 1 | all | theory_uniform | 28220 | 0.69315 | 0.50000 | 0.5003 |
| 1 | single_price | theory_uniform | 10654 | 0.69315 | 0.50000 | 0.5031 |
| 1 | twap30 | theory_uniform | 2016 | 0.69315 | 0.50000 | 0.4945 |
| 1 | twap60 | theory_uniform | 15550 | 0.69315 | 0.50000 | 0.4991 |
| 1 | all | A_raw | 28220 | 0.69261 | 0.49947 | 0.5163 |
| 1 | single_price | A_raw | 10654 | 0.69282 | 0.49967 | 0.5090 |
| 1 | twap30 | A_raw | 2016 | 0.69101 | 0.49787 | 0.5268 |
| 1 | twap60 | A_raw | 15550 | 0.69268 | 0.49953 | 0.5199 |
| 1 | all | A_cal | 28220 | 0.69247 | 0.49933 | 0.5163 |
| 1 | single_price | A_cal | 10654 | 0.69262 | 0.49947 | 0.5090 |
| 1 | twap30 | A_cal | 2016 | 0.69131 | 0.49816 | 0.5268 |
| 1 | twap60 | A_cal | 15550 | 0.69253 | 0.49938 | 0.5199 |
| 1 | all | B_raw | 28220 | 0.69250 | 0.49936 | 0.5178 |
| 1 | single_price | B_raw | 10654 | 0.69334 | 0.50019 | 0.5149 |
| 1 | twap30 | B_raw | 2016 | 0.69120 | 0.49806 | 0.5253 |
| 1 | twap60 | B_raw | 15550 | 0.69210 | 0.49895 | 0.5187 |
| 1 | all | B_cal | 28220 | 0.69244 | 0.49930 | 0.5178 |
| 1 | single_price | B_cal | 10654 | 0.69318 | 0.50003 | 0.5149 |
| 1 | twap30 | B_cal | 2016 | 0.69132 | 0.49818 | 0.5253 |
| 1 | twap60 | B_cal | 15550 | 0.69209 | 0.49894 | 0.5187 |
| 2 | all | global_benchmark | 28216 | 1.38635 | 0.75003 | 0.2519 |
| 2 | mixed | global_benchmark | 2 | 1.38000 | 0.74691 | 0.0000 |
| 2 | single_price | global_benchmark | 10651 | 1.38622 | 0.74996 | 0.2551 |
| 2 | twap30 | global_benchmark | 2015 | 1.38704 | 0.75037 | 0.2392 |
| 2 | twap60 | global_benchmark | 15548 | 1.38636 | 0.75003 | 0.2514 |
| 2 | all | conditional_benchmark | 28216 | 1.38666 | 0.75018 | 0.2510 |
| 2 | mixed | conditional_benchmark | 2 | 1.41405 | 0.76375 | 0.0000 |
| 2 | single_price | conditional_benchmark | 10651 | 1.38645 | 0.75007 | 0.2587 |
| 2 | twap30 | conditional_benchmark | 2015 | 1.38660 | 0.75015 | 0.2581 |
| 2 | twap60 | conditional_benchmark | 15548 | 1.38682 | 0.75026 | 0.2449 |
| 2 | all | theory_uniform | 28216 | 1.38629 | 0.75000 | 0.2484 |
| 2 | mixed | theory_uniform | 2 | 1.38629 | 0.75000 | 0.0000 |
| 2 | single_price | theory_uniform | 10651 | 1.38629 | 0.75000 | 0.2481 |
| 2 | twap30 | theory_uniform | 2015 | 1.38629 | 0.75000 | 0.2556 |
| 2 | twap60 | theory_uniform | 15548 | 1.38629 | 0.75000 | 0.2477 |
| 2 | all | A_raw | 28216 | 1.38529 | 0.74950 | 0.2609 |
| 2 | mixed | A_raw | 2 | 1.37620 | 0.74509 | 0.5000 |
| 2 | single_price | A_raw | 10651 | 1.38480 | 0.74926 | 0.2638 |
| 2 | twap30 | A_raw | 2015 | 1.38304 | 0.74836 | 0.2878 |
| 2 | twap60 | A_raw | 15548 | 1.38591 | 0.74982 | 0.2553 |
| 2 | all | A_cal | 28216 | 1.38511 | 0.74942 | 0.2609 |
| 2 | mixed | A_cal | 2 | 1.37895 | 0.74639 | 0.5000 |
| 2 | single_price | A_cal | 10651 | 1.38477 | 0.74924 | 0.2638 |
| 2 | twap30 | A_cal | 2015 | 1.38367 | 0.74868 | 0.2878 |
| 2 | twap60 | A_cal | 15548 | 1.38553 | 0.74963 | 0.2553 |
| 2 | all | B_raw | 28216 | 1.38492 | 0.74932 | 0.2658 |
| 2 | mixed | B_raw | 2 | 1.34546 | 0.72969 | 0.5000 |
| 2 | single_price | B_raw | 10651 | 1.38575 | 0.74974 | 0.2574 |
| 2 | twap30 | B_raw | 2015 | 1.38315 | 0.74843 | 0.2918 |
| 2 | twap60 | B_raw | 15548 | 1.38458 | 0.74915 | 0.2681 |
| 2 | all | B_cal | 28216 | 1.38494 | 0.74933 | 0.2658 |
| 2 | mixed | B_cal | 2 | 1.34730 | 0.73060 | 0.5000 |
| 2 | single_price | B_cal | 10651 | 1.38573 | 0.74973 | 0.2574 |
| 2 | twap30 | B_cal | 2015 | 1.38326 | 0.74848 | 0.2918 |
| 2 | twap60 | B_cal | 15548 | 1.38462 | 0.74917 | 0.2681 |
| 3 | all | global_benchmark | 28212 | 2.07958 | 0.87503 | 0.1243 |
| 3 | mixed | global_benchmark | 4 | 2.08166 | 0.87556 | 0.2500 |
| 3 | single_price | global_benchmark | 10648 | 2.07943 | 0.87500 | 0.1297 |
| 3 | twap30 | global_benchmark | 2014 | 2.08081 | 0.87534 | 0.1167 |
| 3 | twap60 | global_benchmark | 15546 | 2.07952 | 0.87502 | 0.1216 |
| 3 | all | conditional_benchmark | 28212 | 2.08022 | 0.87519 | 0.1248 |
| 3 | mixed | conditional_benchmark | 4 | 2.13110 | 0.88768 | 0.0000 |
| 3 | single_price | conditional_benchmark | 10648 | 2.08046 | 0.87525 | 0.1249 |
| 3 | twap30 | conditional_benchmark | 2014 | 2.07998 | 0.87512 | 0.1331 |
| 3 | twap60 | conditional_benchmark | 15546 | 2.08007 | 0.87516 | 0.1237 |
| 3 | all | theory_uniform | 28212 | 2.07944 | 0.87500 | 0.1209 |
| 3 | mixed | theory_uniform | 4 | 2.07944 | 0.87500 | 0.0000 |
| 3 | single_price | theory_uniform | 10648 | 2.07944 | 0.87500 | 0.1227 |
| 3 | twap30 | theory_uniform | 2014 | 2.07944 | 0.87500 | 0.1331 |
| 3 | twap60 | theory_uniform | 15546 | 2.07944 | 0.87500 | 0.1182 |
| 3 | all | A_raw | 28212 | 2.07808 | 0.87468 | 0.1313 |
| 3 | mixed | A_raw | 4 | 2.04610 | 0.86683 | 0.0000 |
| 3 | single_price | A_raw | 10648 | 2.07776 | 0.87459 | 0.1345 |
| 3 | twap30 | A_raw | 2014 | 2.07575 | 0.87406 | 0.1569 |
| 3 | twap60 | A_raw | 15546 | 2.07862 | 0.87482 | 0.1258 |
| 3 | all | A_cal | 28212 | 2.07823 | 0.87470 | 0.1313 |
| 3 | mixed | A_cal | 4 | 2.06019 | 0.87024 | 0.0000 |
| 3 | single_price | A_cal | 10648 | 2.07806 | 0.87466 | 0.1345 |
| 3 | twap30 | A_cal | 2014 | 2.07697 | 0.87438 | 0.1569 |
| 3 | twap60 | A_cal | 15546 | 2.07851 | 0.87478 | 0.1258 |
| 3 | all | B_raw | 28212 | 2.07772 | 0.87458 | 0.1282 |
| 3 | mixed | B_raw | 4 | 2.04570 | 0.86664 | 0.2500 |
| 3 | single_price | B_raw | 10648 | 2.07887 | 0.87486 | 0.1254 |
| 3 | twap30 | B_raw | 2014 | 2.07601 | 0.87414 | 0.1390 |
| 3 | twap60 | B_raw | 15546 | 2.07716 | 0.87445 | 0.1287 |
| 3 | all | B_cal | 28212 | 2.07782 | 0.87460 | 0.1282 |
| 3 | mixed | B_cal | 4 | 2.05007 | 0.86772 | 0.2500 |
| 3 | single_price | B_cal | 10648 | 2.07881 | 0.87485 | 0.1254 |
| 3 | twap30 | B_cal | 2014 | 2.07636 | 0.87423 | 0.1390 |
| 3 | twap60 | B_cal | 15546 | 2.07733 | 0.87449 | 0.1287 |
| 4 | all | global_benchmark | 28208 | 2.77287 | 0.93754 | 0.0629 |
| 4 | mixed | global_benchmark | 6 | 2.78962 | 0.93959 | 0.0000 |
| 4 | single_price | global_benchmark | 10645 | 2.77245 | 0.93748 | 0.0693 |
| 4 | twap30 | global_benchmark | 2013 | 2.77469 | 0.93775 | 0.0546 |
| 4 | twap60 | global_benchmark | 15544 | 2.77292 | 0.93754 | 0.0596 |
| 4 | all | conditional_benchmark | 28208 | 2.77399 | 0.93767 | 0.0621 |
| 4 | mixed | conditional_benchmark | 6 | 2.80936 | 0.94224 | 0.0000 |
| 4 | single_price | conditional_benchmark | 10645 | 2.77479 | 0.93777 | 0.0635 |
| 4 | twap30 | conditional_benchmark | 2013 | 2.77532 | 0.93783 | 0.0606 |
| 4 | twap60 | conditional_benchmark | 15544 | 2.77325 | 0.93758 | 0.0613 |
| 4 | all | theory_uniform | 28208 | 2.77259 | 0.93750 | 0.0584 |
| 4 | mixed | theory_uniform | 6 | 2.77259 | 0.93750 | 0.0000 |
| 4 | single_price | theory_uniform | 10645 | 2.77259 | 0.93750 | 0.0583 |
| 4 | twap30 | theory_uniform | 2013 | 2.77259 | 0.93750 | 0.0710 |
| 4 | twap60 | theory_uniform | 15544 | 2.77259 | 0.93750 | 0.0568 |
| 4 | all | A_raw | 28208 | 2.77151 | 0.93738 | 0.0649 |
| 4 | mixed | A_raw | 6 | 2.73477 | 0.93300 | 0.0000 |
| 4 | single_price | A_raw | 10645 | 2.77127 | 0.93734 | 0.0655 |
| 4 | twap30 | A_raw | 2013 | 2.76991 | 0.93717 | 0.0715 |
| 4 | twap60 | A_raw | 15544 | 2.77189 | 0.93743 | 0.0636 |
| 4 | all | A_cal | 28208 | 2.77144 | 0.93736 | 0.0649 |
| 4 | mixed | A_cal | 6 | 2.75439 | 0.93527 | 0.0000 |
| 4 | single_price | A_cal | 10645 | 2.77133 | 0.93734 | 0.0655 |
| 4 | twap30 | A_cal | 2013 | 2.77079 | 0.93728 | 0.0715 |
| 4 | twap60 | A_cal | 15544 | 2.77160 | 0.93738 | 0.0636 |
| 4 | all | B_raw | 28208 | 2.77068 | 0.93727 | 0.0678 |
| 4 | mixed | B_raw | 6 | 2.72438 | 0.93150 | 0.1667 |
| 4 | single_price | B_raw | 10645 | 2.77197 | 0.93742 | 0.0698 |
| 4 | twap30 | B_raw | 2013 | 2.76824 | 0.93695 | 0.0705 |
| 4 | twap60 | B_raw | 15544 | 2.77013 | 0.93721 | 0.0661 |
| 4 | all | B_cal | 28208 | 2.77081 | 0.93728 | 0.0678 |
| 4 | mixed | B_cal | 6 | 2.73525 | 0.93285 | 0.1667 |
| 4 | single_price | B_cal | 10645 | 2.77179 | 0.93740 | 0.0698 |
| 4 | twap30 | B_cal | 2013 | 2.76899 | 0.93705 | 0.0705 |
| 4 | twap60 | B_cal | 15544 | 2.77039 | 0.93723 | 0.0661 |
| 5 | all | global_benchmark | 28204 | 3.46611 | 0.96877 | 0.0326 |
| 5 | mixed | global_benchmark | 8 | 3.48851 | 0.97013 | 0.0000 |
| 5 | single_price | global_benchmark | 10642 | 3.46551 | 0.96874 | 0.0350 |
| 5 | twap30 | global_benchmark | 2012 | 3.46827 | 0.96890 | 0.0318 |
| 5 | twap60 | global_benchmark | 15542 | 3.46624 | 0.96878 | 0.0310 |
| 5 | all | conditional_benchmark | 28204 | 3.46858 | 0.96893 | 0.0312 |
| 5 | mixed | conditional_benchmark | 8 | 3.49332 | 0.97048 | 0.1250 |
| 5 | single_price | conditional_benchmark | 10642 | 3.46982 | 0.96900 | 0.0306 |
| 5 | twap30 | conditional_benchmark | 2012 | 3.47033 | 0.96903 | 0.0283 |
| 5 | twap60 | conditional_benchmark | 15542 | 3.46749 | 0.96886 | 0.0320 |
| 5 | all | theory_uniform | 28204 | 3.46574 | 0.96875 | 0.0279 |
| 5 | mixed | theory_uniform | 8 | 3.46574 | 0.96875 | 0.0000 |
| 5 | single_price | theory_uniform | 10642 | 3.46574 | 0.96875 | 0.0268 |
| 5 | twap30 | theory_uniform | 2012 | 3.46574 | 0.96875 | 0.0388 |
| 5 | twap60 | theory_uniform | 15542 | 3.46574 | 0.96875 | 0.0272 |
| 5 | all | A_raw | 28204 | 3.46508 | 0.96871 | 0.0313 |
| 5 | mixed | A_raw | 8 | 3.45396 | 0.96816 | 0.0000 |
| 5 | single_price | A_raw | 10642 | 3.46438 | 0.96867 | 0.0331 |
| 5 | twap30 | A_raw | 2012 | 3.46293 | 0.96857 | 0.0393 |
| 5 | twap60 | A_raw | 15542 | 3.46584 | 0.96876 | 0.0291 |
| 5 | all | A_cal | 28204 | 3.46486 | 0.96870 | 0.0313 |
| 5 | mixed | A_cal | 8 | 3.46001 | 0.96842 | 0.0000 |
| 5 | single_price | A_cal | 10642 | 3.46456 | 0.96868 | 0.0331 |
| 5 | twap30 | A_cal | 2012 | 3.46397 | 0.96864 | 0.0393 |
| 5 | twap60 | A_cal | 15542 | 3.46518 | 0.96872 | 0.0291 |
| 5 | all | B_raw | 28204 | 3.46391 | 0.96864 | 0.0339 |
| 5 | mixed | B_raw | 8 | 3.41597 | 0.96571 | 0.1250 |
| 5 | single_price | B_raw | 10642 | 3.46520 | 0.96872 | 0.0345 |
| 5 | twap30 | B_raw | 2012 | 3.46218 | 0.96852 | 0.0348 |
| 5 | twap60 | B_raw | 15542 | 3.46329 | 0.96860 | 0.0334 |
| 5 | all | B_cal | 28204 | 3.46404 | 0.96865 | 0.0339 |
| 5 | mixed | B_cal | 8 | 3.43298 | 0.96673 | 0.1250 |
| 5 | single_price | B_cal | 10642 | 3.46486 | 0.96870 | 0.0345 |
| 5 | twap30 | B_cal | 2012 | 3.46299 | 0.96858 | 0.0348 |
| 5 | twap60 | B_cal | 15542 | 3.46363 | 0.96862 | 0.0334 |

## Paired block bootstrap (second minus first)

| n | regime | comparison | metric | mean | 95% interval | days | blocks | N |
|---|---|---|---|---|---|---|---|---|
| 1 | all | B_cal-A_cal | log_loss | -0.00003 | [-0.00040, +0.00030] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-A_cal | log_loss | +0.00056 | [+0.00017, +0.00096] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-A_cal | log_loss | +0.00001 | [-0.00100, +0.00088] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-A_cal | log_loss | -0.00044 | [-0.00090, -0.00000] | 54 | 18 | 15550 |
| 1 | all | B_cal-A_cal | brier | -0.00003 | [-0.00040, +0.00030] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-A_cal | brier | +0.00056 | [+0.00016, +0.00095] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-A_cal | brier | +0.00001 | [-0.00100, +0.00088] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-A_cal | brier | -0.00044 | [-0.00090, -0.00000] | 54 | 18 | 15550 |
| 1 | all | B_cal-A_cal | hit | +0.00145 | [-0.00361, +0.00656] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-A_cal | hit | +0.00591 | [-0.00084, +0.01277] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-A_cal | hit | -0.00149 | [-0.02183, +0.01885] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-A_cal | hit | -0.00122 | [-0.00830, +0.00662] | 54 | 18 | 15550 |
| 1 | all | A_cal-global_benchmark | log_loss | -0.00070 | [-0.00111, -0.00026] | 98 | 33 | 28220 |
| 1 | single_price | A_cal-global_benchmark | log_loss | -0.00059 | [-0.00135, +0.00012] | 37 | 13 | 10654 |
| 1 | twap30 | A_cal-global_benchmark | log_loss | -0.00179 | [-0.00253, -0.00098] | 7 | 3 | 2016 |
| 1 | twap60 | A_cal-global_benchmark | log_loss | -0.00063 | [-0.00119, -0.00004] | 54 | 18 | 15550 |
| 1 | all | A_cal-global_benchmark | brier | -0.00070 | [-0.00111, -0.00026] | 98 | 33 | 28220 |
| 1 | single_price | A_cal-global_benchmark | brier | -0.00059 | [-0.00135, +0.00012] | 37 | 13 | 10654 |
| 1 | twap30 | A_cal-global_benchmark | brier | -0.00179 | [-0.00253, -0.00098] | 7 | 3 | 2016 |
| 1 | twap60 | A_cal-global_benchmark | brier | -0.00063 | [-0.00119, -0.00004] | 54 | 18 | 15550 |
| 1 | all | A_cal-global_benchmark | hit | +0.01658 | [+0.00974, +0.02342] | 98 | 33 | 28220 |
| 1 | single_price | A_cal-global_benchmark | hit | +0.01211 | [+0.00207, +0.02206] | 37 | 13 | 10654 |
| 1 | twap30 | A_cal-global_benchmark | hit | +0.02133 | [+0.00050, +0.04365] | 7 | 3 | 2016 |
| 1 | twap60 | A_cal-global_benchmark | hit | +0.01904 | [+0.00855, +0.02842] | 54 | 18 | 15550 |
| 1 | all | B_cal-global_benchmark | log_loss | -0.00073 | [-0.00126, -0.00020] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-global_benchmark | log_loss | -0.00003 | [-0.00086, +0.00077] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-global_benchmark | log_loss | -0.00178 | [-0.00263, -0.00109] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-global_benchmark | log_loss | -0.00107 | [-0.00172, -0.00038] | 54 | 18 | 15550 |
| 1 | all | B_cal-global_benchmark | brier | -0.00073 | [-0.00126, -0.00020] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-global_benchmark | brier | -0.00003 | [-0.00086, +0.00076] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-global_benchmark | brier | -0.00178 | [-0.00263, -0.00109] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-global_benchmark | brier | -0.00107 | [-0.00172, -0.00038] | 54 | 18 | 15550 |
| 1 | all | B_cal-global_benchmark | hit | +0.01804 | [+0.01070, +0.02481] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-global_benchmark | hit | +0.01802 | [+0.00779, +0.02891] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-global_benchmark | hit | +0.01984 | [+0.00794, +0.03175] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-global_benchmark | hit | +0.01781 | [+0.00695, +0.02765] | 54 | 18 | 15550 |
| 1 | all | A_cal-conditional_benchmark | log_loss | -0.00091 | [-0.00140, -0.00042] | 98 | 33 | 28220 |
| 1 | single_price | A_cal-conditional_benchmark | log_loss | -0.00069 | [-0.00148, +0.00003] | 37 | 13 | 10654 |
| 1 | twap30 | A_cal-conditional_benchmark | log_loss | -0.00207 | [-0.00349, -0.00048] | 7 | 3 | 2016 |
| 1 | twap60 | A_cal-conditional_benchmark | log_loss | -0.00091 | [-0.00161, -0.00019] | 54 | 18 | 15550 |
| 1 | all | A_cal-conditional_benchmark | brier | -0.00091 | [-0.00140, -0.00042] | 98 | 33 | 28220 |
| 1 | single_price | A_cal-conditional_benchmark | brier | -0.00069 | [-0.00148, +0.00003] | 37 | 13 | 10654 |
| 1 | twap30 | A_cal-conditional_benchmark | brier | -0.00207 | [-0.00349, -0.00048] | 7 | 3 | 2016 |
| 1 | twap60 | A_cal-conditional_benchmark | brier | -0.00091 | [-0.00161, -0.00020] | 54 | 18 | 15550 |
| 1 | all | A_cal-conditional_benchmark | hit | +0.01595 | [+0.00723, +0.02406] | 98 | 33 | 28220 |
| 1 | single_price | A_cal-conditional_benchmark | hit | +0.00188 | [-0.01267, +0.01745] | 37 | 13 | 10654 |
| 1 | twap30 | A_cal-conditional_benchmark | hit | +0.02579 | [-0.00744, +0.05704] | 7 | 3 | 2016 |
| 1 | twap60 | A_cal-conditional_benchmark | hit | +0.02431 | [+0.01556, +0.03241] | 54 | 18 | 15550 |
| 1 | all | B_cal-conditional_benchmark | log_loss | -0.00094 | [-0.00148, -0.00039] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-conditional_benchmark | log_loss | -0.00013 | [-0.00093, +0.00065] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-conditional_benchmark | log_loss | -0.00206 | [-0.00280, -0.00134] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-conditional_benchmark | log_loss | -0.00135 | [-0.00206, -0.00059] | 54 | 18 | 15550 |
| 1 | all | B_cal-conditional_benchmark | brier | -0.00094 | [-0.00148, -0.00039] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-conditional_benchmark | brier | -0.00013 | [-0.00093, +0.00064] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-conditional_benchmark | brier | -0.00206 | [-0.00280, -0.00134] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-conditional_benchmark | brier | -0.00135 | [-0.00206, -0.00059] | 54 | 18 | 15550 |
| 1 | all | B_cal-conditional_benchmark | hit | +0.01740 | [+0.00900, +0.02527] | 98 | 33 | 28220 |
| 1 | single_price | B_cal-conditional_benchmark | hit | +0.00779 | [-0.00704, +0.02263] | 37 | 13 | 10654 |
| 1 | twap30 | B_cal-conditional_benchmark | hit | +0.02431 | [+0.00149, +0.04415] | 7 | 3 | 2016 |
| 1 | twap60 | B_cal-conditional_benchmark | hit | +0.02309 | [+0.01389, +0.03209] | 54 | 18 | 15550 |
| 2 | all | B_cal-A_cal | log_loss | -0.00017 | [-0.00075, +0.00036] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-A_cal | log_loss | -0.03164 | [-0.03164, -0.03164] | 2 | 1 | 2 |
| 2 | single_price | B_cal-A_cal | log_loss | +0.00096 | [+0.00024, +0.00169] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-A_cal | log_loss | -0.00041 | [-0.00133, +0.00042] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-A_cal | log_loss | -0.00091 | [-0.00157, -0.00030] | 54 | 18 | 15548 |
| 2 | all | B_cal-A_cal | brier | -0.00009 | [-0.00037, +0.00018] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-A_cal | brier | -0.01579 | [-0.01579, -0.01579] | 2 | 1 | 2 |
| 2 | single_price | B_cal-A_cal | brier | +0.00049 | [+0.00012, +0.00085] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-A_cal | brier | -0.00020 | [-0.00066, +0.00020] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-A_cal | brier | -0.00046 | [-0.00079, -0.00015] | 54 | 18 | 15548 |
| 2 | all | B_cal-A_cal | hit | +0.00489 | [-0.00113, +0.01127] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-A_cal | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 2 |
| 2 | single_price | B_cal-A_cal | hit | -0.00638 | [-0.01342, +0.00084] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-A_cal | hit | +0.00397 | [-0.01192, +0.02083] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-A_cal | hit | +0.01273 | [+0.00521, +0.02053] | 54 | 18 | 15548 |
| 2 | all | A_cal-global_benchmark | log_loss | -0.00124 | [-0.00186, -0.00059] | 98 | 33 | 28216 |
| 2 | mixed | A_cal-global_benchmark | log_loss | -0.00106 | [-0.00106, -0.00106] | 2 | 1 | 2 |
| 2 | single_price | A_cal-global_benchmark | log_loss | -0.00144 | [-0.00248, -0.00036] | 37 | 13 | 10651 |
| 2 | twap30 | A_cal-global_benchmark | log_loss | -0.00337 | [-0.00434, -0.00251] | 7 | 3 | 2015 |
| 2 | twap60 | A_cal-global_benchmark | log_loss | -0.00083 | [-0.00164, +0.00006] | 54 | 18 | 15548 |
| 2 | all | A_cal-global_benchmark | brier | -0.00061 | [-0.00092, -0.00029] | 98 | 33 | 28216 |
| 2 | mixed | A_cal-global_benchmark | brier | -0.00052 | [-0.00052, -0.00052] | 2 | 1 | 2 |
| 2 | single_price | A_cal-global_benchmark | brier | -0.00072 | [-0.00123, -0.00017] | 37 | 13 | 10651 |
| 2 | twap30 | A_cal-global_benchmark | brier | -0.00169 | [-0.00218, -0.00124] | 7 | 3 | 2015 |
| 2 | twap60 | A_cal-global_benchmark | brier | -0.00040 | [-0.00081, +0.00005] | 54 | 18 | 15548 |
| 2 | all | A_cal-global_benchmark | hit | +0.00900 | [+0.00174, +0.01634] | 98 | 33 | 28216 |
| 2 | mixed | A_cal-global_benchmark | hit | +0.50000 | [+0.50000, +0.50000] | 2 | 1 | 2 |
| 2 | single_price | A_cal-global_benchmark | hit | +0.00873 | [-0.00282, +0.01971] | 37 | 13 | 10651 |
| 2 | twap30 | A_cal-global_benchmark | hit | +0.04864 | [+0.03474, +0.06207] | 7 | 3 | 2015 |
| 2 | twap60 | A_cal-global_benchmark | hit | +0.00399 | [-0.00593, +0.01260] | 54 | 18 | 15548 |
| 2 | all | B_cal-global_benchmark | log_loss | -0.00141 | [-0.00213, -0.00069] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-global_benchmark | log_loss | -0.03270 | [-0.03270, -0.03270] | 2 | 1 | 2 |
| 2 | single_price | B_cal-global_benchmark | log_loss | -0.00048 | [-0.00153, +0.00066] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-global_benchmark | log_loss | -0.00378 | [-0.00463, -0.00302] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-global_benchmark | log_loss | -0.00174 | [-0.00262, -0.00080] | 54 | 18 | 15548 |
| 2 | all | B_cal-global_benchmark | brier | -0.00070 | [-0.00105, -0.00034] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-global_benchmark | brier | -0.01630 | [-0.01630, -0.01630] | 2 | 1 | 2 |
| 2 | single_price | B_cal-global_benchmark | brier | -0.00023 | [-0.00075, +0.00035] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-global_benchmark | brier | -0.00189 | [-0.00229, -0.00153] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-global_benchmark | brier | -0.00086 | [-0.00130, -0.00040] | 54 | 18 | 15548 |
| 2 | all | B_cal-global_benchmark | hit | +0.01389 | [+0.00535, +0.02300] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-global_benchmark | hit | +0.50000 | [+0.50000, +0.50000] | 2 | 1 | 2 |
| 2 | single_price | B_cal-global_benchmark | hit | +0.00235 | [-0.00948, +0.01436] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-global_benchmark | hit | +0.05261 | [+0.03226, +0.07341] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-global_benchmark | hit | +0.01672 | [+0.00585, +0.02714] | 54 | 18 | 15548 |
| 2 | all | A_cal-conditional_benchmark | log_loss | -0.00155 | [-0.00221, -0.00086] | 98 | 33 | 28216 |
| 2 | mixed | A_cal-conditional_benchmark | log_loss | -0.03510 | [-0.03510, -0.03510] | 2 | 1 | 2 |
| 2 | single_price | A_cal-conditional_benchmark | log_loss | -0.00168 | [-0.00263, -0.00066] | 37 | 13 | 10651 |
| 2 | twap30 | A_cal-conditional_benchmark | log_loss | -0.00294 | [-0.00485, -0.00081] | 7 | 3 | 2015 |
| 2 | twap60 | A_cal-conditional_benchmark | log_loss | -0.00128 | [-0.00219, -0.00031] | 54 | 18 | 15548 |
| 2 | all | A_cal-conditional_benchmark | brier | -0.00077 | [-0.00110, -0.00042] | 98 | 33 | 28216 |
| 2 | mixed | A_cal-conditional_benchmark | brier | -0.01736 | [-0.01736, -0.01736] | 2 | 1 | 2 |
| 2 | single_price | A_cal-conditional_benchmark | brier | -0.00083 | [-0.00131, -0.00032] | 37 | 13 | 10651 |
| 2 | twap30 | A_cal-conditional_benchmark | brier | -0.00146 | [-0.00242, -0.00040] | 7 | 3 | 2015 |
| 2 | twap60 | A_cal-conditional_benchmark | brier | -0.00063 | [-0.00109, -0.00015] | 54 | 18 | 15548 |
| 2 | all | A_cal-conditional_benchmark | hit | +0.00985 | [+0.00205, +0.01715] | 98 | 33 | 28216 |
| 2 | mixed | A_cal-conditional_benchmark | hit | +0.50000 | [+0.50000, +0.50000] | 2 | 1 | 2 |
| 2 | single_price | A_cal-conditional_benchmark | hit | +0.00516 | [-0.00798, +0.01821] | 37 | 13 | 10651 |
| 2 | twap30 | A_cal-conditional_benchmark | hit | +0.02978 | [+0.00546, +0.05660] | 7 | 3 | 2015 |
| 2 | twap60 | A_cal-conditional_benchmark | hit | +0.01042 | [+0.00090, +0.01871] | 54 | 18 | 15548 |
| 2 | all | B_cal-conditional_benchmark | log_loss | -0.00172 | [-0.00240, -0.00105] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-conditional_benchmark | log_loss | -0.06675 | [-0.06675, -0.06675] | 2 | 1 | 2 |
| 2 | single_price | B_cal-conditional_benchmark | log_loss | -0.00071 | [-0.00170, +0.00030] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-conditional_benchmark | log_loss | -0.00334 | [-0.00510, -0.00159] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-conditional_benchmark | log_loss | -0.00220 | [-0.00306, -0.00133] | 54 | 18 | 15548 |
| 2 | all | B_cal-conditional_benchmark | brier | -0.00085 | [-0.00120, -0.00052] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-conditional_benchmark | brier | -0.03314 | [-0.03314, -0.03314] | 2 | 1 | 2 |
| 2 | single_price | B_cal-conditional_benchmark | brier | -0.00034 | [-0.00085, +0.00017] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-conditional_benchmark | brier | -0.00167 | [-0.00254, -0.00080] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-conditional_benchmark | brier | -0.00109 | [-0.00153, -0.00066] | 54 | 18 | 15548 |
| 2 | all | B_cal-conditional_benchmark | hit | +0.01474 | [+0.00532, +0.02428] | 98 | 33 | 28216 |
| 2 | mixed | B_cal-conditional_benchmark | hit | +0.50000 | [+0.50000, +0.50000] | 2 | 1 | 2 |
| 2 | single_price | B_cal-conditional_benchmark | hit | -0.00122 | [-0.01624, +0.01371] | 37 | 13 | 10651 |
| 2 | twap30 | B_cal-conditional_benchmark | hit | +0.03375 | [+0.00496, +0.05704] | 7 | 3 | 2015 |
| 2 | twap60 | B_cal-conditional_benchmark | hit | +0.02315 | [+0.01312, +0.03299] | 54 | 18 | 15548 |
| 3 | all | B_cal-A_cal | log_loss | -0.00041 | [-0.00106, +0.00023] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-A_cal | log_loss | -0.01011 | [-0.01011, -0.01011] | 2 | 1 | 4 |
| 3 | single_price | B_cal-A_cal | log_loss | +0.00075 | [-0.00014, +0.00160] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-A_cal | log_loss | -0.00061 | [-0.00208, +0.00109] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-A_cal | log_loss | -0.00118 | [-0.00196, -0.00040] | 54 | 18 | 15546 |
| 3 | all | B_cal-A_cal | brier | -0.00010 | [-0.00026, +0.00006] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-A_cal | brier | -0.00252 | [-0.00252, -0.00252] | 2 | 1 | 4 |
| 3 | single_price | B_cal-A_cal | brier | +0.00019 | [-0.00003, +0.00040] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-A_cal | brier | -0.00015 | [-0.00050, +0.00026] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-A_cal | brier | -0.00029 | [-0.00048, -0.00010] | 54 | 18 | 15546 |
| 3 | all | B_cal-A_cal | hit | -0.00305 | [-0.00844, +0.00234] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-A_cal | hit | +0.25000 | [+0.25000, +0.25000] | 2 | 1 | 4 |
| 3 | single_price | B_cal-A_cal | hit | -0.00911 | [-0.01792, -0.00056] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-A_cal | hit | -0.01787 | [-0.03323, +0.00248] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-A_cal | hit | +0.00296 | [-0.00418, +0.00966] | 54 | 18 | 15546 |
| 3 | all | A_cal-global_benchmark | log_loss | -0.00135 | [-0.00197, -0.00073] | 98 | 33 | 28212 |
| 3 | mixed | A_cal-global_benchmark | log_loss | -0.02147 | [-0.02147, -0.02147] | 2 | 1 | 4 |
| 3 | single_price | A_cal-global_benchmark | log_loss | -0.00137 | [-0.00243, -0.00023] | 37 | 13 | 10648 |
| 3 | twap30 | A_cal-global_benchmark | log_loss | -0.00383 | [-0.00492, -0.00282] | 7 | 3 | 2014 |
| 3 | twap60 | A_cal-global_benchmark | log_loss | -0.00101 | [-0.00170, -0.00026] | 54 | 18 | 15546 |
| 3 | all | A_cal-global_benchmark | brier | -0.00033 | [-0.00049, -0.00018] | 98 | 33 | 28212 |
| 3 | mixed | A_cal-global_benchmark | brier | -0.00532 | [-0.00532, -0.00532] | 2 | 1 | 4 |
| 3 | single_price | A_cal-global_benchmark | brier | -0.00034 | [-0.00060, -0.00005] | 37 | 13 | 10648 |
| 3 | twap30 | A_cal-global_benchmark | brier | -0.00096 | [-0.00122, -0.00071] | 7 | 3 | 2014 |
| 3 | twap60 | A_cal-global_benchmark | brier | -0.00024 | [-0.00042, -0.00006] | 54 | 18 | 15546 |
| 3 | all | A_cal-global_benchmark | hit | +0.00691 | [+0.00113, +0.01325] | 98 | 33 | 28212 |
| 3 | mixed | A_cal-global_benchmark | hit | -0.25000 | [-0.25000, -0.25000] | 2 | 1 | 4 |
| 3 | single_price | A_cal-global_benchmark | hit | +0.00479 | [-0.00423, +0.01390] | 37 | 13 | 10648 |
| 3 | twap30 | A_cal-global_benchmark | hit | +0.04022 | [+0.02087, +0.05952] | 7 | 3 | 2014 |
| 3 | twap60 | A_cal-global_benchmark | hit | +0.00412 | [-0.00354, +0.01171] | 54 | 18 | 15546 |
| 3 | all | B_cal-global_benchmark | log_loss | -0.00176 | [-0.00252, -0.00098] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-global_benchmark | log_loss | -0.03158 | [-0.03158, -0.03158] | 2 | 1 | 4 |
| 3 | single_price | B_cal-global_benchmark | log_loss | -0.00062 | [-0.00180, +0.00060] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-global_benchmark | log_loss | -0.00445 | [-0.00540, -0.00341] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-global_benchmark | log_loss | -0.00218 | [-0.00314, -0.00117] | 54 | 18 | 15546 |
| 3 | all | B_cal-global_benchmark | brier | -0.00043 | [-0.00062, -0.00024] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-global_benchmark | brier | -0.00784 | [-0.00784, -0.00784] | 2 | 1 | 4 |
| 3 | single_price | B_cal-global_benchmark | brier | -0.00015 | [-0.00044, +0.00015] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-global_benchmark | brier | -0.00111 | [-0.00133, -0.00087] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-global_benchmark | brier | -0.00053 | [-0.00077, -0.00028] | 54 | 18 | 15546 |
| 3 | all | B_cal-global_benchmark | hit | +0.00386 | [-0.00223, +0.00992] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-global_benchmark | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 4 |
| 3 | single_price | B_cal-global_benchmark | hit | -0.00432 | [-0.01380, +0.00639] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-global_benchmark | hit | +0.02234 | [+0.01190, +0.03477] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-global_benchmark | hit | +0.00708 | [+0.00019, +0.01402] | 54 | 18 | 15546 |
| 3 | all | A_cal-conditional_benchmark | log_loss | -0.00199 | [-0.00271, -0.00122] | 98 | 33 | 28212 |
| 3 | mixed | A_cal-conditional_benchmark | log_loss | -0.07091 | [-0.07091, -0.07091] | 2 | 1 | 4 |
| 3 | single_price | A_cal-conditional_benchmark | log_loss | -0.00239 | [-0.00353, -0.00121] | 37 | 13 | 10648 |
| 3 | twap30 | A_cal-conditional_benchmark | log_loss | -0.00300 | [-0.00590, -0.00069] | 7 | 3 | 2014 |
| 3 | twap60 | A_cal-conditional_benchmark | log_loss | -0.00157 | [-0.00257, -0.00051] | 54 | 18 | 15546 |
| 3 | all | A_cal-conditional_benchmark | brier | -0.00049 | [-0.00067, -0.00030] | 98 | 33 | 28212 |
| 3 | mixed | A_cal-conditional_benchmark | brier | -0.01745 | [-0.01745, -0.01745] | 2 | 1 | 4 |
| 3 | single_price | A_cal-conditional_benchmark | brier | -0.00059 | [-0.00087, -0.00030] | 37 | 13 | 10648 |
| 3 | twap30 | A_cal-conditional_benchmark | brier | -0.00074 | [-0.00146, -0.00017] | 7 | 3 | 2014 |
| 3 | twap60 | A_cal-conditional_benchmark | brier | -0.00038 | [-0.00063, -0.00012] | 54 | 18 | 15546 |
| 3 | all | A_cal-conditional_benchmark | hit | +0.00645 | [+0.00035, +0.01251] | 98 | 33 | 28212 |
| 3 | mixed | A_cal-conditional_benchmark | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 4 |
| 3 | single_price | A_cal-conditional_benchmark | hit | +0.00958 | [-0.00056, +0.02027] | 37 | 13 | 10648 |
| 3 | twap30 | A_cal-conditional_benchmark | hit | +0.02383 | [+0.00298, +0.04220] | 7 | 3 | 2014 |
| 3 | twap60 | A_cal-conditional_benchmark | hit | +0.00206 | [-0.00598, +0.01029] | 54 | 18 | 15546 |
| 3 | all | B_cal-conditional_benchmark | log_loss | -0.00240 | [-0.00319, -0.00164] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-conditional_benchmark | log_loss | -0.08103 | [-0.08103, -0.08103] | 2 | 1 | 4 |
| 3 | single_price | B_cal-conditional_benchmark | log_loss | -0.00165 | [-0.00271, -0.00057] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-conditional_benchmark | log_loss | -0.00362 | [-0.00520, -0.00200] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-conditional_benchmark | log_loss | -0.00274 | [-0.00395, -0.00165] | 54 | 18 | 15546 |
| 3 | all | B_cal-conditional_benchmark | brier | -0.00059 | [-0.00079, -0.00040] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-conditional_benchmark | brier | -0.01997 | [-0.01997, -0.01997] | 2 | 1 | 4 |
| 3 | single_price | B_cal-conditional_benchmark | brier | -0.00040 | [-0.00066, -0.00013] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-conditional_benchmark | brier | -0.00089 | [-0.00129, -0.00047] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-conditional_benchmark | brier | -0.00067 | [-0.00097, -0.00040] | 54 | 18 | 15546 |
| 3 | all | B_cal-conditional_benchmark | hit | +0.00340 | [-0.00192, +0.00869] | 98 | 33 | 28212 |
| 3 | mixed | B_cal-conditional_benchmark | hit | +0.25000 | [+0.25000, +0.25000] | 2 | 1 | 4 |
| 3 | single_price | B_cal-conditional_benchmark | hit | +0.00047 | [-0.00798, +0.00845] | 37 | 13 | 10648 |
| 3 | twap30 | B_cal-conditional_benchmark | hit | +0.00596 | [-0.01194, +0.02381] | 7 | 3 | 2014 |
| 3 | twap60 | B_cal-conditional_benchmark | hit | +0.00502 | [-0.00206, +0.01216] | 54 | 18 | 15546 |
| 4 | all | B_cal-A_cal | log_loss | -0.00063 | [-0.00140, +0.00011] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-A_cal | log_loss | -0.01914 | [-0.01914, -0.01914] | 2 | 1 | 6 |
| 4 | single_price | B_cal-A_cal | log_loss | +0.00046 | [-0.00061, +0.00145] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-A_cal | log_loss | -0.00179 | [-0.00401, +0.00075] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-A_cal | log_loss | -0.00122 | [-0.00210, -0.00032] | 54 | 18 | 15544 |
| 4 | all | B_cal-A_cal | brier | -0.00008 | [-0.00017, +0.00002] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-A_cal | brier | -0.00242 | [-0.00242, -0.00242] | 2 | 1 | 6 |
| 4 | single_price | B_cal-A_cal | brier | +0.00006 | [-0.00007, +0.00018] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-A_cal | brier | -0.00023 | [-0.00049, +0.00008] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-A_cal | brier | -0.00015 | [-0.00026, -0.00004] | 54 | 18 | 15544 |
| 4 | all | B_cal-A_cal | hit | +0.00294 | [-0.00128, +0.00716] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-A_cal | hit | +0.16667 | [+0.16667, +0.16667] | 2 | 1 | 6 |
| 4 | single_price | B_cal-A_cal | hit | +0.00432 | [-0.00254, +0.01174] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-A_cal | hit | -0.00099 | [-0.01438, +0.01244] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-A_cal | hit | +0.00244 | [-0.00335, +0.00791] | 54 | 18 | 15544 |
| 4 | all | A_cal-global_benchmark | log_loss | -0.00143 | [-0.00205, -0.00081] | 98 | 33 | 28208 |
| 4 | mixed | A_cal-global_benchmark | log_loss | -0.03523 | [-0.03523, -0.03523] | 2 | 1 | 6 |
| 4 | single_price | A_cal-global_benchmark | log_loss | -0.00112 | [-0.00206, -0.00019] | 37 | 13 | 10645 |
| 4 | twap30 | A_cal-global_benchmark | log_loss | -0.00390 | [-0.00607, -0.00205] | 7 | 3 | 2013 |
| 4 | twap60 | A_cal-global_benchmark | log_loss | -0.00131 | [-0.00209, -0.00054] | 54 | 18 | 15544 |
| 4 | all | A_cal-global_benchmark | brier | -0.00018 | [-0.00025, -0.00010] | 98 | 33 | 28208 |
| 4 | mixed | A_cal-global_benchmark | brier | -0.00432 | [-0.00432, -0.00432] | 2 | 1 | 6 |
| 4 | single_price | A_cal-global_benchmark | brier | -0.00014 | [-0.00025, -0.00002] | 37 | 13 | 10645 |
| 4 | twap30 | A_cal-global_benchmark | brier | -0.00048 | [-0.00074, -0.00025] | 7 | 3 | 2013 |
| 4 | twap60 | A_cal-global_benchmark | brier | -0.00016 | [-0.00026, -0.00007] | 54 | 18 | 15544 |
| 4 | all | A_cal-global_benchmark | hit | +0.00199 | [-0.00138, +0.00542] | 98 | 33 | 28208 |
| 4 | mixed | A_cal-global_benchmark | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 6 |
| 4 | single_price | A_cal-global_benchmark | hit | -0.00385 | [-0.00893, +0.00131] | 37 | 13 | 10645 |
| 4 | twap30 | A_cal-global_benchmark | hit | +0.01689 | [+0.01045, +0.02285] | 7 | 3 | 2013 |
| 4 | twap60 | A_cal-global_benchmark | hit | +0.00405 | [-0.00006, +0.00792] | 54 | 18 | 15544 |
| 4 | all | B_cal-global_benchmark | log_loss | -0.00206 | [-0.00306, -0.00106] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-global_benchmark | log_loss | -0.05437 | [-0.05437, -0.05437] | 2 | 1 | 6 |
| 4 | single_price | B_cal-global_benchmark | log_loss | -0.00066 | [-0.00218, +0.00086] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-global_benchmark | log_loss | -0.00569 | [-0.00696, -0.00402] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-global_benchmark | log_loss | -0.00253 | [-0.00380, -0.00122] | 54 | 18 | 15544 |
| 4 | all | B_cal-global_benchmark | brier | -0.00025 | [-0.00038, -0.00013] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-global_benchmark | brier | -0.00674 | [-0.00674, -0.00674] | 2 | 1 | 6 |
| 4 | single_price | B_cal-global_benchmark | brier | -0.00008 | [-0.00027, +0.00011] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-global_benchmark | brier | -0.00070 | [-0.00086, -0.00050] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-global_benchmark | brier | -0.00031 | [-0.00046, -0.00015] | 54 | 18 | 15544 |
| 4 | all | B_cal-global_benchmark | hit | +0.00493 | [-0.00011, +0.00996] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-global_benchmark | hit | +0.16667 | [+0.16667, +0.16667] | 2 | 1 | 6 |
| 4 | single_price | B_cal-global_benchmark | hit | +0.00047 | [-0.00893, +0.01053] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-global_benchmark | hit | +0.01590 | [+0.00694, +0.02493] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-global_benchmark | hit | +0.00650 | [+0.00064, +0.01209] | 54 | 18 | 15544 |
| 4 | all | A_cal-conditional_benchmark | log_loss | -0.00255 | [-0.00356, -0.00151] | 98 | 33 | 28208 |
| 4 | mixed | A_cal-conditional_benchmark | log_loss | -0.05498 | [-0.05498, -0.05498] | 2 | 1 | 6 |
| 4 | single_price | A_cal-conditional_benchmark | log_loss | -0.00346 | [-0.00493, -0.00198] | 37 | 13 | 10645 |
| 4 | twap30 | A_cal-conditional_benchmark | log_loss | -0.00453 | [-0.00845, -0.00154] | 7 | 3 | 2013 |
| 4 | twap60 | A_cal-conditional_benchmark | log_loss | -0.00164 | [-0.00300, -0.00025] | 54 | 18 | 15544 |
| 4 | all | A_cal-conditional_benchmark | brier | -0.00031 | [-0.00044, -0.00018] | 98 | 33 | 28208 |
| 4 | mixed | A_cal-conditional_benchmark | brier | -0.00697 | [-0.00697, -0.00697] | 2 | 1 | 6 |
| 4 | single_price | A_cal-conditional_benchmark | brier | -0.00042 | [-0.00061, -0.00024] | 37 | 13 | 10645 |
| 4 | twap30 | A_cal-conditional_benchmark | brier | -0.00055 | [-0.00103, -0.00019] | 7 | 3 | 2013 |
| 4 | twap60 | A_cal-conditional_benchmark | brier | -0.00020 | [-0.00037, -0.00003] | 54 | 18 | 15544 |
| 4 | all | A_cal-conditional_benchmark | hit | +0.00280 | [-0.00053, +0.00603] | 98 | 33 | 28208 |
| 4 | mixed | A_cal-conditional_benchmark | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 6 |
| 4 | single_price | A_cal-conditional_benchmark | hit | +0.00197 | [-0.00319, +0.00676] | 37 | 13 | 10645 |
| 4 | twap30 | A_cal-conditional_benchmark | hit | +0.01093 | [-0.00348, +0.02285] | 7 | 3 | 2013 |
| 4 | twap60 | A_cal-conditional_benchmark | hit | +0.00232 | [-0.00199, +0.00637] | 54 | 18 | 15544 |
| 4 | all | B_cal-conditional_benchmark | log_loss | -0.00318 | [-0.00429, -0.00204] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-conditional_benchmark | log_loss | -0.07411 | [-0.07411, -0.07411] | 2 | 1 | 6 |
| 4 | single_price | B_cal-conditional_benchmark | log_loss | -0.00300 | [-0.00468, -0.00133] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-conditional_benchmark | log_loss | -0.00633 | [-0.00878, -0.00399] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-conditional_benchmark | log_loss | -0.00286 | [-0.00450, -0.00136] | 54 | 18 | 15544 |
| 4 | all | B_cal-conditional_benchmark | brier | -0.00039 | [-0.00053, -0.00024] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-conditional_benchmark | brier | -0.00939 | [-0.00939, -0.00939] | 2 | 1 | 6 |
| 4 | single_price | B_cal-conditional_benchmark | brier | -0.00037 | [-0.00058, -0.00015] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-conditional_benchmark | brier | -0.00078 | [-0.00109, -0.00048] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-conditional_benchmark | brier | -0.00035 | [-0.00055, -0.00016] | 54 | 18 | 15544 |
| 4 | all | B_cal-conditional_benchmark | hit | +0.00574 | [+0.00074, +0.01046] | 98 | 33 | 28208 |
| 4 | mixed | B_cal-conditional_benchmark | hit | +0.16667 | [+0.16667, +0.16667] | 2 | 1 | 6 |
| 4 | single_price | B_cal-conditional_benchmark | hit | +0.00629 | [-0.00272, +0.01484] | 37 | 13 | 10645 |
| 4 | twap30 | B_cal-conditional_benchmark | hit | +0.00994 | [-0.00099, +0.02541] | 7 | 3 | 2013 |
| 4 | twap60 | B_cal-conditional_benchmark | hit | +0.00476 | [-0.00180, +0.01062] | 54 | 18 | 15544 |
| 5 | all | B_cal-A_cal | log_loss | -0.00082 | [-0.00161, -0.00002] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-A_cal | log_loss | -0.02704 | [-0.02704, -0.02704] | 2 | 1 | 8 |
| 5 | single_price | B_cal-A_cal | log_loss | +0.00030 | [-0.00107, +0.00163] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-A_cal | log_loss | -0.00098 | [-0.00294, +0.00119] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-A_cal | log_loss | -0.00155 | [-0.00245, -0.00063] | 54 | 18 | 15542 |
| 5 | all | B_cal-A_cal | brier | -0.00005 | [-0.00010, +0.00000] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-A_cal | brier | -0.00169 | [-0.00169, -0.00169] | 2 | 1 | 8 |
| 5 | single_price | B_cal-A_cal | brier | +0.00002 | [-0.00006, +0.00010] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-A_cal | brier | -0.00006 | [-0.00018, +0.00006] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-A_cal | brier | -0.00009 | [-0.00015, -0.00004] | 54 | 18 | 15542 |
| 5 | all | B_cal-A_cal | hit | +0.00262 | [-0.00099, +0.00634] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-A_cal | hit | +0.12500 | [+0.12500, +0.12500] | 2 | 1 | 8 |
| 5 | single_price | B_cal-A_cal | hit | +0.00141 | [-0.00480, +0.00847] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-A_cal | hit | -0.00447 | [-0.00944, +0.00149] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-A_cal | hit | +0.00431 | [-0.00013, +0.00868] | 54 | 18 | 15542 |
| 5 | all | A_cal-global_benchmark | log_loss | -0.00126 | [-0.00195, -0.00050] | 98 | 33 | 28204 |
| 5 | mixed | A_cal-global_benchmark | log_loss | -0.02849 | [-0.02849, -0.02849] | 2 | 1 | 8 |
| 5 | single_price | A_cal-global_benchmark | log_loss | -0.00095 | [-0.00224, +0.00040] | 37 | 13 | 10642 |
| 5 | twap30 | A_cal-global_benchmark | log_loss | -0.00430 | [-0.00708, -0.00189] | 7 | 3 | 2012 |
| 5 | twap60 | A_cal-global_benchmark | log_loss | -0.00105 | [-0.00183, -0.00019] | 54 | 18 | 15542 |
| 5 | all | A_cal-global_benchmark | brier | -0.00008 | [-0.00012, -0.00003] | 98 | 33 | 28204 |
| 5 | mixed | A_cal-global_benchmark | brier | -0.00171 | [-0.00171, -0.00171] | 2 | 1 | 8 |
| 5 | single_price | A_cal-global_benchmark | brier | -0.00006 | [-0.00014, +0.00002] | 37 | 13 | 10642 |
| 5 | twap30 | A_cal-global_benchmark | brier | -0.00026 | [-0.00043, -0.00012] | 7 | 3 | 2012 |
| 5 | twap60 | A_cal-global_benchmark | brier | -0.00007 | [-0.00011, -0.00001] | 54 | 18 | 15542 |
| 5 | all | A_cal-global_benchmark | hit | -0.00128 | [-0.00344, +0.00082] | 98 | 33 | 28204 |
| 5 | mixed | A_cal-global_benchmark | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 8 |
| 5 | single_price | A_cal-global_benchmark | hit | -0.00197 | [-0.00564, +0.00160] | 37 | 13 | 10642 |
| 5 | twap30 | A_cal-global_benchmark | hit | +0.00746 | [+0.00298, +0.01195] | 7 | 3 | 2012 |
| 5 | twap60 | A_cal-global_benchmark | hit | -0.00193 | [-0.00463, +0.00077] | 54 | 18 | 15542 |
| 5 | all | B_cal-global_benchmark | log_loss | -0.00207 | [-0.00314, -0.00098] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-global_benchmark | log_loss | -0.05553 | [-0.05553, -0.05553] | 2 | 1 | 8 |
| 5 | single_price | B_cal-global_benchmark | log_loss | -0.00065 | [-0.00248, +0.00105] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-global_benchmark | log_loss | -0.00528 | [-0.00728, -0.00298] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-global_benchmark | log_loss | -0.00261 | [-0.00396, -0.00123] | 54 | 18 | 15542 |
| 5 | all | B_cal-global_benchmark | brier | -0.00013 | [-0.00019, -0.00006] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-global_benchmark | brier | -0.00339 | [-0.00339, -0.00339] | 2 | 1 | 8 |
| 5 | single_price | B_cal-global_benchmark | brier | -0.00004 | [-0.00015, +0.00007] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-global_benchmark | brier | -0.00033 | [-0.00045, -0.00019] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-global_benchmark | brier | -0.00016 | [-0.00024, -0.00007] | 54 | 18 | 15542 |
| 5 | all | B_cal-global_benchmark | hit | +0.00135 | [-0.00170, +0.00454] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-global_benchmark | hit | +0.12500 | [+0.12500, +0.12500] | 2 | 1 | 8 |
| 5 | single_price | B_cal-global_benchmark | hit | -0.00056 | [-0.00649, +0.00610] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-global_benchmark | hit | +0.00298 | [-0.00099, +0.00699] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-global_benchmark | hit | +0.00238 | [-0.00116, +0.00579] | 54 | 18 | 15542 |
| 5 | all | A_cal-conditional_benchmark | log_loss | -0.00373 | [-0.00507, -0.00237] | 98 | 33 | 28204 |
| 5 | mixed | A_cal-conditional_benchmark | log_loss | -0.03331 | [-0.03331, -0.03331] | 2 | 1 | 8 |
| 5 | single_price | A_cal-conditional_benchmark | log_loss | -0.00527 | [-0.00724, -0.00331] | 37 | 13 | 10642 |
| 5 | twap30 | A_cal-conditional_benchmark | log_loss | -0.00635 | [-0.01026, -0.00321] | 7 | 3 | 2012 |
| 5 | twap60 | A_cal-conditional_benchmark | log_loss | -0.00231 | [-0.00396, -0.00064] | 54 | 18 | 15542 |
| 5 | all | A_cal-conditional_benchmark | brier | -0.00023 | [-0.00031, -0.00015] | 98 | 33 | 28204 |
| 5 | mixed | A_cal-conditional_benchmark | brier | -0.00206 | [-0.00206, -0.00206] | 2 | 1 | 8 |
| 5 | single_price | A_cal-conditional_benchmark | brier | -0.00033 | [-0.00045, -0.00021] | 37 | 13 | 10642 |
| 5 | twap30 | A_cal-conditional_benchmark | brier | -0.00040 | [-0.00064, -0.00019] | 7 | 3 | 2012 |
| 5 | twap60 | A_cal-conditional_benchmark | brier | -0.00014 | [-0.00024, -0.00004] | 54 | 18 | 15542 |
| 5 | all | A_cal-conditional_benchmark | hit | +0.00007 | [-0.00319, +0.00337] | 98 | 33 | 28204 |
| 5 | mixed | A_cal-conditional_benchmark | hit | -0.12500 | [-0.12500, -0.12500] | 2 | 1 | 8 |
| 5 | single_price | A_cal-conditional_benchmark | hit | +0.00244 | [-0.00207, +0.00659] | 37 | 13 | 10642 |
| 5 | twap30 | A_cal-conditional_benchmark | hit | +0.01093 | [+0.00000, +0.02291] | 7 | 3 | 2012 |
| 5 | twap60 | A_cal-conditional_benchmark | hit | -0.00290 | [-0.00721, +0.00129] | 54 | 18 | 15542 |
| 5 | all | B_cal-conditional_benchmark | log_loss | -0.00454 | [-0.00592, -0.00320] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-conditional_benchmark | log_loss | -0.06034 | [-0.06034, -0.06034] | 2 | 1 | 8 |
| 5 | single_price | B_cal-conditional_benchmark | log_loss | -0.00497 | [-0.00697, -0.00300] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-conditional_benchmark | log_loss | -0.00734 | [-0.01033, -0.00433] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-conditional_benchmark | log_loss | -0.00386 | [-0.00592, -0.00207] | 54 | 18 | 15542 |
| 5 | all | B_cal-conditional_benchmark | brier | -0.00028 | [-0.00036, -0.00020] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-conditional_benchmark | brier | -0.00375 | [-0.00375, -0.00375] | 2 | 1 | 8 |
| 5 | single_price | B_cal-conditional_benchmark | brier | -0.00031 | [-0.00043, -0.00019] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-conditional_benchmark | brier | -0.00046 | [-0.00065, -0.00027] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-conditional_benchmark | brier | -0.00024 | [-0.00036, -0.00012] | 54 | 18 | 15542 |
| 5 | all | B_cal-conditional_benchmark | hit | +0.00269 | [-0.00046, +0.00581] | 98 | 33 | 28204 |
| 5 | mixed | B_cal-conditional_benchmark | hit | +0.00000 | [+0.00000, +0.00000] | 2 | 1 | 8 |
| 5 | single_price | B_cal-conditional_benchmark | hit | +0.00385 | [+0.00000, +0.00780] | 37 | 13 | 10642 |
| 5 | twap30 | B_cal-conditional_benchmark | hit | +0.00646 | [-0.00198, +0.01594] | 7 | 3 | 2012 |
| 5 | twap60 | B_cal-conditional_benchmark | hit | +0.00142 | [-0.00361, +0.00598] | 54 | 18 | 15542 |

## Selective analysis (frozen thresholds)

| n | variant | threshold | selected | rate | hit | log loss | global hit same rows |
|---|---|---|---|---|---|---|---|
| 1 | A | 10pct | 7460 | 0.2644 | 0.5294906166219839 | 0.6916325125254296 | 0.5024128686327077 |
| 1 | A | 1pct | 1622 | 0.0575 | 0.5345252774352651 | 0.6913268545111995 | 0.4907521578298397 |
| 1 | B | 10pct | 3910 | 0.1386 | 0.5301790281329923 | 0.691723204198087 | 0.5020460358056266 |
| 1 | B | 1pct | 319 | 0.0113 | 0.5517241379310345 | 0.6878265001368664 | 0.5768025078369906 |
| 1 | A and B | 10pct | 2431 | 0.0861 | A 0.5294117647058824 / B 0.530234471410942 | | |
| 1 | A and B | 1pct | 130 | 0.0046 | A 0.5461538461538461 / B 0.5461538461538461 | | |
| 2 | A | 10pct | 10371 | 0.3676 | 0.2706585671584225 | 1.383718508005086 | 0.24934914665895286 |
| 2 | A | 1pct | 3854 | 0.1366 | 0.2732226258432797 | 1.383159348220934 | 0.2537623248572911 |
| 2 | B | 10pct | 0 | 0.0000 | nan | nan | nan |
| 2 | B | 1pct | 0 | 0.0000 | nan | nan | nan |
| 2 | A and B | 10pct | 0 | 0.0000 | A nan / B nan | | |
| 2 | A and B | 1pct | 0 | 0.0000 | A nan / B nan | | |
| 3 | A | 10pct | 4616 | 0.1636 | 0.14081455805892548 | 2.0767301136671494 | 0.12326689774696707 |
| 3 | A | 1pct | 139 | 0.0049 | 0.1223021582733813 | 2.0749797971674626 | 0.12949640287769784 |
| 3 | B | 10pct | 0 | 0.0000 | nan | nan | nan |
| 3 | B | 1pct | 0 | 0.0000 | nan | nan | nan |
| 3 | A and B | 10pct | 0 | 0.0000 | A nan / B nan | | |
| 3 | A and B | 1pct | 0 | 0.0000 | A nan / B nan | | |
| 4 | A | 10pct | 1460 | 0.0518 | 0.057534246575342465 | 2.7702303416277574 | 0.052054794520547946 |
| 4 | A | 1pct | 8 | 0.0003 | 0.0 | 2.7938041359273216 | 0.0 |
| 4 | B | 10pct | 1 | 0.0000 | 0.0 | 2.7822482892903353 | 0.0 |
| 4 | B | 1pct | 0 | 0.0000 | nan | nan | nan |
| 4 | A and B | 10pct | 0 | 0.0000 | A nan / B nan | | |
| 4 | A and B | 1pct | 0 | 0.0000 | A nan / B nan | | |
| 5 | A | 10pct | 198 | 0.0070 | 0.03535353535353535 | 3.4618956741749742 | 0.030303030303030304 |
| 5 | A | 1pct | 1 | 0.0000 | 0.0 | 3.463097803001797 | 0.0 |
| 5 | B | 10pct | 0 | 0.0000 | nan | nan | nan |
| 5 | B | 1pct | 0 | 0.0000 | nan | nan | nan |
| 5 | A and B | 10pct | 0 | 0.0000 | A nan / B nan | | |
| 5 | A and B | 1pct | 0 | 0.0000 | A nan / B nan | | |

## Manifest

```json
{
  "bundle_sha256": "04db94470c373b6f5d5efc946c34ec1edd6a64675f611f83c31add41457a49b2",
  "official_counts": {
    "rows_read": 79680,
    "first_market_start_utc": "2025-12-18T04:25:00+00:00",
    "last_market_start_utc": "2026-10-06T23:55:00+00:00",
    "series_ids": [
      "10684"
    ],
    "outcome_missing_or_unverified": 14,
    "outcome_target_onehot_disagree": 0,
    "outcome_known_resolved_at_missing": 14
  },
  "official_file_sha256": "cc314da279bddf5df4204cc0ff607091f85a985136ac7e7edd1308abcf1b65ba",
  "dev_part_identical_to_export": true,
  "holdout_scored_sha256": "7349b71f3fb0082c6652491d0d56d0596856b0491ce75a3c1a787c2caa96d4d1",
  "python": "3.14.5 (tags/v3.14.5:5607950, May 10 2026, 10:43:50) [MSC v.1944 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26300-SP0",
  "elapsed_seconds": 4.0
}
```
