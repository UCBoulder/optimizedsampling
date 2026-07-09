# Summarize

Parse experiment logs from `sampling/` into CSVs, then generate tables/figures.

```bash
python parse_out_log.py --multiple True   # parse logs into CSVs
python generate_latex_table.py            # main results table
python plot_multiple_initial_set.py       # initial-set size sweep figure
python plot_alpha.py                      # cost-sensitivity figure
```
