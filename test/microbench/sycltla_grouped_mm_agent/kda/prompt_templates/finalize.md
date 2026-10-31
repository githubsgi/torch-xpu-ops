The code review found blocking issues. Fix ONLY the [P0]/[P1] findings below in `cand/`, keep the optimisation, then run `ggemm-eval run cand --suite quick --verify-only` and `ggemm-eval run cand --suite quick` to prove correctness and that the score did not regress, record it with `kda-record bench --id final-fix ...`, and finish with a 3-line message.

<!-- FINDINGS -->
{{FINDINGS}}
<!-- END -->
