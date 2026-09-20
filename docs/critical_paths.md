# Critical paths, and the tests that pin them

Coverage is a floor here, not a target (`pyproject.toml`: 85%, set five points
under the measured 88% with branches on 2026-09-12). The number an interviewer
should ask about is not the total but whether the paths where this system can
be *quietly wrong* are pinned by a test that would fail. This is that list,
with the tests by name, and the honest remainder: what is not covered and why.
CI publishes the per-file line and branch table in each run's job summary and
the HTML report as an artifact (`coverage-report`).

| critical path | what a silent failure would look like | pinned by |
|---|---|---|
| **Label construction** — a positive is a posting absent from two consecutive complete runs *after its first sighting*, final at corroboration | a late-arriving posting labelled removed before it existed; a single absence counted; a reappearance revising a final label | `test_assemble.py`: `test_removal_corroborated_by_two_absences_is_a_positive`, `test_a_single_absence_is_not_a_removal`, `test_absences_before_first_sight_are_not_a_closure`, `test_a_label_is_final_once_corroborated_even_if_the_posting_returns`, `test_the_newest_labelled_wave_can_never_carry_a_positive`, `test_seconds_of_clock_drift_do_not_decide_the_label` |
| **Right-censoring** — an unobservable outcome is dropped, never zero | recent postings taught as "stays open" | `test_assemble.py`: `test_a_closure_inside_an_unelapsed_window_is_dropped_too`; `test_preprocessing.py`: `test_unlabelled_rows_are_refused_before_fitting` |
| **The temporal split and the embargo** | validation rows whose label was computed from training-time futures; a shuffled split | `test_split.py`: `test_blocks_are_strictly_ordered_in_time_and_clear_the_embargo`, `test_embargo_uses_the_worst_run_gap_not_the_typical_one`, `test_a_single_late_crawl_widens_the_embargo_for_the_whole_panel`, `test_assignment_ignores_row_order_entirely`, `test_minimum_waves_counts_what_the_embargo_burns` |
| **The test block is opened once, in one place** | a module reading `split.test` for tuning | `test_evaluate.py`: `test_the_test_block_is_read_only_where_it_should_be` (an AST walk of `src/`), `test_the_shell_scripts_do_not_read_the_test_block_either`; `test_deploy.py`: `test_the_rehearsal_never_opens_the_test_block`, `test_the_regeneration_refuses_dirty_source_and_never_opens_the_block_itself` |
| **Leakage enforcement** — every column has a verdict; transformers fit on the training fold only | a new panel column reaching the model unaudited; an imputer learning the full frame's median | `test_preprocessing.py`: `test_every_panel_column_has_exactly_one_verdict`, `test_a_column_with_no_verdict_is_refused`, `test_imputer_learns_the_training_folds_median_not_the_full_frames`, `test_fit_on_training_fold_cannot_see_validation_or_test`, `test_a_category_unseen_at_fit_time_does_not_raise` |
| **Threshold selection** — the budget-th validation score, never 0.5 | a threshold chosen on test, or a budget the sweep does not cover | `test_evaluate.py`: `test_threshold_for_budget_on_the_hand_computed_array`, `test_the_sweep_covers_the_chosen_budget_so_the_choice_is_defended`; `test_metrics.py` |
| **Model selection by the pre-registered rule** | a winner named with no folds; complexity read as a selection | `test_evaluate.py`: `test_selection_refuses_a_candidate_with_too_few_folds`, `test_selection_says_so_when_nothing_could_be_scored`, `test_a_gap_inside_one_standard_deviation_hands_the_pick_to_the_simpler_model`, `test_the_heuristic_floor_is_a_gate_on_the_pick_not_only_on_the_leader`, `test_a_fitted_pick_that_clears_the_floor_carries_the_measurement_that_says_so`, `test_the_complexity_reading_never_reads_as_a_selection` |
| **Recalibration, transfer and board-context rules** — decided before the numbers | a rule that fires on the wrong comparison; a monotone map reordering the alert list | `test_calibration.py` (fires where it says; ranks survive; the alert list never loses a member); `test_generalisation.py` (collapse needs two boards; serve-time withholding); `test_board_context.py` (imputed vs refit, not supplied vs refit) |
| **The freeze's refusals** | a freeze on a shallow split, with no folds, on a collapsed candidate, or from a dirty tree | `test_freeze.py`: `test_a_legal_split_with_no_folds_is_refused`, `test_a_candidate_that_collapses_on_unseen_boards_is_refused`, `test_the_transfer_check_never_opens_the_test_block`, `test_a_real_freeze_refuses_a_dirty_worktree`, `test_the_clean_tree_check_comes_after_the_depth_checks`; each override exists, works, and is recorded |
| **Artifact compatibility and traceability** | a bare estimator saved; a booster read by a different XGBoost; a hash naming a file nobody has | `test_inference.py`: `test_a_fixed_posting_scores_the_same_number_forever`, `test_loading_a_missing_artifact_says_how_to_build_one`; `test_freeze.py`: `test_the_artifact_names_everything_it_is_traceable_to`; `test_deploy.py`: `test_the_image_refuses_an_artifact_from_a_different_library`, `test_the_lock_is_committed_and_nothing_ignores_it` |
| **Training/serving skew** — one fitted pipeline, one derivation | `/rank` and `/predict` disagreeing; a client re-deriving board context | `test_inference.py`: `test_a_posting_scores_the_same_through_rank_as_through_predict`; `test_board_snapshot.py`: `test_the_derivation_is_the_panels_own_arithmetic`; `test_app.py`: `test_the_client_holds_no_copy_of_the_ranking_rule_or_the_derivation`, `test_the_ui_calls_the_api_and_never_the_model` |
| **`/rank` and the board flow** | a page's ranking mistaken for the board's; a budget that flags the wrong count | `test_boards.py`: `test_a_paged_board_ranks_as_one_call_would`, `test_a_snapshot_board_derives_context_over_the_whole_board`; `test_api.py`: `test_the_budget_decides_how_many_are_watched`, `test_a_board_uploaded_in_pages_ranks_as_one_batch_would`, `test_client_ids_come_back_on_the_matching_ranked_posting` |
| **Release loading** — fetch by tag, verify by checksum, load strictly | a truncated download served; a checksum mismatch passed | `test_fetch.py`: `test_a_file_that_does_not_match_its_published_checksum_is_refused`, `test_nothing_is_written_when_any_file_fails_verification`, `test_an_asset_with_no_checksum_entry_is_an_error_not_a_pass` |
| **Deployment contracts** — the workflow, the Dockerfile, the smoke tests, the UI's requirements | a smoke test asking for a renamed field; a build filter skipping `MODEL_TAG`; the UI importing the model | `test_deploy.py`: `test_the_smoke_test_asks_for_fields_the_api_actually_returns`, `test_the_smoke_test_exercises_rank_the_way_an_operator_would`, `test_the_build_filter_never_ignores_what_changes_the_image`, `test_every_environment_installs_from_the_lock`, `test_the_ui_check_reads_the_sentences_the_app_actually_writes`; `test_app.py`: `test_the_ui_requirements_cover_everything_the_ui_imports` |
| **Provenance and the reports** | a report without a snapshot or commit; a README number behind the reports | `test_reports.py`: `test_every_generated_report_identifies_its_provenance`, `test_a_report_about_a_pinned_snapshot_names_that_snapshot_not_the_newest`; `test_readme_summary.py`: `test_the_readme_block_matches_the_committed_reports` |

## What is not covered, and why

Measured on 2026-09-12 at 88% of statements and branches over `src/`, `api/`
and `app/`, and re-measured on 2026-09-20 at 92% after the five files below
were given tests that need no scraper snapshot, no network and no Linux
kernel — a miniature SQLite database, a server on localhost, and hand-written
`/proc` files. What remains uncovered, and the reason each line is where it is:

| file | 09-12 | 09-20 | what is still uncovered, and why |
|---|---|---|---|
| `src/data/profile.py` | 0% | 85% | the coverage fingerprint, run completeness, panel shape, the markdown renderer and the report's determinism are now pinned on the same miniature snapshot `test_load.py` builds (`test_profile.py`); what remains is the figure branch, which renders with matplotlib and logs to MLflow — both side effects on the working tree |
| `src/data/label_check.py` | 68% | 99% | the fetcher is exercised against a server on localhost (200 with body, 404 as a status, not an exception), the sampler's proportional draw, the "no instrument" verdict and the whole report — relisted titles counted as removals, unverifiable rows excluded from the rate — are pinned (`test_label_check.py`); the one line left is the `--delay` sleep between live requests |
| `api/runtime.py` | 74% | 100% | the `/proc` arithmetic — field 22 after the *last* `)`, since a command name may contain one — and the kB/bytes disagreement in `ru_maxrss` are exercised with fake files on the macOS runner (`test_runtime.py`) |
| `src/models/evaluate.py` | 78% | 91% | the selection rule's remaining branches are now pinned: a sole eligible candidate, a leader that is already the simplest of its tied set, and **the step-4 heuristic-floor gate** on a geometry where parsimony alone would ship a fitted model that never beat a constant (`test_evaluate.py`); the figure writer and the leave-one-board-out section too. Left: `_draw`'s MLflow logging and a handful of report sentences the `main()` driver assembles only on a real panel |
| `src/plots.py` | omitted | omitted | matplotlib figures; a plot is checked by looking at it |

Nothing in that list was a path where the system could be quietly wrong about
a label, a split, a threshold, or a served number — with one exception worth
naming: the heuristic-floor gate in `select` *is* such a path, and until
2026-09-20 its only evidence was a sentence in a docstring. The total is still
not the point; the floor stays at 85%.
