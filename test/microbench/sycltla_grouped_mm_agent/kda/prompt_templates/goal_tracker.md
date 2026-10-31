# Goal Tracker

This tracker keeps the loop aligned with the plan. The IMMUTABLE SECTION is frozen after Round 0 (the driver restores it if a later round edits it). Keep the MUTABLE SECTION current every round.

## IMMUTABLE SECTION

### Ultimate Goal

{{GOAL}}

### Acceptance Criteria

{{ACS}}

---

## MUTABLE SECTION

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
| Task | Target AC | Status | Lane | Notes |
|------|-----------|--------|------|-------|

### Direction Ledger
| Direction | Status (untried/in-progress/kept/revised/rejected) | Iterations used (max 5) | Best result vs production | Evidence (benchmark.csv ids, profile dir) |
|-----------|----------------------------------------------------|-------------------------|---------------------------|--------------------------------------------|

### Blocking Side Issues
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|

### Completed and Verified
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|--------------------|
