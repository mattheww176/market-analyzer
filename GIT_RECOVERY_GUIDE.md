# Git File Recovery Guide 🚨

**EMERGENCY FILE RECOVERY - READ THIS IF YOU LOST A FILE!**

## Quick Recovery Commands

### 1. File Just Deleted (Not Committed Yet)
```bash
# Recover specific file
git checkout HEAD -- filename.py

# Recover ALL deleted files
git checkout HEAD -- .

# Example:
git checkout HEAD -- stock_project7.py
```

### 2. File Deleted and Already Committed
```bash
# See recent commits
git log --oneline

# Recover from previous commit
git checkout HEAD~1 -- filename.py

# Example:
git checkout HEAD~1 -- stock_project7.py
```

---

## Complete Git Tutorial for Beginners

### What is Git?
Git is like a **time machine** for your files. Every time you "commit" changes, Git takes a snapshot of ALL your files. You can go back to any snapshot anytime.

### Daily Git Workflow (2 Commands Only!)

**After making changes to your files:**
```bash
# 1. Stage your changes (prepare for snapshot)
git add .

# 2. Take the snapshot (commit)
git commit -m "Describe what you changed"
```

**Example:**
```bash
git add .
git commit -m "Added new technical indicators to stock analysis"
```

### Essential Recovery Commands

#### Check Status
```bash
# See what files have changed
git status

# See what specific changes were made
git diff
```

#### View History
```bash
# See all commits (snapshots)
git log --oneline

# See history of a specific file
git log --follow -- filename.py
```

#### Recovery Scenarios

**Scenario 1: Accidentally deleted a file**
```bash
git status                    # Shows "deleted: filename.py"
git checkout HEAD -- filename.py    # Recovers the file
```

**Scenario 2: File was deleted days ago**
```bash
git log --oneline            # Find commit before deletion
git checkout abc1234 -- filename.py    # Use actual commit hash
```

**Scenario 3: Want to see old version without recovering**
```bash
git show HEAD~1:filename.py    # Show file from 1 commit ago
git show HEAD~5:filename.py    # Show file from 5 commits ago
```

**Scenario 4: Committed deletion by mistake**
```bash
git reset --hard HEAD~1      # DANGER: Undoes last commit entirely
```

### Advanced Recovery

#### Find When File Was Deleted
```bash
git log --diff-filter=D --summary
```

#### Recover Specific Version
```bash
# List all commits that touched this file
git log --oneline -- filename.py

# Recover from specific commit
git checkout COMMIT_HASH -- filename.py
```

#### Recover to Different Name
```bash
git show HEAD:filename.py > recovered_filename.py
```

### Git Best Practices

#### Daily Habits
1. **Commit often** - at least once per day
2. **Use descriptive messages** - "Fixed bug in RSI calculation" not "changes"
3. **Check status before committing** - `git status`

#### Good Commit Messages
```bash
git commit -m "Added Bollinger Bands indicator"
git commit -m "Fixed MACD calculation bug"
git commit -m "Added email report functionality"
```

#### Bad Commit Messages
```bash
git commit -m "stuff"
git commit -m "changes"
git commit -m "update"
```

### Emergency Commands Reference

| Situation | Command |
|-----------|---------|
| Just deleted file | `git checkout HEAD -- filename.py` |
| Deleted file yesterday | `git checkout HEAD~1 -- filename.py` |
| See what changed | `git status` |
| See commit history | `git log --oneline` |
| Undo last commit | `git reset --hard HEAD~1` |
| See old file version | `git show HEAD~1:filename.py` |

### Understanding Git Terms

- **Repository (repo)**: Your project folder with git tracking
- **Commit**: A snapshot of all your files at a point in time
- **HEAD**: The current commit (latest snapshot)
- **HEAD~1**: One commit before current
- **HEAD~5**: Five commits before current
- **Staging**: Preparing files for commit with `git add`

### Visual Timeline Example

```
HEAD~3 ← HEAD~2 ← HEAD~1 ← HEAD (current)
  |        |        |        |
Day 1    Day 2    Day 3    Today
```

You can recover files from ANY of these points!

### Troubleshooting

#### "Not a git repository" error
```bash
cd /path/to/your/project
git init
git add .
git commit -m "Initial commit"
```

#### "Nothing to commit" message
This means no files have changed since last commit. This is normal!

#### Can't find deleted file
```bash
# Search for file in all commits
git log --all --full-history -- filename.py
```

### Setup Reminder

**If git isn't set up yet:**
```bash
cd your-project-folder
git init
git add .
git commit -m "Initial commit"
```

---

## 🆘 EMERGENCY CHECKLIST

**File is missing? Follow these steps:**

1. ✅ `cd` to your project directory
2. ✅ Run `git status` - does it show "deleted: filename.py"?
3. ✅ If yes: `git checkout HEAD -- filename.py`
4. ✅ If no: `git log --oneline` then `git checkout HEAD~1 -- filename.py`
5. ✅ Check if file is back: `ls -la filename.py`

**Still can't find it?**
- Try `git log --follow -- filename.py`
- Look for the file in different commits
- Use `git checkout COMMIT_HASH -- filename.py`

---

## Remember: Git Only Protects Committed Files!

**Files are ONLY safe if you've committed them at least once.**

Make it a habit:
```bash
git add .
git commit -m "Daily backup"
```

**Do this every day you work on your project!**
